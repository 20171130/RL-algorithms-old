from copy import deepcopy
from models import *
"""
    Hierarchiy:
        algorithm
            batchsize
            the number of updates per interaction
            preprocessing the env    
            Both the agent and the model do not care about the tensor shapes or model architecture
        agent
            contains models
            An agent exposes:
                .act() does the sampling
                .update_x(batch), x = p, q, pi
        (abstract) models
            q 
                takes the env, and kwargs
                when continous, = Q(s, a) 
                when discrete, 
                    Q returns Q for all/average/an action,
                    depending on the input action
            pi returns a distribution
        network
            CNN, MLP, ...
"""

class QLearning(nn.Module):
    """ Double Dueling clipped (from TD3) Q Learning"""
    def __init__(self, logger, env_fn, q_args, gamma, eps, target_sync_rate, **kwargs):
        """
            q_net is the network class
        """
        super().__init__()
        self.logger = logger
        self.gamma = gamma
        self.target_sync_rate=target_sync_rate
        self.eps = eps
        self.action_space=env_fn().action_space

        self.q1 = QCritic(q_args)
        self.q2 = QCritic(q_args)
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
            
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_args['lr'])
        
    def updateQ(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2 = self.pi(o2) # a distribution when discrete
            q_next = torch.min(self.q1(o2), self.q2(o2))
            a = q_next.argmax(dim=1)
            q1_pi_targ = self.target.q1(o2, a)
            q2_pi_targ = self.target.q2(o2, a)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        self.logger.log(q_mean=q_pi_targ.mean(), q_hist=q_pi_targ, q_diff=((q1+q2)/2-backup).mean())

        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q.backward()
        torch.nn.utils.clip_grad_norm_(parameters=self.q_params, max_norm=5, norm_type=2)
        self.q_optimizer.step()

        # Record things
        self.logger.log(q_update=None, loss_q=loss_q/2)
        
        # update the target nets
        with torch.no_grad():
            for current, target in [(self.q1, self.q1_target), (self.q2, self.q2_target)]:
                for p, p_targ in zip(current.parameters(), target.parameters()):
                    p_targ.data.mul_(1 - self.target_sync_rate)
                    p_targ.data.add_(self.target_sync_rate * p.data)
                
        
    def act(self, o, deterministic=False):
        """returns a scalar, not differentiable"""
        with torch.no_grad():
            if len(o.shape) == 1:
                o = o.unsqueeze(0)

            q1 = self.q1(o)
            q2 = self.q2(o)
            q = torch.min(q1, q2)
            a = q.argmax(dim=1)[0]
            if not deterministic and random.random()<self.eps:
                return torch.as_tensor(self.action_space.sample())
            return a
    
class PPO(nn.Module):
    """ Actor Critic (V function) """
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPVFunction(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]
    
class SAC(nn.Module):
    """
        Actor Critic (Q function)
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, logger=None, q_lr=1e-4, pi_lr=1e-4, gamma=0.99, alpha=0.2,
                 polyak=0.995, eps=0, dqn=False):
        super().__init__()
        self.logger = logger
        self.gamma = gamma
        self.alpha = alpha
        self.polyak=polyak
        self.dqn = dqn
        self.eps = eps
        self.action_space=action_space
        
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
            print(f"action dim: {act_dim}")
            for i, _ in enumerate(action_space.high):
                print(f"action space size: {action_space.low[i]}, {action_space.high[i]}")
        elif isinstance(action_space, Discrete):
            act_dim = action_space.n
            print(f"number of actions: {act_dim}")
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation, eps)

        # build policy and value functions
        
        self.q1 = MLPQFunction(obs_dim, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, action_space, hidden_sizes, activation)
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi.parameters(), lr=pi_lr)
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_lr)
        
    def setTarget(self, target):
        self.target = target

    def act(self, o, deterministic=False):
        """not batched"""
        with torch.no_grad():
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
                
            if self.dqn:
                q1 = self.q1(o)
                q2 = self.q2(o)
                q = torch.min(q1, q2)
                a = q.argmax(dim=1)[0]
                if not deterministic and random.random()<self.eps:
                    return torch.as_tensor(self.action_space.sample())
                return a

            if isinstance(self.action_space, Discrete):
                a = self.pi(o)
                if deterministic:
                    a = a.argmax(dim=1)[0]
                else:
                    a = Categorical(a[0]).sample()
            else:
                a = self.pi(o, deterministic)
                if isinstance(a, tuple):
                    a = a[0]
                a = a.squeeze(dim=0)
            return a
        
        
    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2 = self.pi(o2) # a distribution when discrete
            if self.dqn:
                q_next = torch.min(self.q1(o2), self.q2(o2))
                a = q_next.argmax(dim=1)
                q1_pi_targ = self.target.q1(o2, a)
                q2_pi_targ = self.target.q2(o2, a)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + self.gamma * (1 - d) * (q_pi_targ)
                
            elif isinstance(self.action_space, Discrete):
                logp_a2 = torch.log(a2)
                # Target Q-values
                q1_pi_targ = self.target.q1(o2)
                q2_pi_targ = self.target.q2(o2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r.unsqueeze(dim=1) + self.gamma * (1 - d.unsqueeze(dim=1)) * (q_pi_targ - self.alpha * logp_a2)
                backup = (backup* a2).sum(dim=1)
            else:
                a2, logp_a2 = a2
                # Target Q-values
                q1_pi_targ = self.target.q1(o2, a2)
                q2_pi_targ = self.target.q2(o2, a2)
                q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
                backup = r + self.gamma * (1 - d) * (q_pi_targ - self.alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        self.logger.log(q_mean=q_pi_targ.mean(), q_hist=q_pi_targ, q_diff=((q1+q2)/2-backup).mean())
        
        if torch.isnan(loss_q):
            pdb.set_trace()

        return loss_q

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        if isinstance(self.action_space, Discrete):
            pi = self.pi(o)
            logp = torch.log(pi)
            q1 = self.q1(o)
            q2 = self.q2(o)
            q = torch.min(q1, q2)
            q = q - self.alpha * logp
            optimum = q.max(dim=1, keepdim=True)[0].detach()
            regret = optimum - (pi*q).sum(dim=1)
            loss = regret.mean()
            entropy = -(pi*logp).sum(dim=1).mean(dim=0)
            self.logger.log(entropy=entropy, pi_regret=loss)
        else:
            action, logp = self.pi(o)
            q1 = self.q1(o, action)
            q2 = self.q2(o, action)
            q = torch.min(q1, q2)
            q = q - self.alpha * logp
            loss = (-q).mean()
            self.logger.log(logp=logp, pi_reward=q)

        return loss

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q= self.compute_loss_q(data)
        if not torch.isnan(loss_q):
            loss_q.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.q_params, max_norm=5, norm_type=2)
            self.q_optimizer.step()
        else:
            print("q loss is nan")

        # Record things
        self.logger.log(q_update=None, loss_q=loss_q/2)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        if not torch.isnan(loss_pi):
            loss_pi.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.pi.parameters(), max_norm=5, norm_type=2)
            self.pi_optimizer.step()
        else:
            print("pi loss is nan")

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
            
        with torch.no_grad():
            """ need to change this, there is not a pi target"""
            for p, p_targ in zip(self.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
                
        # Record things
        self.logger.log(pi_update=None)
        
class MBPO(nn.Module):
    """ PPO """
    def __init__(self, observation_space, action_space, 
                 hidden_sizes=(64,64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder depends on action space
        if isinstance(action_space, Box):
            self.pi = MLPGaussianActor(obs_dim, action_space.shape[0], hidden_sizes, activation)
        elif isinstance(action_space, Discrete):
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v  = MLPVFunction(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self, obs):
        return self.step(obs)[0]