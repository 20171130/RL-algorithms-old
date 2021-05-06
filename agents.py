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

        self.q1 = QCritic(**q_args._toDict())
        self.q2 = QCritic(**q_args._toDict())
        self.q1_target = deepcopy(self.q1)
        self.q2_target = deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False
            
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=q_args.lr)
        
    def updateQ(self, data):
        o, a, r, o2, d = data['s'], data['a'], data['r'], data['s1'], data['d']

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            q_next = torch.min(self.q1(o2), self.q2(o2))
            a = q_next.argmax(dim=1)
            q1_pi_targ = self.q1_target(o2, a)
            q2_pi_targ = self.q2_target(o2, a)
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
    
class SAC(QLearning):
    """ Actor Critic (Q function) """
    def __init__(self, logger, env_fn, q_args, pi_args, alpha, gamma, target_sync_rate, **kwargs):
        """
            q_net is the network class
        """
        super().__init__(logger, env_fn, q_args, gamma, 0, target_sync_rate, **kwargs)
        # eps = 0
        self.logger = logger
        self.gamma = gamma
        self.alpha = alpha
        self.action_space=env_fn().action_space

    def act(self, obs):
        return self.step(obs)[0]
    
    def updatepi(self, data):
        return None
        
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