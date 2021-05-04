import numpy as np
import pdb
import itertools
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
from torch.optim import Adam


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input: 
        vector x, 
        [x0, 
         x1, 
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,  
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Actor(nn.Module):

    def _distribution(self, obs):
        raise NotImplementedError

    def _log_prob_from_distribution(self, pi, act):
        raise NotImplementedError

    def forward(self, obs):
        # Produce action distributions for given observations, and 
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        return pi

class MLPCategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, eps=0):
        super().__init__()
        self.eps = eps
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        """ Distribution of action"""
        logits = self.logits_net(obs)
        prob = Categorical(logits=logits).probs
        prob = prob + self.eps # eps for numerical stability
        prob = prob/(prob.sum(dim=1, keepdim=True))
        return prob


class MLPGaussianActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        log_std = -0.5 * np.ones(act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std))
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        mu = self.mu_net(obs)
        std = torch.exp(self.log_std)
        return Normal(mu, std)

    def _log_prob_from_distribution(self, pi, act):
        return pi.log_prob(act).sum(axis=-1)    # Last axis sum needed for Torch Normal distribution

LOG_STD_MAX = 2
LOG_STD_MIN = -20
class SquashedGaussianMLPActor(nn.Module):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, act_limit):
        super().__init__()
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.log_std_layer = nn.Linear(hidden_sizes[-1], act_dim)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False, with_logprob=True):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        # Pre-squash distribution and sample
        pi_distribution = Normal(mu, std)
        if deterministic:
            # Only used for evaluating policy at test time.
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        if with_logprob:
            # Compute logprob from Gaussian, and then apply correction for Tanh squashing.
            # NOTE: The correction formula is a little bit magic. To get an understanding 
            # of where it comes from, check out the original SAC paper (arXiv 1801.01290) 
            # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
            # Try deriving it yourself as a (very difficult) exercise. :)
            logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
            logp_pi -= (2*(np.log(2) - pi_action - F.softplus(-2*pi_action))).sum(axis=1)
        else:
            logp_pi = None

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi

class MLPVFunction(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class MLPQFunction(nn.Module):

    def __init__(self, obs_dim, action_space, hidden_sizes, activation):
        super().__init__()
        if isinstance(action_space, Box):
            self.action_space = 'continous'
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
            self.q = mlp([obs_dim + act_dim] + list(hidden_sizes) + [1], activation)
                
        elif isinstance(action_space, Discrete):
            self.action_space = 'discrete'
            self.q = mlp([obs_dim] + list(hidden_sizes) + [action_space.n], activation)
       
    def forward(self, obs, action=None):
        if self.action_space == 'continous':
            q = self.q(torch.cat([obs, act], dim=-1))
        else:
            q = self.q(obs) 
            if action is None:
                return q.squeeze(1)
            if action.dtype == torch.long or action.dtype == torch.int64 or action.dtype == torch.int32:
                q = torch.gather(input=q,dim=1,index=action.unsqueeze(-1))
                return q.squeeze(1)
            elif not action is None: # expectation
                q = (q*action).sum(dim=-1)
                return q.squeeze(1)


class MLPVActorCritic(nn.Module):
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
    
class MLPDQActorCritic(nn.Module):
    """
        pi returns a distribution
        .act() does the sampling
        q = Q(s, a) when continous
        when discrete, Q returns Q for all/average/an action,
        depending on the input action
    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, logger=None, lr=1e-4, gamma=0.99, alpha=0.2, polyak=0.995, eps=0):
        super().__init__()
        self.logger = logger
        self.gamma = gamma
        self.alpha = alpha
        self.polyak=polyak
        
        obs_dim = observation_space.shape[0]
        if isinstance(action_space, Box):
            self.action_space='continous'
            act_dim = action_space.shape[0]
            act_limit = action_space.high[0]
            self.pi = SquashedGaussianMLPActor(obs_dim, act_dim, hidden_sizes, activation, act_limit)
        elif isinstance(action_space, Discrete):
            self.action_space='discrete'
            act_dim = action_space.n
            self.pi = MLPCategoricalActor(obs_dim, action_space.n, hidden_sizes, activation, eps)

        # build policy and value functions
        
        self.q1 = MLPQFunction(obs_dim, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, action_space, hidden_sizes, activation)
        
        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.pi.parameters(), lr=lr)
        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.q1.parameters(), self.q2.parameters())
        self.q_optimizer = Adam(self.q_params, lr=lr)
        
    def setTarget(self, target):
        self.target = target

    def act(self, o, deterministic=False):
        """not batched"""
        with torch.no_grad():
            o = torch.as_tensor(o, dtype=torch.float32)
            if len(o.shape) == 1:
                o = o.unsqueeze(0)
            a = self.pi(o)
            if deterministic:
                a = a.argmax(dim=1)[0]
            else:
                a = Categorical(a[0]).sample()
            return a.numpy()
        
    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = self.q1(o, a)
        q2 = self.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2 = self.pi(o2).detach() # a distribution when discrete
            logp_a2 = torch.log(a2)

            # Target Q-values
            q1_pi_targ = self.target.q1(o2)
            q2_pi_targ = self.target.q2(o2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            q_pi_targ = q_pi_targ 
            backup = r.unsqueeze(dim=1) + self.gamma * (1 - d.unsqueeze(dim=1)) * (q_pi_targ - self.alpha * logp_a2)
            backup = (backup* a2).sum(dim=1)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data['obs']
        if self.action_space =='discrete':
            pi = self.pi(o)
            logp = torch.log(pi)
            q1 = self.q1(o)
            q2 = self.q2(o)
            q = torch.min(q1, q2)
            loss = (self.alpha * logp - q)*pi
            loss = loss.sum(dim=1).mean(dim=0)
            
            entropy = -(pi*logp).sum(dim=1).mean(dim=0)
            self.logger.log(entropy=entropy, pi_loss=loss)

        return loss

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.log(q_update=None, **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True
            
        with torch.no_grad():
            for p, p_targ in zip(self.parameters(), self.target.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)
                
        # Record things
        self.logger.log(pi_update=None)