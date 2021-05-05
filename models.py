import numpy as np
import pdb
import itertools
import scipy.signal
from gym.spaces import Box, Discrete
import random

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


def MLP(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def CNN(sizes, kernels, strides, paddings, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Convolution(sizes[j], sizes[j+1], kernels[j], strides[j], paddings[j]), act()]
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

class WorldModel(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.
    
class VCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

class QCritic(nn.Module):
    """
    Dueling Q, currently only implemented for discrete action space
    if n_embedding > 0, assumes the action space needs embedding
    Notice that the output shape should be 1+action_space.n for discrete dueling Q
    """
    def __init__(self, q_net, q_args):
        super().__init__()
        self.q = q_net(**q_args)
       
    def forward(self, obs, action=None):
        if self.action_space == 'continous':
            q = self.q(torch.cat([obs, action], dim=-1))
        else:
            q = self.q(obs) 
            # [b, a+1]
            v = q[:, -1:]
            q = q[:, :-1]
            q = q - q.mean(dim=1, keepdim=True) + v
            if action is None: 
                # q for all actions
                return q
            elif action.dtype == torch.long or action.dtype == torch.int64 or action.dtype == torch.int32:
                # q for a particular action
                q = torch.gather(input=q,dim=1,index=action.unsqueeze(-1))
                return q.squeeze(dim=1)
            else: 
                # average q for a distribution of actions
                q = (q*action).sum(dim=-1)
                return q.squeeze(dim=1)

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
        if (torch.isnan(pi).any()):
            print('action is nan!')
            pdb.set_trace()
        return pi

class CategoricalActor(Actor):
    
    def __init__(self, obs_dim, act_dim, hidden_sizes, activation, eps=0):
        super().__init__()
        self.eps = eps
        self.softmax = nn.Softmax(dim=1)
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs, eps=None):
        """ Distribution of action"""
        logits = self.logits_net(obs)
        prob = self.softmax(logits)
        if eps is None:
            eps = self.eps
        if eps > 0:
            prob = prob + self.eps # eps for numerical stability
            prob = prob/(prob.sum(dim=1, keepdim=True))
        return prob


class GaussianActor(Actor):

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
class SquashedGaussianActor(nn.Module):

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
        if deterministic or with_logprob == False:
            return pi_action
        return pi_action, logp_pi

