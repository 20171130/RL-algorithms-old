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


def MLP(sizes, activation, output_activation=nn.Identity, **kwargs):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j+1]), act()]
    return nn.Sequential(*layers)

def CNN(sizes, kernels, strides, paddings, activation, output_activation=nn.Identity, **kwargs):
    layers = []
    for j in range(len(sizes)-1):
        act = activation if j < len(sizes)-2 else output_activation
        layers += [nn.Conv2d(sizes[j], sizes[j+1], kernels[j], strides[j], paddings[j]), act()]
    return nn.Sequential(*layers)

class ParameterizedModel(nn.Module):
    """
        assumes parameterized state representation
        we may use a gaussian prediciton,
        but it degenrates without a kl hyperparam
        unlike the critic and the actor class, 
        the sizes argument does not include the dim of the state
    """
    def __init__(self, env_fn, logger, **net_args):
        super().__init__()
        self.logger = logger
        self.action_space=env_fn().action_space
        observation_dim=env_fn().observation_space.shape[0]
        input_dim = net_args['sizes'][0]
        output_dim = net_args['sizes'][-1]
        if isinstance(self.action_space, Discrete):
            self.action_embedding = nn.Embedding(self.action_space.n,input_dim)
        self.net = MLP(**net_args)
        self.state_head = nn.Linear(output_dim, observation_dim)
        self.reward_head = nn.Linear(output_dim, 1)
        self.done_head = nn.Linear(output_dim, 1)
        self.MSE = nn.MSELoss(reduction='none')
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, s, a, r=None, s1=None, d=None):
        if r is None: #inference
            with torch.no_grad():
                embedding = s
                if isinstance(self.action_space, Discrete):
                    embedding = embedding + self.action_embedding(a)
                embedding = self.net(embedding)

                state = self.state_head(embedding)
                reward = self.reward_head(embedding).squeeze(1)
                done = torch.sigmoid(self.done_head(embedding))
                done = torch.cat([1-done, done], dim = 1)
                done = Categorical(done).sample() # [b]
                return  reward, state, done
        else: # training
            embedding = s
            if isinstance(self.action_space, Discrete):
                embedding = embedding + self.action_embedding(a)
            embedding = self.net(embedding)

            state = self.state_head(embedding)
            reward = self.reward_head(embedding).squeeze(1)
            done = self.done_head(embedding).squeeze(1)
            
            state_loss = self.MSE(state, s1).mean(dim = 1)
            reward_loss = self.MSE(reward, r)
            done_loss = self.BCE(done, d)
            done = done > 0

            done_true_positive = (done*d).mean()
            d = d.mean()
            
            self.logger.log(state_loss=state_loss, reward_loss=reward_loss, done_loss=done_loss)
            self.logger.log(done_true_positive=done_true_positive, done=d, rolling=100)
            return state_loss+reward_loss+10*done_loss
        
    
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
    def __init__(self, env_fn, **q_args):
        super().__init__()
        q_net = q_args['network']
        self.action_space=env_fn().action_space
        self.q = q_net(**q_args)
       
    def forward(self, obs, action=None):
        if isinstance(self.action_space, Box):
            q = self.q(torch.cat([obs, action], dim=-1))
        else:
            q = self.q(obs)
            while len(q.shape) > 2:
                q = q.squeeze(-1) # HW of size 1 if CNN
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

class CategoricalActor(nn.Module):
    """ 
    always returns a distribution
    """
    def __init__(self, **net_args):
        super().__init__()
        self.softmax = nn.Softmax(dim=1)
        net_fn = net_args['network']
        self.network = net_fn(**net_args)
    
    def forward(self, obs):
        logit = self.network(obs)
        while len(logit.shape) > 2:
            logit = logit.squeeze(-1) # HW of size 1 if CNN
        return self.softmax(logit)
    
class RegressionActor(nn.Module):
    """
    determinsitc actor, used in DDPG and TD3
    """
    def __init__(self, **net_args):
        super().__init__()
        net_fn = net_args['network']
        self.network = net_fn(**net_args)
    
    def forward(self, obs):
        out = self.network(obs)
        while len(out.shape) > 2:
            out = out.squeeze(-1) # HW of size 1 if CNN
        return out
    
class GaussianActor(nn.Module):

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
    """
    stochastic actor for continous action sapce, used in SAC 
    """
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

