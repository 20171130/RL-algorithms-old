from copy import deepcopy
import pdb
import numpy as np
import torch
import gym
import time
import spinup.models as core
from tqdm import tqdm

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        if len(act_dim) > 0:
            self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        else:
            self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.long)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v).to(self.device) for k,v in batch.items()}



def RL(env_fn, agent, 
      replay_size=int(1e6), max_ep_len=2000, start_steps=20000, 
      batch_size=128, n_step=4000, n_update=50, n_test=10, 
      logger=None, device='cpu', seed=0, n_epoch=100):
    """ 
    a generic algorithm for model-free reinforcement learning
    plugin state preprocessing if necessary, by wrapping the env
    """

    env, test_env = env_fn(), env_fn()
    observation_space = env.observation_space
    action_space = env.action_space
    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=observation_space.shape,
                                 act_dim=action_space.shape,
                                 size=replay_size, device=device)
    
    pbar = iter(tqdm(range(epochs)))


    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    print('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    def test_agent():
        for j in range(n_test):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                action = ac.act(torch.as_tensor(o,  dtype=torch.float).to(device), True)
                o, r, d, _ = test_env.step(action.cpu().numpy())
                ep_ret += r
                ep_len += 1
            logger.log(TestEpRet=ep_ret, TestEpLen=ep_len, testEpisode=None)

    # Prepare for interaction with environment
    total_steps = n_step * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t >= start_steps:
            a = ac.act(torch.as_tensor(o,  dtype=torch.float).to(device))
            a = a.detach().cpu().numpy()
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        d = False if ep_len==max_ep_len else d
        replay_buffer.store(o, a, r, o2, d)
        o = o2
        if d or (ep_len == max_ep_len):
            logger.log(EpRet=ep_ret, EpLen=ep_len, episode=None)
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling, can be extended
        if t >= start_steps:
            if t % (n_step//n_update) == 0:
                batch = replay_buffer.sample_batch(batch_size)
                ac.update(data=batch)
                
        # End of epoch handling
        if (t+1) % n_step == 0:
            epoch = (t+1) // n_step
            next(pbar)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            # Log info about epoch
            logger.log(epoch=None)
            logger.log(TotalEnvInteracts=t)
            logger.flush()
    
    return ac