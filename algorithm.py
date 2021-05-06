import pdb
import numpy as np
import torch
import gym
import time
import random
from tqdm import tqdm
from utils import combined_shape

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    Utilizes lazy frames of FrameStack to save memory.
    """

    def __init__(self, max_size, device):
        self.max_size = max_size
        self.data = []
        self.ptr = 0
        self.device = device

    def store(self, obs, act, rew, next_obs, done):
        if len(self.data) == self.ptr:
            self.data.append({})
        self.data[self.ptr] = {'s':obs, 'a':act, 'r':rew, 's1':next_obs, 'd':float(done)}
        # lazy frames here
        # cuts Q bootstrap if done (next_obs is arbitrary)
        self.ptr = (self.ptr+1) % self.max_size
        
    def sample_batch(self, batch_size):
        idxs = np.random.randint(0, len(self.data), size=batch_size)
        raw_batch = [self.data[i] for i in idxs]
        batch = {}
        for key in raw_batch[0]:
            try:
                lst = [torch.as_tensor(dic[key]) for dic in raw_batch]
                batch[key] = torch.stack(lst).to(self.device)
            except:
                pdb.set_trace()
        return batch



def RL(logger, device,
       env_fn, agent, 
      replay_size, max_ep_len, n_warmup, 
      batch_size, n_step, n_update, n_test, 
       seed, n_epoch, **kwargs):
    """ 
    a generic algorithm for model-free reinforcement learning
    plugin state preprocessing if necessary, by wrapping the env
    warmup:
        model, q, and policy each warmup for n_warmup steps before used
    """

    env, test_env = env_fn(), env_fn()
    observation_space = env.observation_space
    action_space = env.action_space
    # Experience buffer
    replay_buffer = ReplayBuffer(max_size=replay_size, device=device)
    
    pbar = iter(tqdm(range(n_epoch)))

    def test_agent():
        for j in range(n_test):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                action = agent.act(torch.as_tensor(o,  dtype=torch.float).to(device), True)
                o, r, d, _ = test_env.step(action.cpu().numpy())
                ep_ret += r
                ep_len += 1
            logger.log(TestEpRet=ep_ret, TestEpLen=ep_len, testEpisode=None)

    # Prepare for interaction with environment
    total_steps = n_step * n_epoch
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        logger.log(interaction=None)
        if t >= n_warmup:
            a = agent.act(torch.as_tensor(o,  dtype=torch.float).to(device))
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
        if t % (n_step//n_update) == 0:
            batch = replay_buffer.sample_batch(batch_size)
            agent.updateQ(data=batch)
                
        # End of epoch handling
        if (t+1) % n_step == 0:
            next(pbar)

            # Test the performance of the deterministic version of the agent.
            test_agent()
            # Log info about epoch
            logger.log(epoch=None)
            logger.flush()
    
    return ac