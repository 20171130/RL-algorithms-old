import gym
import numpy as np
import random
import torch

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class Config(object):
    def __init__(self):
        return None
    def _toDict(self, recursive=False):
        """
            converts to dict for **kwargs
            recursive for logging
        """
        pr = {}
        for name in dir(self):
            value = getattr(self, name)
            if not name.startswith('_') and not name.endswith('_'):
                if isinstance(value, Config) and recursive:
                    value = value._toDict(recursive)
                pr[name] = value
        return pr
    
class TabularLogger(object):
    """
    A text interface logger, outputs mean and std several times per epoch
    """
    def __init__(self):
        self.buffer = {}
        
    def log(dic, commit=False):
        if commit:
            print
        
class Logger(object):
    """
    A logger wrapper for visualized loggers, such as tb or wandb
    Automatically counts steps, epoch, etc. and sets logging interval
    to prevent the log becoming too big
    uses kwargs instead of dict for convenience
    all None valued keys are counters
    """
    def __init__(self, logger):
        self.logger = logger
        self.counters = {'epoch':0}
        self.frequency = 10 # logs per epoch
        
    def log(self, data=None, **kwargs):
        if data is None:
            data = {}
        data.update(kwargs)
        # counting
        for key in data:
            if not key in self.counters:
                self.counters[key] = 0
            self.counters[key] += 1
            
        to_store = {}
        epoch = self.counters['epoch']
        for key in data:
            count = self.counters[key]
            period = count//(epoch+1) + 1
            flag = random.random()< self.frequency/period
            if flag:
                if data[key] is None:
                    to_store[key] = self.counters[key]
                else:
                    valid = True
                    if isinstance(data[key], torch.Tensor):
                        data[key] = data[key].detach().cpu()
                        if  torch.isnan(data[key]).any():
                            valid = False
                    elif np.isnan(data[key]).any():
                        valid = False
                    if not valid:
                        print(f'{key} is nan!')
                        continue
                    to_store[key] = data[key]
                
        if len(to_store) > 0:
            self.logger.log(to_store, commit=True)
        
    def flush(self):
        self.logger.log(data={'epoch':self.counters['epoch']}, commit=True)