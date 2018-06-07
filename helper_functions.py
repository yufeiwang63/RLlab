# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:43:56 2018

@author: Wangyf
"""

import sklearn.pipeline
import sklearn.preprocessing
import numpy as np
from sklearn.kernel_approximation import RBFSampler
import random
from collections import deque


class Featurize_state():
    def __init__(self, env, no_change = False):
        self.no_change = no_change
        if no_change == True:
            self.After_featurize_state_dim = env.observation_space.shape[0]
            return 
        
        observation_examples = np.array([env.observation_space.sample() for x in range(10000)])
        
        self.scaler = sklearn.preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to converte a state to a featurizes represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = sklearn.pipeline.FeatureUnion([
        ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
        ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
        ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
        ("rbf4", RBFSampler(gamma=0.5, n_components=100))
        ])
        self.featurizer.fit(observation_examples)
        self.After_featurize_state_dim = 400
        
    def get_featurized_state_dim(self):
        return self.After_featurize_state_dim
        
    def transfer(self, state):
        if self.no_change:
            return state
        
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]
        #return state
        


# from https://github.com/songrotek/DDPG/blob/master/ou_noise.py
class OUNoise:
    def __init__(self, action_dimension, initial_scale = 1, final_scale = 0.2, decay = 0.9995, mu=0, theta=0.15, sigma=0.2):
        self.action_dimension = action_dimension
        self.scale = initial_scale
        self.final_scale = final_scale
        self.decay = decay
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu

    def decaynoise(self):
        self.scale *= self.decay
        self.scale = max(self.scale, self.final_scale)

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        res = self.state * self.scale
        return res[0]
    
    def noisescale(self):
        return self.scale
    
class GaussNoise():
    def __init__(self, initial_var = 10, final_var = 0, decay = 0.995):
        self.var = initial_var
        self.final_var = final_var
        self.decay = decay
        
    def decaynoise(self):
        self.var *= self.decay
        self.var = max(self.final_var, self.var)
        
    def noise(self):
        return np.random.normal(0, self.var)
    
    def noisescale(self):
        return self.var


class SlidingMemory():
    
    def __init__(self, mem_size):
        self.mem = deque()
        self.mem_size = mem_size
        
    def add(self, state, action, reward, next_state, if_end):
        self.mem.append([state, action, reward, next_state, if_end])
        if len(self.mem) > self.mem_size:
            self.mem.popleft()
            
    def num(self):
        return len(self.mem)
    
    def sample(self, batch_size):
        return random.sample(self.mem, batch_size)
    
    def clear(self):
        self.mem.clear()




class SumTree:
    write = 0

    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros( 2*capacity - 1 )
        self.data = np.zeros( capacity, dtype=object )
        self.number = 0

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        
        if idx >= self.capacity - 1:
            return idx
        
        left = 2 * idx + 1
        right = left + 1

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, data, p):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
            
        self.number = min(self.number + 1, self.capacity)

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        
        
        return (idx, self.tree[idx], self.data[dataIdx])
    
    def num(self):
        return self.number

    

class PERMemory():
    
    def __init__(self, mem_size, alpha = 0.8, beta = 0.8, eps = 1e-2):
        self.alpha, self.beta, self.eps = alpha, beta, eps
        self.mem_size = mem_size
        self.mem = SumTree(mem_size)
        
    def add(self, state, action, reward, next_state, if_end):
        # here use reward for initial p, instead of maximum for initial p
        p = 1000
        self.mem.add([state, action, reward, next_state, if_end], p)
        
    def update(self, batch_idx, batch_td_error):
        for idx, error in zip(batch_idx, batch_td_error):
            p = (error + self.eps)  ** self.alpha 
            self.mem.update(idx, p)
        
    def num(self):
        return self.mem.num()
    
    def sample(self, batch_size):
        
        data_batch = []
        idx_batch = []
        p_batch = []
        
        segment = self.mem.total() / batch_size
        #print(self.mem.total())
        #print(segment * batch_size)

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            #print(s < self.mem.total())
            idx, p, data = self.mem.get(s)
            data_batch.append(data)
            idx_batch.append(idx)
            p_batch.append(p)
        
        p_batch = (1.0/ np.array(p_batch) /self.mem_size) ** self.beta
        p_batch /= max(p_batch)
        
        self.beta = min(self.beta * 1.00005, 1)
    
        return (data_batch, idx_batch, p_batch)
        
if __name__ == 'main':        
    mymem = PERMemory(4)
    
    mymem.mem.add('a',1.1)
    mymem.mem.add('b',2.2)
    mymem.mem.add('c',3.34352245)
    
    print(mymem.sample(2))
