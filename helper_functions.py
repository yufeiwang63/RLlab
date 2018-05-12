# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 16:43:56 2018

@author: Wangyf
"""

import sklearn.pipeline
import sklearn.preprocessing
import numpy as np
from sklearn.kernel_approximation import RBFSampler


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


