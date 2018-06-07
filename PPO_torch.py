#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 23 09:58:01 2018

@author: yufei
"""

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import random
from collections import deque
from torch_networks import AC_v_fc_network, CAC_a_fc_network
from helper_functions import SlidingMemory, PERMemory
import warnings

warnings.simplefilter("error", RuntimeWarning)

        

class PPO():    
    def __init__(self, state_dim, action_dim, action_low, action_high, mem_size, train_batch_size, gamma, actor_lr, critic_lr, 
                 tau, eps, update_epoach):
        self.mem_size, self.train_batch_size = mem_size, train_batch_size
        self.gamma, self.actor_lr, self.critic_lr = gamma, actor_lr, critic_lr
        self.global_step = 0
        self.tau, self.eps = tau, eps
        self.state_dim, self.action_dim = state_dim, action_dim
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.replay_mem = SlidingMemory(mem_size)
        self.device = 'cpu'
        self.action_low, self.action_high = action_low, action_high
        self.actor_policy_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target_net = CAC_a_fc_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.critic_policy_net = AC_v_fc_network(state_dim).to(self.device)
        self.critic_target_net = AC_v_fc_network(state_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        self.hard_update(self.critic_target_net, self.critic_policy_net)
        self.update_epoach = update_epoach 
    
    
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    #  training process                          
    def train(self, pre_state, action, reward, next_state, if_end):
        
        self.replay_mem.add(pre_state, action, reward, next_state, if_end)
        
        if self.replay_mem.num() < self.mem_size:
            return
         
        print("train epoach!")
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        
        for i in range(self.update_epoach):

            train_batch = self.replay_mem.sample(self.train_batch_size)
    
            # adjust dtype to suit the gym default dtype
            pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float, device = self.device) 
            action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.float, device = self.device) 
            # view to make later computation happy
            reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float, device = self.device).view(self.train_batch_size,1)
            next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float, device = self.device)
            if_end = [x[4] for x in train_batch]
            if_end = torch.tensor(np.array(if_end).astype(float),device = self.device, dtype=torch.float).view(self.train_batch_size,1)
            
            
            # use the target_Q_network to get the target_Q_value
            with torch.no_grad():
                v_next_state = self.critic_target_net(next_state_batch).detach()
                v_target = self.gamma * v_next_state * (1 - if_end) + reward_batch
    
            v_pred = self.critic_policy_net(pre_state_batch)
            
            advantage = v_pred.detach() - v_target
            
            
            old_action_prob = self.actor_target_net(pre_state_batch).log_prob(action_batch)
        
            self.actor_optimizer.zero_grad()
            log_action_prob = self.actor_policy_net(pre_state_batch).log_prob(action_batch)
                
            aloss1 = log_action_prob / old_action_prob * advantage
            aloss2 = torch.clamp(log_action_prob / old_action_prob, 1 - self.eps, 1 + self.eps) * advantage
            aloss = - torch.min(aloss1, aloss2)
            aloss = aloss.mean()
            aloss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_policy_net.parameters(),1)
            self.actor_optimizer.step()
            
            
        self.critic_optimizer.zero_grad()
        closs = (v_pred - v_target) ** 2 
        closs = closs.mean()
        closs.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_policy_net.parameters(),1)
        self.critic_optimizer.step()
        
        
        # update target network
        self.replay_mem.clear()
        self.soft_update(self.critic_target_net, self.critic_policy_net, self.tau)
        self.global_step += 1
    
    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
        
    
    # use the policy net to choose the action with the highest Q value
    def action(self, s, sample = True): # use flag to suit other models' action interface
        s = torch.tensor(s, dtype=torch.float, device = self.device).unsqueeze(0)
        with torch.no_grad():
            m = self.actor_policy_net(s)
            a = np.clip(m.sample(), self.action_low, self.action_high) if sample else m.mean
            return a.numpy()[0]
    
        
    
        