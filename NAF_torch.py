import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch_networks import NAF_network



class NAF():    
    def __init__(self, state_dim, action_dim, mem_size, train_batch_size, gamma, lr,
                 action_high, action_low, tau, noise, flag):
        self.mem_size, self.train_batch_size = mem_size, train_batch_size
        self.gamma, self.lr = gamma, lr
        self.global_step = 0
        self.tau, self.explore = tau, noise
        self.state_dim, self.action_dim = state_dim, action_dim
        self.action_high, self.action_low = action_high, action_low
        self.replay_mem = deque()
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.cret = nn.MSELoss()
        self.policy_net = NAF_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.target_net = NAF_network(state_dim, action_dim,action_low, action_high).to(self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), self.lr)
        self.hard_update(self.target_net, self.policy_net)
        
        self.flag = flag
    
    
    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)
    
    #  training process                          
    def train(self, pre_state, action, reward, next_state, if_end):
        
        self.perceive(pre_state, action, reward, next_state, if_end)
        
        if len(self.replay_mem) < self.mem_size:
            return
        
        self.explore.decaynoise()
        
        
        # sample $self.train_batch_size$ samples from the replay memory, and use them to train
        train_batch = random.sample(self.replay_mem, self.train_batch_size)
        
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
            q_target_, _ = self.target_net(next_state_batch)
            q_target = self.gamma * q_target_ * (1 - if_end) + reward_batch


        q_pred = self.policy_net(pre_state_batch, action_batch)
        
        self.optimizer.zero_grad()
        loss = self.cret(q_pred, q_target)
        if self.flag:
            loss -= q_pred.mean() # to test one of my ideas
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
        # update target network
        self.soft_update(self.target_net, self.policy_net, self.tau)
        self.global_step += 1
    
    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
            
            
    # give a state and action, return the action value
    def get_value(self, s, a):
        s = torch.tensor(s,dtype=torch.float, device = self.device)
        with torch.no_grad():
            val = self.policy_net(s.unsqueeze(0)).gather(1, torch.tensor(a,dtype = torch.long).unsqueeze(1)).view(1,1)
            
        return np.clip(val.item() + np.random.rand(1, self.explore_rate), self.action_low, self.action_high)
        
    
    # use the policy net to choose the action with the highest Q value
    def action(self, s, noise = 0.0):
        s = torch.tensor(s, dtype=torch.float, device = self.device).unsqueeze(0)
        with torch.no_grad():
            _, action = self.policy_net(s) 
        
        noise = self.explore.noise()
        # use item() to get the vanilla number instead of a tensor
        #return [np.clip(np.random.normal(action.item(), self.explore_rate), self.action_low, self.action_high)]
        return [np.clip(action.item() + noise, self.action_low, self.action_high)]
    
    