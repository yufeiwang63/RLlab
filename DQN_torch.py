import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch_networks import DQN_fc_network, DQN_dueling_network
from helper_functions import SlidingMemory, PERMemory

        

class DQN():    
    def __init__(self, state_dim, action_dim, mem_size = 10000, train_batch_size = 32, 
                 gamma = 0.99, lr = 1e-3, tau = 0.1,
                 if_dueling = False, if_PER = False, load_path = None ):
        self.mem_size, self.train_batch_size = mem_size, train_batch_size
        self.gamma, self.lr = gamma, lr
        self.global_step = 0
        self.tau = tau
        self.state_dim, self.action_dim = state_dim, action_dim
        self.if_PER = if_PER
        self.replay_mem = PERMemory(mem_size) if if_PER else SlidingMemory(mem_size)
        self.policy_net = DQN_fc_network(state_dim, action_dim,1) 
        self.target_net = DQN_fc_network(state_dim, action_dim,1)
        self.epsilon, self.min_eps = 0.9, 0.4
        
        if load_path is not None:
            self.policy_net.load_state_dict(torch.load(load_path))
        
        if if_dueling:
            self.policy_net = DQN_dueling_network(state_dim, action_dim,1)
            self.target_net = DQN_dueling_network(state_dim, action_dim,1)
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), self.lr)
        self.hard_update(self.target_net, self.policy_net)
           
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
        
        # sample $self.train_batch_size$ samples from the replay memory, and use them to train
        if not self.if_PER:
            train_batch = self.replay_mem.sample(self.train_batch_size)
        else:
            train_batch, idx_batch, weight_batch = self.replay_mem.sample(self.train_batch_size)
            weight_batch = torch.tensor(weight_batch, dtype = torch.float).unsqueeze(1)
            
        # adjust dtype to suit the gym default dtype
        pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float) 
        action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.long) # dtype = long for gater
        reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float).view(self.train_batch_size,1)
        next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float)
        if_end = [x[4] for x in train_batch]
        if_end = torch.tensor(np.array(if_end).astype(float), dtype=torch.float).view(self.train_batch_size,1)
        
        # use the target_Q_network to get the target_Q_value
        # torch.max[0] gives the max value, torch.max[1] gives the argmax index
        
        # vanilla dqn
        #q_target_ = self.target_net(next_state_batch).max(1)[0].detach() # detach to not bother the gradient
        #q_target_ = q_target_.view(self.train_batch_size,1)
        
        # double dqn
        
        with torch.no_grad():
            next_best_action = self.policy_net(next_state_batch).max(1)[1].detach()
            q_target_ = self.target_net(next_state_batch).gather(1, next_best_action.unsqueeze(1))
            q_target_ = q_target_.view(self.train_batch_size,1).detach()
            
        q_target = self.gamma * q_target_ * ( 1 - if_end) + reward_batch
        
        # unsqueeze to make gather happy
        q_pred = self.policy_net(pre_state_batch).gather(1, action_batch.unsqueeze(1)) 
        
        
        if self.if_PER:
            TD_error_batch = np.abs(q_target.numpy() - q_pred.detach().numpy())
            self.replay_mem.update(idx_batch, TD_error_batch)
        
    
        self.optimizer.zero_grad()
        
        loss = (q_pred - q_target) ** 2 
        if self.if_PER:
            loss *= weight_batch
            
        loss = torch.mean(loss)    
        loss.backward()
        torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
        # update target network
        self.soft_update(self.target_net, self.policy_net, self.tau)
        
        self.epsilon = max(self.epsilon * 0.99995, 0.22)
    
    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
            
    # give a state and action, return the action value
    def get_value(self, s, a):
        s = torch.tensor(s,dtype=torch.float)
        with torch.no_grad():
            val = self.policy_net(s.unsqueeze(0)).gather(1, torch.tensor(a,dtype = torch.long).unsqueeze(1)).view(1,1)
            
        return val.item()
    
    def save_model(self, save_path = './model/dqn_params'):
        torch.save(self.policy_net.state_dict(), save_path)
        
        
    # use the policy net to choose the action with the highest Q value
    def action(self, s, epsilon_greedy = True):
        p = random.random() 
        if epsilon_greedy and p <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            s = torch.tensor(s, dtype=torch.float).unsqueeze(0)     
        
            with torch.no_grad():
            # torch.max gives max value, torch.max[1] gives argmax index
                action = self.policy_net(s).max(dim=1)[1].view(1,1) # use view for later item 
            return action.item() # use item() to get the vanilla number instead of a tensor
    
    # choose an action according to the epsilon-greedy method
    def e_action(self, s):
        p = random.random()
        if p <= self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return self.action(s)
        