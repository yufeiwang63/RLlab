import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from torch_networks import DQN_fc_network


        

class DQN():    
    def __init__(self, state_dim, action_dim, mem_size, train_batch_size, gamma, lr, tau):
        self.mem_size, self.train_batch_size = mem_size, train_batch_size
        self.gamma, self.lr = gamma, lr
        self.global_step = 0
        self.tau = tau
        self.state_dim, self.action_dim = state_dim, action_dim
        self.replay_mem = deque()
        self.policy_net = DQN_fc_network(state_dim, action_dim)
        self.target_net = DQN_fc_network(state_dim, action_dim)
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
        
        self.perceive(pre_state, action, reward, next_state, if_end)
        
        if len(self.replay_mem) < self.train_batch_size:
            return
        
        # sample $self.train_batch_size$ samples from the replay memory, and use them to train
        train_batch = random.sample(self.replay_mem, self.train_batch_size)
        
        # adjust dtype to suit the gym default dtype
        pre_state_batch = torch.tensor([x[0] for x in train_batch], dtype=torch.float) 
        action_batch = torch.tensor([x[1] for x in train_batch], dtype = torch.long) # dtype = long for gater
        reward_batch = torch.tensor([x[2] for x in train_batch], dtype=torch.float)
        next_state_batch = torch.tensor([x[3] for x in train_batch], dtype=torch.float)
        if_end = [x[4] for x in train_batch]
        if_end = torch.tensor(np.array(if_end).astype(float), dtype=torch.float)
        
        # use the target_Q_network to get the target_Q_value
        # torch.max[0] gives the max value, torch.max[1] gives the argmax index
        q_target_ = self.target_net(next_state_batch).max(1)[0].detach() # detach to not bother the gradient
        q_target = self.gamma * q_target_ * ( 1 - if_end) + reward_batch
        
        
        # unsqueeze to make gather happy
        q_pred = self.policy_net(pre_state_batch).gather(1, action_batch.unsqueeze(1)) 
        
        self.optimizer.zero_grad()
        cret = nn.MSELoss()
        loss = cret(q_pred, q_target.unsqueeze(1))
        loss.backward()
        #torch.nn.utils.clip_grad_norm(self.policy_net.parameters(), 1)
        self.optimizer.step()
    
        # update target network
        self.soft_update(self.target_net, self.policy_net, self.tau)
        
    
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
        
    # use the policy net to choose the action with the highest Q value
    def action(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            # torch.max gives max value, torch.max[1] gives argmax index
            action = self.policy_net(s).max(dim=1)[1].view(1,1) # use view for later item 
        return action.item() # use item() to get the vanilla number instead of a tensor
    
    # choose an action according to the epsilon-greedy method
    def e_action(self, s):
        p = random.random()
        if p <= 1.0 / np.log(self.global_step + 3):
            return random.randint(0, self.action_dim - 1)
        else:
            return self.action(s)
        