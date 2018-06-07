import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from helper_functions import SlidingMemory, PERMemory
from torch_networks import DDPG_actor_network, DDPG_critic_network, NAF_network

        

class DDPG():    
    def __init__(self, state_dim, action_dim, mem_size, train_batch_size, gamma, actor_lr, critic_lr, 
                 action_low, action_high, tau, noise, if_PER = True):
        self.mem_size, self.train_batch_size = mem_size, train_batch_size
        self.gamma, self.actor_lr, self.critic_lr = gamma, actor_lr, critic_lr
        self.global_step = 0
        self.tau, self.explore = tau, noise
        self.state_dim, self.action_dim = state_dim, action_dim
        self.action_high, self.action_low = action_high, action_low
        self.replay_mem = PERMemory(mem_size) if if_PER else SlidingMemory(mem_size)
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = 'cpu'
        self.if_PER = if_PER
        self.actor_policy_net = DDPG_actor_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_target_net = DDPG_actor_network(state_dim, action_dim,action_low, action_high).to(self.device)
        #self.critic_policy_net = DDPG_critic_network(state_dim, action_dim).to(self.device)
        #self.critic_target_net = DDPG_critic_network(state_dim, action_dim).to(self.device)
        self.critic_policy_net = NAF_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.critic_target_net = NAF_network(state_dim, action_dim, action_low, action_high).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor_policy_net.parameters(), self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_policy_net.parameters(), self.critic_lr)
        self.hard_update(self.actor_target_net, self.actor_policy_net)
        self.hard_update(self.critic_target_net, self.critic_policy_net)
    
    
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

        self.explore.decaynoise()
        
        # sample $self.train_batch_size$ samples from the replay memory, and use them to train
        if not self.if_PER:
            train_batch = self.replay_mem.sample(self.train_batch_size)
        else:
            train_batch, idx_batch, weight_batch = self.replay_mem.sample(self.train_batch_size)
            weight_batch = torch.tensor(weight_batch, dtype = torch.float).unsqueeze(1)
        
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
            next_action_batch = self.actor_target_net(next_state_batch)
            #print(next_action_batch)
            q_target_ = self.critic_target_net(next_state_batch, next_action_batch)
            q_target = self.gamma * q_target_ * (1 - if_end) + reward_batch

        q_pred = self.critic_policy_net(pre_state_batch, action_batch)
        
        if self.if_PER:
            TD_error_batch = np.abs(q_target.numpy() - q_pred.detach().numpy())
            self.replay_mem.update(idx_batch, TD_error_batch)
        
        self.critic_optimizer.zero_grad()
        closs = (q_pred - q_target) ** 2 
        if self.if_PER:
            closs *= weight_batch
            
        closs = torch.mean(closs)
        closs.backward()
        torch.nn.utils.clip_grad_norm(self.critic_policy_net.parameters(), 1)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        aloss = - self.critic_policy_net(pre_state_batch, self.actor_policy_net(pre_state_batch))
        aloss = aloss.mean()
        aloss.backward()
        torch.nn.utils.clip_grad_norm(self.actor_policy_net.parameters(), 1)
        self.actor_optimizer.step()
    

        # update target network
        self.soft_update(self.actor_target_net, self.actor_policy_net, self.tau)
        self.soft_update(self.critic_target_net, self.critic_policy_net, self.tau)
        self.global_step += 1
        
        
        if self.global_step >0 and self.global_step % 10000 == 0:
            torch.save(self.actor_policy_net.state_dict(), './record/ddpg_actor_param.txt')
            torch.save(self.critic_policy_net.state_dict(), './record/ddpg_critic_param.txt')
    
    # store the (pre_s, action, reward, next_state, if_end) tuples in the replay memory
    def perceive(self, pre_s, action, reward, next_state, if_end):
        self.replay_mem.append([pre_s, action, reward, next_state, if_end])
        if len(self.replay_mem) > self.mem_size:
            self.replay_mem.popleft()
        
    
    # use the policy net to choose the action with the highest Q value
    def action(self, s, add_noise = True):
        s = torch.tensor(s, dtype=torch.float, device = self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.actor_policy_net(s) 
        
        noise = self.explore.noise() if add_noise else 0.0
        # use item() to get the vanilla number instead of a tensor
        #return [np.clip(np.random.normal(action.item(), self.explore_rate), self.action_low, self.action_high)]
        return np.clip(action.numpy()[0] + noise, self.action_low, self.action_high)
    
    
    
    # choose an action according to the epsilon-greedy method
    #def e_action(self, s):
    #    p = random.random()
    #    if p <= 1.0 / np.log(self.global_step + 3):
    #        return random.randint(0, self.action_dim - 1)
    #    else:
    #        return self.action(s)
        