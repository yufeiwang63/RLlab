import torch
import torch.nn as nn
import torch.nn.functional as F


class NAF_network(nn.Module):
        def __init__(self, state_dim, action_dim, action_low, action_high):
            super(NAF_network, self).__init__()
            
            self.sharefc1 = nn.Linear(state_dim, 30)
            self.sharefc2 = nn.Linear(30, 30)
            
            self.v_fc1 = nn.Linear(30, 1)
            
            self.miu_fc1 = nn.Linear(30, action_dim)
            
            self.L_fc1 = nn.Linear(30, action_dim ** 2)
            
            self.action_dim = action_dim
            self.action_low, self.action_high = action_low, action_high

            
        def forward(self, s, a = None):
            
            s = F.relu(self.sharefc1(s))
            s = F.relu(self.sharefc2(s))
            
            v = self.v_fc1(s)
            
            miu = self.miu_fc1(s)
            
            # currently could only clip according to the same one single value.
            # but different dimensions may mave different high and low bounds
            # modify to clip along different action dimension
            miu = torch.clamp(miu, self.action_low, self.action_high)
            
            if a is None:
                return v, miu
            
            L = self.L_fc1(s)
            L = L.view(-1, self.action_dim, self.action_dim)
            
            tril_mask = torch.tril(torch.ones(
             self.action_dim, self.action_dim), diagonal=-1).unsqueeze(0)
            diag_mask = torch.diag(torch.diag(
             torch.ones(self.action_dim, self.action_dim))).unsqueeze(0)
                
            L = L * tril_mask.expand_as(L) + torch.exp(L) * diag_mask.expand_as(L)
            
            P = torch.bmm(L, L.transpose(2, 1))

            u_mu = (a - miu).unsqueeze(2)
            A = -0.5 * \
                torch.bmm(torch.bmm(u_mu.transpose(2, 1), P), u_mu)[:, :, 0]
            
            q = A + v
            
            return q
        
        
class DQN_fc_network(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(DQN_fc_network, self).__init__()
            self.fc1 = nn.Linear(input_dim, 30)
            self.fc2 = nn.Linear(30, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return x
        

class DDPG_critic_network(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        
        super(DDPG_critic_network, self).__init__()
        
        self.sfc1 = nn.Linear(state_dim, 30)
        self.sfc2 = nn.Linear(30,15)
        
        self.afc1 = nn.Linear(action_dim, 30)
        self.afc2 = nn.Linear(30,15)
        
        self.sharefc1 = nn.Linear(30,30)
        self.sharefc2 = nn.Linear(30,1)
        
    def forward(self, s, a):
        s = F.relu(self.sfc1(s))
        s = F.relu(self.sfc2(s))
        
        a = F.relu(self.afc1(a))
        a = F.relu(self.afc2(a))
        
        qsa = torch.cat((s,a), 1)
        qsa = F.relu(self.sharefc1(qsa))
        qsa = self.sharefc1(qsa)
        
        return qsa
    
class DDPG_actor_network(nn.Module):
    def __init__(self, state_dim, action_dim, action_low, action_high):
        
        super(DDPG_actor_network, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 30)
        self.fc2 = nn.Linear(30, action_dim)
        
        self.action_low, self.action_high = action_low, action_high
        
    def forward(self, s):
        
        s = F.relu(self.fc1(s))
        a = self.fc2(s)
        a = a.clamp(self.action_low, self.action_high)
        
        return a
    
class AC_v_fc_network(nn.Module):
    
    def __init__(self, state_dim):
        super(AC_v_fc_network, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30,1)
        
    def forward(self, s):
        s = F.relu(self.fc1(s))
        v = F.relu(self.fc2(s))
        v = self.fc3(v)
        
        return v
    
class AC_a_fc_network(nn.Module):
        def __init__(self, input_dim, output_dim):
            super(AC_a_fc_network, self).__init__()
            self.fc1 = nn.Linear(input_dim, 30)
            self.fc2 = nn.Linear(30, 30)
            self.fc3 = nn.Linear(30, output_dim)
            
        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            
            return F.softmax(x, dim = 1)
        

        