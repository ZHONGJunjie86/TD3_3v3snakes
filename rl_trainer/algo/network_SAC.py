from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.distributions import Normal
from torch.distributions import Categorical
import numpy as np

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim,  num_agent, args):
        super(QNetwork, self).__init__()
        self.act_dim = 3
        # Q1 architecture
        self.conv1_1 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464
        self.linear_CNN_1_1 = nn.Linear(3584, 128) #14464 = 3584
        self.linear_CNN_2_1 = nn.Linear(128,128)
        #
        self.linear_1_1 = nn.Linear(self.act_dim, 32)#32*3=96
        self.linear_2_1 = nn.Linear(32, 128)
        self.lstm_1 = nn.LSTM(128, 128, 1)
        #
        self.linear_3_1 = nn.Linear(256,64)
        self.linear_4_1 = nn.Linear(64,3)
        ##########################################
        # Q2 architecture
        self.conv1_2 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464
        self.linear_CNN_1_2 = nn.Linear(3584, 128) #14464 = 3584
        self.linear_CNN_2_2 = nn.Linear(128,128)
        #
        self.linear_1_2 = nn.Linear(self.act_dim, 32)#32*3=96
        self.linear_2_2 = nn.Linear(32, 128)
        self.lstm_2 = nn.LSTM(128, 128, 1)
        #
        self.linear_3_2 = nn.Linear(256,64)
        self.linear_4_2 = nn.Linear(64,3)


    def forward(self, tensor_cv, action_batch):
        # CV
        x = F.relu(self.conv1_1(tensor_cv))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv4_1(x))
        #
        batch_size = x.size()[0]
        x=  x.reshape(batch_size,1,3584)
        x = F.relu(self.linear_CNN_1_1(x))#.reshape(x.size()[0],3,128)
        x = F.relu(self.linear_CNN_2_1(x))
        x,_ = self.lstm_1(x)
        #
        action_batch = action_batch.reshape(batch_size,1,self.act_dim)
        y = F.relu(self.linear_1_1(action_batch))
        y = F.relu(self.linear_2_1(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        z = torch.relu(self.linear_3_1(z))
        out_1 = torch.tanh(self.linear_4_1(z)).reshape(batch_size,1,3)
        #out_1 = torch.vstack((o_1,o_2,o_3)).t().reshape(batch_size,3,1)
        #######################################################
        # CV
        x = F.relu(self.conv1_2(tensor_cv))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv4_2(x))
        #
        x=  x.reshape(batch_size,1,3584)#.repeat(1,3,1)
        x = F.relu(self.linear_CNN_1_2(x))#.reshape(x.size()[0],3,128)
        x = F.relu(self.linear_CNN_2_2(x))
        x,_ = self.lstm_2(x)
        #
        action_batch = action_batch.reshape(batch_size,1,self.act_dim)
        y = F.relu(self.linear_1_2(action_batch))
        y = F.relu(self.linear_2_2(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        z = torch.relu(self.linear_3_2(z))
        out_2 = torch.tanh(self.linear_4_2(z)).reshape(batch_size,1,3)

        return out_1,out_2

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agent, args): #num_inputs, num_actions, hidden_dim, action_space=None
        super(GaussianPolicy, self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        
        self.conv1 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464

        self.linear_1 = nn.Linear(3584, 128) #14464 = 3584

        self.mean_linear = nn.Linear(128, 3)
        self.log_std_linear = nn.Linear(128, 3)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale = 2
        self.action_bias = 2

        self.batch_size = 0

        """if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)"""

    def forward(self, tensor_cv):
        x = F.relu(self.conv1(tensor_cv))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=  x.reshape(x.size()[0],1,3584)
        self.batch_size = x.size()[0]
        x = F.relu(self.linear_1(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean

    #def to(self, device):
    #    self.action_scale = self.action_scale.to(self.device)
    #    self.action_bias = self.action_bias.to(self.device)
    #    return super(GaussianPolicy, self).to(self.device)


class DeterministicPolicy(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim, action_space=None):
        super(DeterministicPolicy, self).__init__()
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean = nn.Linear(hidden_dim, num_actions)
        self.noise = torch.Tensor(num_actions)

        self.apply(weights_init_)

        # action rescaling
        if action_space is None:
            self.action_scale = 1.
            self.action_bias = 0.
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = torch.tanh(self.mean(x)) * self.action_scale + self.action_bias
        return mean

    def sample(self, state):
        mean = self.forward(state)
        noise = self.noise.normal_(0., std=0.1)
        noise = noise.clamp(-0.25, 0.25)
        action = mean + noise
        return action, torch.tensor(0.), mean

    def to(self, device):
        self.action_scale = self.action_scale.to(device)
        self.action_bias = self.action_bias.to(device)
        self.noise = self.noise.to(device)
        return super(DeterministicPolicy, self).to(device)