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
        self.conv1_1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 1888
        self.conv3_1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 1888 -> 16616
        self.linear_cnn_1 = nn.Linear(800, 128)
        #
        self.linear_1_1 = nn.Linear(3, 32)
        self.linear_2_1 = nn.Linear(32, 128)
        #
        self.linear_1 = nn.Linear(256,3)
        ##########################################
        # Q2 architecture
        self.conv1_2 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 1888
        self.conv3_2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 1888 -> 16616
        self.linear_cnn_2 = nn.Linear(800, 128)
        #
        self.linear_1_2 = nn.Linear(3, 32)
        self.linear_2_2 = nn.Linear(32, 128)
        #
        self.linear_2 = nn.Linear(256,3)

    def forward(self, tensor_cv, action_batch):
        # CV
        batch_size = tensor_cv.size()[0]
        i = tensor_cv
        x = F.relu(self.conv1_1(i))
        i = i + x
        x = F.relu(self.conv2_1(i))
        i = i + x
        x = F.relu(self.conv3_1(i))
        #
        x=  x.reshape(batch_size,1,800)
        x = F.relu(self.linear_cnn_1(x))
        #
        action_batch = action_batch.reshape(batch_size,1,self.act_dim)
        y = F.relu(self.linear_1_1(action_batch))
        y = F.relu(self.linear_2_1(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        out_1 = torch.tanh(self.linear_1(z)).reshape(batch_size,1,3)
        #out_1 = torch.vstack((o_1,o_2,o_3)).t().reshape(batch_size,3,1)
        #######################################################
        # CV
        i = tensor_cv
        x = F.relu(self.conv1_2(i))
        i = i + x
        x = F.relu(self.conv2_2(i))
        i = i + x
        x = F.relu(self.conv3_2(i))
        #
        x=  x.reshape(batch_size,1,800)
        x = F.relu(self.linear_cnn_2(x))
        #
        action_batch = action_batch.reshape(batch_size,1,self.act_dim)
        y = F.relu(self.linear_1_2(action_batch))
        y = F.relu(self.linear_2_2(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        out_2 = torch.tanh(self.linear_2(z)).reshape(batch_size,1,3)

        return out_1,out_2

class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agent, args): #num_inputs, num_actions, hidden_dim, action_space=None
        super(GaussianPolicy, self).__init__()
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.conv1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104

        self.mean_linear = nn.Linear(800, 3)
        self.log_std_linear = nn.Linear(800, 3)
        #self.linear = nn.Linear(800, 12)

        self.apply(weights_init_)

        # action rescaling
        self.action_scale = 2
        self.action_bias = 2

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        self.batch_size = 0
        
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

        """if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)"""

    def forward(self, tensor_cv):
        batch_size = tensor_cv.size()[0]
        i_1 = tensor_cv
        # CV
        x = F.relu(self.conv1(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3(i_3))
        i_4 = i_3 + x
        i_4 = i_4.reshape(batch_size,1,800)
        
        mean = self.mean_linear(i_4)
        log_std = self.log_std_linear(i_4)
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    #def to(self, device):
    #    self.action_scale = self.action_scale.to(self.device)
    #    self.action_bias = self.action_bias.to(self.device)
    #    return super(GaussianPolicy, self).to(self.device)

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.clamp(-20, 2).exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action_run = y_t * self.action_scale + self.action_bias



        '''compute logprob according to mean and std of action (stochastic policy)'''
        # # self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))
        # logprob = a_std_log + self.sqrt_2pi_log + noise.pow(2).__mul__(0.5)  # noise.pow(2) * 0.5
        # different from above (gradient)
        noise = torch.randn_like(mean, requires_grad=True)
        action = mean + std * noise
        a_tan = action.tanh()
        delta = ((mean - action) / std).pow(2).__mul__(0.5)
        log_prob = log_std + self.sqrt_2pi_log + delta
        # same as below:
        # from torch.distributions.normal import Normal
        # logprob_noise = Normal(a_avg, a_std).logprob(a_noise)
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()
        # same as below:
        # a_delta = (a_avg - a_noise).pow(2) /(2*a_std.pow(2))
        # logprob_noise = -a_delta - a_std.log() - np.log(np.sqrt(2 * np.pi))
        # logprob = logprob_noise + (-a_noise_tanh.pow(2) + 1.000001).log()

        #log_prob = log_prob + (-a_tan.pow(2) + 1.000001).log()  # fix logprob using the derivative of action.tanh()
        
        # same as below:
        # epsilon = 1e-6
        # logprob = logprob_noise - (1 - a_noise_tanh.pow(2) + epsilon).log()

        
        #log_prob = log_prob.sum(1, keepdim=True)
        

        return action_run.clamp(0,3.99), log_prob, 0 #mean



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



"""
def forward(self, tensor_cv):
        batch_size = tensor_cv.size()[0]
        i_1 = tensor_cv
        # CV
        x = F.relu(self.conv1(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3(i_3))
        i_4 = i_3 + x
        i_4 = i_4.reshape(batch_size,1,800)
        
        #mean = self.mean_linear(i_4)
        #log_std = self.log_std_linear(i_4)
        #log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        probability = self.soft_max(self.linear(i_4))
        return probability       #mean, log_std

    def sample(self, state):
        batch_size = state.size()[0]
        probability = self.forward(state).reshape(batch_size,3,4)
        print("self.Categorical.has_rsample",self.Categorical.has_rsample)
        dis = self.Categorical(*probability)
        action = dis.rsample()
        log_prob = dis.log_prob(action)

        entropy = -torch.exp(log_prob) * log_prob

        #params = policy_network(state)
        #m = self.Normal(*params)
        # Any distribution with .has_rsample == True could work based on the application
        #action = m.rsample()
        #next_state, reward = env.step(action)  # Assuming that reward is differentiable
        #loss = -reward
        #loss.backward()

        return action.detach(), log_prob, 0 #mean


"""
"""原写法
        log_prob = normal.log_prob(y_t) #(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2))+ epsilon ) # ????? 
        # 在此基础上，PPO计算新旧策略的熵使用的是经过tanh之前的动作，
        # 而SAC计算策略的熵使用了tanh之后的动作，因而在动作均值接近动作边界时，
        # 即便方差很大，它策略的熵不见得很大。
        # 因此SAC为了抵消tanh的影响，在计算策略熵的时候，添加了 tanh(a) 的导数项 [公式] 作为修正
        # 加上极小值 epsilon 是为了防止log计算溢出。
        # 然而，当动作接近边界值 -1 或 1时，这里的计算误差非常大，甚至会导致梯度方向出现错误。

        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias"""