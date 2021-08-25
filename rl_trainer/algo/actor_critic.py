from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
import torch.nn as nn
import torch.nn.functional as F
import torch
from common import soft_update, hard_update, device
import numpy as np

from torch.distributions import Categorical
from torch.distributions import Normal

HIDDEN_SIZE = 256


torch.set_default_tensor_type(torch.DoubleTensor)

class Memory:
    def __init__(self):
        self.m_obs = []
        self.m_obs_next = []
    
    def clear_memory(self):
        del self.m_obs[:]
        del self.m_obs_next[:]

class Actor(nn.Module):
    def __init__(self):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464

        self.linear_1 = nn.Linear(3584, 256) #14464 = 3584
        self.linear_2 = nn.Linear(256,64)
        self.MU = nn.Linear(64,3)
        self.STD = nn.Linear(64,3)

        self.action_scale = 2
        self.action_bias = 2        
        self.epsilon = 1e-6

    def forward(self, tensor_cv): #,batch_size
        # CV
        x = F.relu(self.conv1(tensor_cv.unsqueeze(0)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.reshape(1,3584)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))

        mean = self.MU(x)
        std = self.STD(x).clamp(-20,2)
        std = std.exp()
        normal = Normal(mean, std)
        
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True) 

        entropy = -torch.exp(log_prob) * log_prob

        return action.detach(),log_prob,entropy



class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464
        self.linear_CNN_1_1 = nn.Linear(896, 128) #14464 = 3584
        self.linear_CNN_2_1 = nn.Linear(128,128)
        #
        self.lstm = nn.LSTM(128, 128, 1, batch_first=True)
        #
        self.linear_3 = nn.Linear(512,64)
        self.linear_4 = nn.Linear(64,3)


    def forward(self, tensor_cv, h_state=(torch.zeros(1,4,128).to(device),
                                               torch.zeros(1,4,128).to(device))):
        # CV
        x = F.relu(self.conv1_1(tensor_cv.unsqueeze(0)))
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv4_1(x)).reshape(4,1,896)   
        #
        x = F.relu(self.linear_CNN_1_1(x))
        x = F.relu(self.linear_CNN_2_1(x))
        x,h_state = self.lstm(x,h_state)

        x = x.reshape(1,512)

        z = torch.relu(self.linear_3(x))
        out = torch.tanh(self.linear_4(z)).reshape(1,3)
       
        return out,(h_state[0].data,h_state[1].data)

class Actor_Critic:
    def __init__(self,args):
        
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.gama = 0.9
        self.value = 0
        self.eps = args.epsilon

        self.log_prob = 0
        self.entropy = 0
        self.a_loss = 0
        self.c_loss = 0

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.critic_next = Critic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False, first_time = False):

        obs = torch.Tensor(obs).to(self.device)
        #print("obs.size()",obs.size())
        action,self.log_prob,self.entropy = self.actor(obs)
        action = action.cpu().detach().numpy()
        if first_time:
            self.value,self.h_state=  self.critic(obs,self.h_state)  
        else:
            self.value,self.h_state=  self.critic(obs)  

        return action

    def update(self,new_lr,next_obs,reward,done):

        if new_lr != self.a_lr:
            print("new_lr",new_lr)
            self.a_lr = new_lr
            self.c_lr = new_lr
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),  self.a_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  self.c_lr)
        
        next_obs = torch.Tensor(next_obs).to(device)

        value_next,_ = self.critic_next(next_obs,self.h_state)   #_next

        with torch.autograd.set_detect_anomaly(True):
            # TD:r(s) +  gama*v(s+1) - v(s)
            reward = torch.Tensor(reward).to(self.device)
            advantage = reward.detach() + self.gama*value_next.detach() - self.value 
            loss_actor = -(self.log_prob * advantage.detach() + 5*self.a_lr *self.entropy).mean()
            loss_critic =torch.nn.SmoothL1Loss()(reward.detach() + self.gama*value_next.detach()*(1 - done) , self.value)

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            loss_critic.backward()  
            loss_actor.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            self.critic_next.load_state_dict(self.critic.state_dict())

        self.a_loss += loss_actor.item()
        self.c_loss += loss_critic.item()
      
        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor = torch.load(model_actor_path, map_location=device)
            critic = torch.load(model_critic_path, map_location=device)
            self.actor.load_state_dict(actor)
            self.critic.load_state_dict(critic)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        print("---------------save-------------------")
        base_path = os.path.join(run_dir, 'trained_model')
        print("new_lr: ",self.a_lr)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_"  + ".pth") #+ str(episode)
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_" + ".pth") #+ str(episode) 
        torch.save(self.critic.state_dict(), model_critic_path)