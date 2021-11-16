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

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
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
        self.conv1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0) # 20104 -> 1888
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0) # 1888 -> 16616

        self.lstm = nn.LSTM(192, 32, 1, batch_first=True) #128

        self.linear_1 = nn.Linear(128, 12)
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

        self.action_scale = 2
        self.action_bias = 2        
        self.epsilon = 1e-6

    def forward(self, tensor_cv,num_state,h_state=(torch.zeros(1,1,32).to(device),torch.zeros(1,1,32).to(device))): #,batch_size
        #i_1 = tensor_cv.unsqueeze(0).detach()
        # CV
        tensor_cv = tensor_cv.unsqueeze(0)
        x = F.relu(self.conv1(tensor_cv))
        x = tensor_cv + x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = F.relu(self.conv4(x))
        x=  x.reshape(1,4,192)

        x,h_state = self.lstm(x,h_state)

        x = x.reshape(1,128)

        prob = self.soft_max(self.linear_1(x).reshape(3,4))
        #print("log_prob",log_prob)
        dis = self.Categorical(prob)
        action = dis.sample()
        #print("action",action)
        #samples_2d = torch.multinomial(log_prob.reshape(batch_size,12), num_samples=1, replacement=True)
        #print("samples_2d",samples_2d)
        #action = samples_2d.reshape(i.size(0))
        log_prob = dis.log_prob(action)
        #print("log_prob",log_prob)

        entropy = -torch.exp(log_prob) * log_prob

        return action.detach(),log_prob,entropy,(h_state[0].data,h_state[1].data) #.sum(2)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        ####### 1
        self.conv1_1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_1 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0) # 20104 -> 1888
        self.conv3_1 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0) # 1888 -> 16616
        #
        self.lstm_1 = nn.LSTM(192, 32, 1, batch_first=True)
        #
        self.linear_num_1_1 = nn.Linear(6,64)
        self.linear_num_2_1 = nn.Linear(64,64)
        #
        self.linear_1 = nn.Linear(192,3)
        ####### 2
        self.conv1_2 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=0) # 20104 -> 1888
        self.conv3_2 = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=0) # 1888 -> 16616
        #
        self.lstm_2 = nn.LSTM(192, 32, 1, batch_first=True)
        #
        self.linear_num_1_2 = nn.Linear(6,64)
        self.linear_num_2_2 = nn.Linear(64,64)
        #
        self.linear_2 = nn.Linear(192,3)


    def forward(self, tensor_cv,num_state, h_state_1=(torch.zeros(1,1,32).to(device),torch.zeros(1,1,32).to(device)),
                                 h_state_2=(torch.zeros(1,1,32).to(device),torch.zeros(1,1,32).to(device))):
        #i_1 = tensor_cv.unsqueeze(0).detach()
        tensor_cv = tensor_cv.unsqueeze(0)
        # CV
        x = F.relu(self.conv1_1(tensor_cv))
        x = tensor_cv + x
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv3_1(x)).reshape(1,4,192)
        #
        x,h_state_1 = self.lstm_1(x,h_state_1)
        x = x.reshape(1,128)
        #
        y = F.relu(self.linear_num_1_1(num_state))
        y = F.relu(self.linear_num_2_1(y))
        #
        z = torch.cat((x,y), dim=-1)
        out_1 = torch.tanh(self.linear_1(z)).reshape(3) #1

        ###################################################################
        # CV
        x = F.relu(self.conv1_2(tensor_cv))
        x = tensor_cv + x
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv3_2(x))
        #
        x=  x.reshape(1,4,192)

        x,h_state_2 = self.lstm_2(x,h_state_2)
        x = x.reshape(1,128)
        #
        y = F.relu(self.linear_num_1_2(num_state))
        y = F.relu(self.linear_num_2_2(y))
        #
        z = torch.cat((x,y), dim=-1)
        out_2 = torch.tanh(self.linear_2(z)).reshape(3) #1
       
        return out_1,out_2, (h_state_1[0].data,h_state_1[1].data),(h_state_2[0].data,h_state_2[1].data)

class Actor_Critic:
    def __init__(self,args):
        
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.gama = 0.9
        self.eps = args.epsilon
        self.tau = 0.1

        self.log_prob = 0
        self.entropy = 0
        self.a_loss = 0
        self.c_loss = 0

        
        self.value_1 = 0
        self.value_2 = 0
        self.h_state_1 = 0
        self.h_state_2 = 0
        self.h_state_a = 0

        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.critic_next = Critic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)

        self.update_deday = 0

    # Random process N using epsilon greedy
    def choose_action(self, obs, num_state,first_time = True):

        obs = torch.Tensor(obs).to(self.device)
        #print("obs.size()",obs.size())
        
        if first_time:
            action,self.log_prob,self.entropy,self.h_state_a = self.actor(obs,num_state)
            self.value_1,self.value_2,self.h_state_1,self.h_state_2 =  self.critic(obs,num_state)  
        else:
            action,self.log_prob,self.entropy,self.h_state_a = self.actor(obs,num_state,self.h_state_a)
            self.value_1,self.value_2,self.h_state_1,self.h_state_2 =  self.critic(obs,num_state,self.h_state_1,self.h_state_2)  
        
        action = action.cpu().detach().numpy()

        return action

    def update(self,new_lr,next_obs,num_state_next,reward,done):
        self.update_deday += 1

        if new_lr != self.a_lr:
            print("new_lr",new_lr) 
            self.a_lr = new_lr
            self.c_lr = new_lr
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),  self.a_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  self.c_lr)
        
        next_obs = torch.Tensor(next_obs).to(device)

        value_next_1,value_next_2,_,_ = self.critic_next(next_obs,num_state_next,self.h_state_1,self.h_state_2)   #_next
        value_next = torch.min(value_next_1,value_next_2)  ## TD3 style: min target

        with torch.autograd.set_detect_anomaly(True):
            # TD:r(s) +  gama*v(s+1) - v(s)
            reward = torch.Tensor(reward).to(self.device)
            done = torch.Tensor(done).to(self.device)
            R_TD = reward.detach() + self.gama*value_next.detach()*(1 - done)
            loss_critic =torch.nn.SmoothL1Loss()(self.value_1, R_TD) +  torch.nn.SmoothL1Loss()(self.value_2, R_TD)

            self.critic_optimizer.zero_grad()
            loss_critic.backward()  
            self.critic_optimizer.step()
            self.critic_next.load_state_dict(self.critic.state_dict())

            # update actor
            #if self.update_deday!=0 and self.update_deday % 2 == 0:
                # mean_value style 
                # advantage = reward.detach() + self.gama*(value_next_1+value_next_2)/2 - (self.value_1+self.value_2)/2
                # min advantage style
                # advantage = min(r + Q' - Q) = r + min(Q' - Q) =  r + minQ' - maxQ 
                ## SAC style: when maximize the min_qf_pi use the min
                # advantage = reward.detach() + self.gama*value_next - torch.max(self.value_1,self.value_2)
                # advantage_Underestimating = r + minQ' - maxQ (V) 
            advantage = reward.detach() + self.gama*value_next - torch.max(self.value_1,self.value_2)#/2
            loss_actor = -(self.log_prob * advantage.detach() + 5*self.a_lr *self.entropy).mean()
        
            self.actor_optimizer.zero_grad()
            loss_actor.backward()
            self.actor_optimizer.step()
            self.a_loss += loss_actor.item()

            #update target critic
            """if self.update_deday!=0 and self.update_deday % 1 == 0:
                for src_param, tgt_param in zip(self.critic.parameters(), self.critic_next.parameters()):
                    tgt_param.data.copy_(
                        src_param.data * self.tau + tgt_param.data * (1.0 - self.tau) 
                    )"""

        self.c_loss += loss_critic.item()
      
        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss
    
    def update_critic(self):
        self.critic_next.load_state_dict(self.critic.state_dict())

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)

        model_actor_path = os.path.join(base_path, "actor_" + ".pth") #+ str(episode) + str(episode) 
        model_critic_path = os.path.join(base_path, "critic_" + ".pth")
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

"""

class Actor(nn.Module):
    def __init__(self):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104

        self.lstm = nn.LSTM(200, 32, 1, batch_first=True) #128

        self.MU = nn.Linear(128,3)#(256,3)
        self.STD = nn.Linear(128,3)#(256,3)

        self.action_scale = 2
        self.action_bias = 2        
        self.epsilon = 1e-6

    def forward(self, tensor_cv,num_state,h_state=(torch.zeros(1,1,32).to(device),torch.zeros(1,1,32).to(device))): #,batch_size
        i_1 = tensor_cv.unsqueeze(0).detach()
        # CV
        x = F.relu(self.conv1(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3(i_3))
        i_4 = i_3 + x
        i_4 = i_4.reshape(1,4,200)

        x,h_state = self.lstm(i_4,h_state)

        x = x.reshape(1,128)

        mean = self.MU(x)
        std = self.STD(x).clamp(-20,2)
        std = std.exp()
        normal = Normal(mean, std)
        
        x_t = normal.sample()  # for reparameterization trick (mean + std * N(0,1))

        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias

        log_prob = normal.log_prob(x_t)
        #log_prob = log_prob.sum(1, keepdim=True)
        entropy = -torch.exp(log_prob) * log_prob
        # Enforcing Action Bound
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        #log_prob = log_prob.sum(1, keepdim=True) 
        #mean =torch.diag( self.MU(x))
        #std = torch.diag(self.STD(x).clamp(-20,2).exp()).unsqueeze(0)
        #normal = MultivariateNormal(mean, std)

        return action.detach(),log_prob,entropy,(h_state[0].data,h_state[1].data)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        ####### 1
        self.conv1_1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 1888
        self.conv3_1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 1888 -> 16616
        #
        self.lstm_1 = nn.LSTM(200, 32, 1, batch_first=True)
        #
        self.linear_num_1_1 = nn.Linear(6,64)
        self.linear_num_2_1 = nn.Linear(64,64)
        #
        self.linear_1 = nn.Linear(192,3)
        ####### 2
        self.conv1_2 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 1888
        self.conv3_2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 1888 -> 16616
        #
        self.lstm_2 = nn.LSTM(200, 32, 1, batch_first=True)
        #
        self.linear_num_1_2 = nn.Linear(6,64)
        self.linear_num_2_2 = nn.Linear(64,64)
        #
        self.linear_2 = nn.Linear(192,3)


    def forward(self, tensor_cv,num_state, h_state_1=(torch.zeros(1,1,32).to(device),torch.zeros(1,1,32).to(device)),
                                 h_state_2=(torch.zeros(1,1,32).to(device),torch.zeros(1,1,32).to(device))):
        i_1 = tensor_cv.unsqueeze(0).detach()
        # CV
        x = F.relu(self.conv1_1(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2_1(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3_1(i_3))
        i_4 = i_3 + x
        i_4 = i_4.reshape(1,4,200)
        #
        x,h_state_1 = self.lstm_1(i_4,h_state_1)
        x = x.reshape(1,128)
        #
        y = F.relu(self.linear_num_1_1(num_state))
        y = F.relu(self.linear_num_2_1(y))
        #
        z = torch.cat((x,y), dim=-1)
        out_1 = torch.tanh(self.linear_1(z)).reshape(1,3) #1

        ###################################################################
        # CV
        i_1 = tensor_cv.unsqueeze(0).detach()
        # CV
        x = F.relu(self.conv1_2(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2_2(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3_2(i_3))
        i_4 = i_3 + x
        i_4 = i_4.reshape(1,4,200)
        #
        x,h_state_2 = self.lstm_2(i_4,h_state_2)
        x = x.reshape(1,128)
        #
        y = F.relu(self.linear_num_1_2(num_state))
        y = F.relu(self.linear_num_2_2(y))
        #
        z = torch.cat((x,y), dim=-1)
        out_2 = torch.tanh(self.linear_2(z)).reshape(1,3) #1
       
        return out_1,out_2, (h_state_1[0].data,h_state_1[1].data),(h_state_2[0].data,h_state_2[1].data)


"""