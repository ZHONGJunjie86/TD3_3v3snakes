from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
import torch.nn as nn
import torch.nn.functional as F
import torch

HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464

        self.linear_1_1 = nn.Linear(3584, 128) #14464 = 3584
        self.linear_1_2 = nn.Linear(128,4)
        self.linear_2_1 = nn.Linear(3584, 128) #14464 = 3584
        self.linear_2_2 = nn.Linear(128,4)
        self.linear_3_1 = nn.Linear(3584, 128) #14464 = 3584
        self.linear_3_2 = nn.Linear(128,4)
        #self.linear_CNN_3 = nn.Linear(24,4)
        

    def forward(self, tensor_cv): #,batch_size
        # CV
        x = F.relu(self.conv1(tensor_cv))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=  x.reshape(x.size()[0],1,3584)
        batch_size = x.size()[0]
        #print("x.size()",x.size())
        #x= x.repeat(1,3,1)
        #print("after x.size()",x.size())
        #x = F.relu(self.linear_CNN_1(x)) #.reshape(x.size()[0],3,128)
        #action1 = F.relu(self.linear_CNN_2(x))
        #action = F.relu(self.linear_CNN_3(x))+1e-5
        #action = self.linear_CNN_3(x)   #.clamp(1e-10,1e10)
        a_1 = F.relu(self.linear_1_1(x)) #.reshape(x.size()[0],3,128)
        a_1 = F.relu(self.linear_1_2(a_1))
        #print("a_1",a_1)
        a_2 = F.relu(self.linear_2_1(x)) #.reshape(x.size()[0],3,128)
        a_2 = F.relu(self.linear_2_2(a_2))
        a_3 = F.relu(self.linear_3_1(x)) #.reshape(x.size()[0],3,128)
        a_3 = F.relu(self.linear_3_2(a_3))
        action = torch.vstack((a_1,a_2,a_3)).reshape(batch_size,3,4)
        #print("action.size()",action.size())

        return action


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()
        self.conv1_1 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2_1 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3_1 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4_1 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464
        self.linear_CNN_1_1 = nn.Linear(3584, 128) #14464 = 3584
        self.linear_CNN_2_1 = nn.Linear(128,128)
        #
        self.linear_1_1 = nn.Linear(12, 32)#32*3=96
        self.linear_2_1 = nn.Linear(32, 128)
        self.lstm_1 = nn.LSTM(128, 128, 1)
        #
        self.linear_3_1_1 = nn.Linear(256,64)
        self.linear_4_1_1 = nn.Linear(64,1)
        self.linear_3_1_2 = nn.Linear(256,64)
        self.linear_4_1_2 = nn.Linear(64,1)
        self.linear_3_1_3 = nn.Linear(256,64)
        self.linear_4_1_3 = nn.Linear(64,1)
        ##########################################
        self.conv1_2 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2_2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3_2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464
        self.linear_CNN_1_2 = nn.Linear(3584, 128) #14464 = 3584
        self.linear_CNN_2_2 = nn.Linear(128,128)
        #
        self.linear_1_2 = nn.Linear(12, 32)#32*3=96
        self.linear_2_2 = nn.Linear(32, 128)
        self.lstm_2 = nn.LSTM(128, 128, 1)
        #
        self.linear_3_2 = nn.Linear(256,64)
        self.linear_4_2 = nn.Linear(64,1)
        #
        self.linear_3_2_1 = nn.Linear(256,64)
        self.linear_4_2_1 = nn.Linear(64,1)
        self.linear_3_2_2 = nn.Linear(256,64)
        self.linear_4_2_2 = nn.Linear(64,1)
        self.linear_3_2_3 = nn.Linear(256,64)
        self.linear_4_2_3 = nn.Linear(64,1)


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
        action_batch = action_batch.reshape(batch_size,1,12)
        y = F.relu(self.linear_1_1(action_batch))
        y = F.relu(self.linear_2_1(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        o_1 = torch.relu(self.linear_3_1_1(z))
        o_1 = self.linear_4_1_1(o_1).reshape(1,batch_size)
        o_2 = torch.relu(self.linear_3_1_2(z))
        o_2 = self.linear_4_1_2(o_2).reshape(1,batch_size)
        o_3 = torch.relu(self.linear_3_1_3(z))
        o_3 = self.linear_4_1_3(o_3).reshape(1,batch_size)
        out_1 = torch.vstack((o_1,o_2,o_3)).t().reshape(batch_size,3,1)
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
        action_batch = action_batch.reshape(batch_size,1,12)
        y = F.relu(self.linear_1_2(action_batch))
        y = F.relu(self.linear_2_2(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        o_1 = torch.relu(self.linear_3_2_1(z))
        o_1 = self.linear_4_2_1(o_1).reshape(1,batch_size)
        o_2 = torch.relu(self.linear_3_2_2(z))
        o_2 = self.linear_4_2_2(o_2).reshape(1,batch_size)
        o_3 = torch.relu(self.linear_3_2_3(z))
        o_3 = self.linear_4_2_3(o_3).reshape(1,batch_size)
        out_2 = torch.vstack((o_1,o_2,o_3)).t().reshape(batch_size,3,1)

        return out_1,out_2 





