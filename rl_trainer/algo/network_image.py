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
        self.conv1 = nn.Conv2d(4,4, kernel_size=1, stride=1, padding=0) # 20*10*4 -> 20*10*4
        self.maxp1 = nn.MaxPool2d(3, stride = 1, padding=0) #  20*10*4 -> 18*8*4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0) # 18*8*4 -> 18*8*8
        self.maxp2 = nn.MaxPool2d(2, stride = 1, padding=0) #  18*8*8 -> 17*7*8
        self.linear_CNN_1 = nn.Linear(952, 129) #17*7*8 = 952
        self.linear_CNN_2 = nn.Linear(43,24)
        self.linear_CNN_3 = nn.Linear(24,4)
        

    def forward(self, tensor_cv): #,batch_size
        # CV
        x = F.relu(self.maxp1(self.conv1(tensor_cv)))
        x = F.relu(self.maxp2(self.conv2(x)))
        #print("",x.size()[0])
        x=  x.reshape(x.size()[0],1,952)
        x = F.relu(self.linear_CNN_1(x)).reshape(x.size()[0],3,43)
        x = F.relu(self.linear_CNN_2(x))
        action = F.tanh(self.linear_CNN_3(x))

        return action


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()
        
        self.conv1 = nn.Conv2d(4,4, kernel_size=1, stride=1, padding=0) # 20*10*4 -> 20*10*4
        self.maxp1 = nn.MaxPool2d(3, stride = 1, padding=0) #  20*10*4 -> 18*8*4
        self.conv2 = nn.Conv2d(4, 8, kernel_size=1, stride=1, padding=0) # 18*8*4 -> 18*8*8
        self.maxp2 = nn.MaxPool2d(2, stride = 1, padding=0) #  18*8*8 -> 17*7*8
        self.linear_CNN_1 = nn.Linear(952, 128) #13*8 = 104
        #
        self.linear_1 = nn.Linear(4, 32)#32*3=96
        self.linear_2 = nn.Linear(96, 128)
        #
        self.linear_3_1 = nn.Linear(256,64)
        self.linear_4_1 = nn.Linear(64,1)
        self.linear_3_2 = nn.Linear(256,64)
        self.linear_4_2 = nn.Linear(64,1)


    def forward(self, tensor_cv, action_batch):
        # CV
        x = F.relu(self.maxp1(self.conv1(tensor_cv)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x=  x.reshape(x.size()[0],1,952)
        x = F.relu(self.linear_CNN_1(x))
        #
        #print("action_batch.size()",action_batch.size())
        y = F.relu(self.linear_1(action_batch))
        #print("y.size()",y.size())
        y = y.reshape(y.size()[0],1,96)
        y = F.relu(self.linear_2(y))
        #
        #print("x y size",x.size(),y.size())
        z = torch.cat((x,y), dim=-1)
        out_1 = F.relu(self.linear_3_1(z))
        out_1 = F.tanh(self.linear_4_1(out_1))
        out_2 = F.relu(self.linear_3_2(z))
        out_2 = F.tanh(self.linear_4_2(out_2))
        #print("critic done")
        return out_1,out_2 





