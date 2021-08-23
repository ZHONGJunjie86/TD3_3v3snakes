import os
from pathlib import Path
import sys
import torch
from torch import nn
from torch.distributions import Categorical
import numpy as np
import torch.nn.functional as F


HIDDEN_SIZE=256
device = torch.device("cuda" if torch.cuda.is_available() else"cpu")  

from typing import Union
Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': torch.nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}
        
def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map

def visual_ob(state):
    image = np.zeros((20, 10))
    for i in range(7):
        snake_i = state[i+1] #[[7, 0], [0, 0], [7, 17], [0, 16], [3, 5]]
        for cordinate in snake_i:#[7, 0]
            image[cordinate[1]][cordinate[0]] = i+1
    return image



class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,8, kernel_size=3, stride=1, padding=1) # 20104 -> 20108
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=0) # 20108 -> 18816
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0) # 18816 -> 16632
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0) # 16632 -> 14464
        self.linear_CNN_1 = nn.Linear(3584, 384) #14464 = 3584
        self.linear_CNN_2 = nn.Linear(128,24)
        self.linear_CNN_3 = nn.Linear(24,4)
        

    def forward(self, tensor_cv): #,batch_size
        # CV
        x = F.relu(self.conv1(tensor_cv))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print("",x.size()[0])
        x=  x.reshape(x.size()[0],1,3584)
        x = F.relu(self.linear_CNN_1(x)).reshape(x.size()[0],3,128)
        x = F.relu(self.linear_CNN_2(x))
        #action = F.relu(self.linear_CNN_3(x))+1e-5
        action = self.linear_CNN_3(x).clamp(1e-10,1e10)

        return action


class RLAgent(object):
    def __init__(self, obs_dim, act_dim, num_agent):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.output_activation = 'softmax'
        self.actor = Actor(obs_dim, act_dim, num_agent, self.output_activation).to(self.device)

    def choose_action(self, obs):
        obs = torch.Tensor([obs]).to(self.device)
        logits = self.actor(obs).cpu().detach().numpy()[0]
        return logits

    def select_action_to_env(self, obs, ctrl_index):
        logits = self.choose_action(obs)
        actions = logits2action(logits)
        action_to_env = to_joint_action(actions, ctrl_index)
        return action_to_env

    def load_model(self, filename):
        self.actor.load_state_dict(torch.load(filename, map_location='cpu')) #agent.torch.load(actor_net, map_location='cpu')


def to_joint_action(action, ctrl_index):
    joint_action_ = []
    action_a = action[ctrl_index]
    each = [0] * 4
    each[action_a] = 1
    joint_action_.append(each)
    return joint_action_


def logits2action(logits):
    logits = torch.Tensor(logits).to(device)
    actions = np.array([Categorical(out).sample().item() for out in logits])
    return np.array(actions)


Memory_size = 4
agent = RLAgent(26*Memory_size, 4, 3)
actor_net = os.path.dirname(os.path.abspath(__file__)) + "/actor_2000.pth"
agent.load_model(actor_net)
memory = []


def my_controller(observation_list, action_space_list, is_act_continuous):
    obs_dim = 26
    obs = observation_list.copy()
    board_width = obs['board_width']
    board_height = obs['board_height']
    o_index = obs['controlled_snake_index']  # 2, 3, 4, 5, 6, 7 -> indexs = [0,1,2,3,4,5]
    o_indexs_min = 3 if o_index > 4 else 0
    indexs = [o_indexs_min, o_indexs_min+1, o_indexs_min+2]

    observation = visual_ob(obs[0])/10
    #observation = get_observations(obs, indexs, obs_dim, height=board_height, width=board_width)/10

    #Memory
    if len(memory) !=0: 
        del memory[:1]
        memory.append(observation)
    else: 
        for _ in range(Memory_size): 
            memory.append(observation)
    observation = np.stack(memory)

    actions = agent.select_action_to_env(observation, indexs.index(o_index-2))
    return actions