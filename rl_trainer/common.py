import numpy as np
import torch
import torch.nn as nn
import math
import copy
from typing import Union
from torch.distributions import Categorical
import os
import yaml

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

def hard_update(source, target):
    target.load_state_dict(source.state_dict())


def soft_update(source, target, tau):
    for src_param, tgt_param in zip(source.parameters(), target.parameters()):
        tgt_param.data.copy_(
            tgt_param.data * (1.0 - tau) + src_param.data * tau
        )


Activation = Union[str, nn.Module]

_str_to_activation = {
    'relu': nn.ReLU(),
    'tanh': nn.Tanh(),
    'identity': nn.Identity(),
    'softmax': nn.Softmax(dim=-1),
}


def mlp(sizes,
        activation: Activation = 'relu',
        output_activation: Activation = 'identity'):
    if isinstance(activation, str):
        activation = _str_to_activation[activation]
    if isinstance(output_activation, str):
        output_activation = _str_to_activation[output_activation]

    layers = []
    for i in range(len(sizes) - 1):
        act = activation if i < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[i], sizes[i + 1]), act]
    #print(nn.Sequential(*layers))
    return nn.Sequential(*layers)


def make_grid_map(board_width, board_height, beans_positions:list, snakes_positions:dict):
    snakes_map = [[[0] for _ in range(board_width)] for _ in range(board_height)]
    for index, pos in snakes_positions.items():
        for p in pos:
            snakes_map[p[0]][p[1]][0] = index

    for bean in beans_positions:
        snakes_map[bean[0]][bean[1]][0] = 1

    return snakes_map


def get_min_bean(x, y, beans_position):
    min_distance = math.inf
    min_x = beans_position[0][1]
    min_y = beans_position[0][0]
    index = 0
    for i, (bean_y, bean_x) in enumerate(beans_position):
        distance = math.sqrt((x - bean_x) ** 2 + (y - bean_y) ** 2)
        if distance < min_distance:
            min_x = bean_x
            min_y = bean_y
            min_distance = distance
            index = i
    return min_x, min_y, index


def greedy_snake(state_map, beans, snakes, width, height, ctrl_agent_index):
    beans_position = copy.deepcopy(beans)
    actions = []
    for i in ctrl_agent_index:
        head_x = snakes[i][0][1]
        head_y = snakes[i][0][0]
        head_surrounding = get_surrounding(state_map, width, height, head_x, head_y)
        bean_x, bean_y, index = get_min_bean(head_x, head_y, beans_position)
        beans_position.pop(index)

        next_distances = []
        up_distance = math.inf if head_surrounding[0] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y - 1) % height - bean_y) ** 2)
        next_distances.append(up_distance)
        down_distance = math.inf if head_surrounding[1] > 1 else \
            math.sqrt((head_x - bean_x) ** 2 + ((head_y + 1) % height - bean_y) ** 2)
        next_distances.append(down_distance)
        left_distance = math.inf if head_surrounding[2] > 1 else \
            math.sqrt(((head_x - 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(left_distance)
        right_distance = math.inf if head_surrounding[3] > 1 else \
            math.sqrt(((head_x + 1) % width - bean_x) ** 2 + (head_y - bean_y) ** 2)
        next_distances.append(right_distance)
        actions.append(next_distances.index(min(next_distances)))
    return actions


def visual_ob(state):
    image = np.zeros((20, 10))
    for i in range(7):
        snake_i = state[i+1] #[[7, 0], [0, 0], [7, 17], [0, 16], [3, 5]]
        for cordinate in snake_i:#[7, 0]
            image[cordinate[1]][cordinate[0]] = i+1
    return image

# Self position:        0:head_x; 1:head_y
# Head surroundings:    2:head_up; 3:head_down; 4:head_left; 5:head_right
# Beans positions:      (6, 7) (8, 9) (10, 11) (12, 13) (14, 15)
# Other snake positions: (16, 17) (18, 19) (20, 21) (22, 23) (24, 25) -- (other_x - self_x, other_y - self_y)
def get_observations(state, agents_index, obs_dim, height, width):
    state_copy = state.copy()
    #image = visual_ob(state)
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    observations = np.zeros((3, obs_dim))
    snakes_position = np.array(snakes_positions_list, dtype=object)
    beans_position = np.array(beans_positions, dtype=object).flatten()
    for i in agents_index:
        # self head position
        observations[i][:2] = snakes_position[i][0][:]

        # head surroundings
        head_x = snakes_position[i][0][1]
        head_y = snakes_position[i][0][0]
        head_surrounding = get_surrounding(state, width, height, head_x, head_y)
        observations[i][2:6] = head_surrounding[:]

        # beans positions
        observations[i][6:16] = beans_position[:]

        # other snake positions
        snake_heads = np.array([snake[0] for snake in snakes_position])
        snake_heads = np.delete(snake_heads, i, 0)
        observations[i][16:] = snake_heads.flatten()[:]

        #print(observations,type(observations))
    return observations#,image


def get_reward(info, snake_index, reward,punishiment_lock, score):\

    snakes_position = np.array(info['snakes_position'], dtype=object)
    beans_position = np.array(info['beans_position'], dtype=object)
    snake_heads = [snake[0] for snake in snakes_position]
    step_reward = np.zeros(len(snake_index))
    
    ###关于长度
    for i in snake_index:
        #周围距离
        self_head = np.array(snake_heads[i])
        #dists_bean = [np.sqrt(np.sum(np.square(beans_head - self_head))) for beans_head in beans_position]
        dists_body = []
        """for j in range (6):
            if j != i:
                dists_body = [np.sqrt(np.sum(np.square(np.array(snakes_body) - np.array(snake_heads[i])))) 
                                for snakes_body in snakes_position]
        if score == 1:    #结束AI赢
            step_reward[i] += 0.05
        elif score == 2:   #结束random赢
            step_reward[i] -= 0.05
        elif score == 0:   #平 一样长
            step_reward[i] = 0
        """
        if score == 4: #未结束random长
            ###关于吃豆
            if reward[i] > 0:  #吃到
                step_reward[i] += 0.02        
            """else:              #没吃到看距离 / 锁
                if min(dists_body) >= 2 and punishiment_lock[i] == 0:
                    #print("min(min(dists_bean)/100-0.01,0.02) ",min(min(dists_bean)/100-0.04,0.02) )
                    step_reward[i] -= min(min(dists_bean)/1000-0.003,0.004) 

                if punishiment_lock[i]>0:
                    step_reward[i] -= min(min(dists_bean)/1000-0.004,0)"""

        else:   #平局或AI长
            if reward[i] > 0:  #吃到
                step_reward[i] += 0.04        
            """else:              #没吃到看距离 / 锁
                if min(dists_body) >= 2 and punishiment_lock[i] == 0:
                    #print("min(min(dists_bean)/100-0.01,0.02) ",min(min(dists_bean)/100-0.04,0.02) )
                    step_reward[i] -= min(min(dists_bean)/1000-0.005,0.002) #0.01?

                if punishiment_lock[i]>0:
                    step_reward[i] -= min(min(dists_bean)/1000-0.006,0)"""
                
        ###关于碰撞
        #if reward[i] < 0:
        #    step_reward[i] -= 0.05
        step_reward[i] -= 0.06*info["hit"][i] #8

        ##关于对方碰撞
        if True in (reward[3:] <0):
            step_reward[i] += 0.02 #0.01

        ##关于dui方分多
        if np.sum(reward[:3]) > np.sum(reward[3:]):
            step_reward[i] -= 0.04#step_reward[i] += 0.01#5
        """elif np.sum(reward[:3]) == np.sum(reward[3:]):  #for 博弈
            step_reward[i] -= 0
        else:
            step_reward[i] -= 0.02 #-0.015
        if info["hit"][i] == 1 or min(dists_body) < 2 or reward[i] > 0:
            punishiment_lock[i] = 6
        else:
            punishiment_lock[i] = max(punishiment_lock[i] - 1, 0)"""

    return step_reward*10


def logits_random(act_dim, logits):
    logits = torch.Tensor(logits).to(device)
    acs = [Categorical(out).sample().item() for out in logits]
    num_agents = len(logits)
    actions = np.random.randint(act_dim, size=num_agents << 1)
    actions[:num_agents] = acs[:]
    return actions


def logits_greedy(state, logits, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

     
    logits = torch.Tensor(logits).to(device)
    logits = logits.reshape(3,4)
    #print(logits)
    logits_action = np.array([Categorical(out).sample().item() for out in logits])
    #logits_action = np.array([out.argmax(dim=0) for out in logits]) #-1每个行向量为一个
        

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list

def logits_AC(state, logits, height, width):
    state_copy = state.copy()
    board_width = state_copy['board_width']
    board_height = state_copy['board_height']
    beans_positions = state_copy[1]
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snake_map = make_grid_map(board_width, board_height, beans_positions, snakes_positions)
    state_ = np.array(snake_map)
    state = np.squeeze(state_, axis=2)

    beans = state_copy[1]
    # beans = info['beans_position']
    snakes_positions = {key: state_copy[key] for key in state_copy.keys() & {2, 3, 4, 5, 6, 7}}
    snakes_positions_list = []
    for key, value in snakes_positions.items():
        snakes_positions_list.append(value)
    snakes = snakes_positions_list

    #logits = torch.Tensor(logits).to(device)
    logits = np.trunc(logits)
    logits_action = np.array([out for out in logits])

    greedy_action = greedy_snake(state, beans, snakes, width, height, [3, 4, 5])

    action_list = np.zeros(6)
    action_list[:3] = logits_action
    action_list[3:] = greedy_action

    return action_list



def get_surrounding(state, width, height, x, y):
    surrounding = [state[(y - 1) % height][x],  # up
                   state[(y + 1) % height][x],  # down
                   state[y][(x - 1) % width],  # left
                   state[y][(x + 1) % width]]  # right

    return surrounding


def save_config(args, save_path):
    file = open(os.path.join(str(save_path), 'config.yaml'), mode='w', encoding='utf-8')
    yaml.dump(vars(args), file)
    file.close()