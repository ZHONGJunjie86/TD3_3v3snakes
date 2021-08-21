from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from common import *
import torch.nn as nn
import torch.nn.functional as F

HIDDEN_SIZE = 256


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim, HIDDEN_SIZE,HIDDEN_SIZE,HIDDEN_SIZE]
        middle_prev = [HIDDEN_SIZE, HIDDEN_SIZE]
        sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, act_dim]


        if self.args.algo == "bicnet":
            self.comm_net = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
            sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, act_dim]

        elif self.args.algo == "ddpg":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, act_dim]

        #print("actor prev_dense")
        self.prev_dense = mlp(sizes_prev)
        #print("actor post_dense")
        self.post_dense = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        out = self.prev_dense(obs_batch)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out = self.post_dense(out)
        return out


class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.args = args

        sizes_prev = [obs_dim + act_dim, HIDDEN_SIZE,HIDDEN_SIZE,HIDDEN_SIZE]

        if self.args.algo == "bicnet":
            self.comm_net = LSTMNet(HIDDEN_SIZE, HIDDEN_SIZE)
            sizes_post = [HIDDEN_SIZE << 1, HIDDEN_SIZE, 1]

        elif self.args.algo == "ddpg":
            sizes_post = [HIDDEN_SIZE, HIDDEN_SIZE, 1]

        #print("critic prev_dense")
        self.prev_dense = mlp(sizes_prev)
        #print("critic post_dense_1")
        self.post_dense_1 = mlp(sizes_post)
        #print("critic post_dense_2")
        self.post_dense_2 = mlp(sizes_post)

    def forward(self, obs_batch, action_batch):
        out = torch.cat((obs_batch, action_batch), dim=-1)
        out = self.prev_dense(out)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out_1 = self.post_dense_1(out)
        out_2 = self.post_dense_2(out)
        return out_1,out_2 


class DUTNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 bidirectional=True):
        super().__init__()

        self.l1 = nn.Linear(input_size, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, hidden_size)

    def forward(self, data, ):
        x = F.relu(self.l1(data)) 
        x = F.relu(self.l2(x)) 
        output = F.relu(self.l3(x)) 
        return output



class LSTMNet(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 batch_first=True,
                 bidirectional=True):
        super(LSTMNet, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=batch_first,
            bidirectional=bidirectional
        )

    def forward(self, data, ):
        output, (_, _) = self.lstm(data)
        return output
