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
    def __init__(self, obs_dim, act_dim, num_agents, args, output_activation='tanh'):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.linear_1 = nn.Linear(obs_dim, obs_dim)
        self.linear_2 = nn.Linear(obs_dim, obs_dim)

        #self.lstm = nn.LSTM(obs_dim , obs_dim , 1,batch_first=True)

        self.args = args

        #sizes_prev = [obs_dim, HIDDEN_SIZE,HIDDEN_SIZE]
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
        self.post_dense_1 = mlp(sizes_post, output_activation=output_activation)
        self.post_dense_2 = mlp(sizes_post, output_activation=output_activation)
        self.post_dense_3 = mlp(sizes_post, output_activation=output_activation)

    def forward(self, obs_batch):
        #x = F.relu(self.linear_1(obs_batch))
        #x = F.relu(self.linear_2(obs_batch))
        #x,_ = self.lstm( x)
        #out = self.prev_dense(x)
        batch_size = obs_batch.size()[0]
        out = self.prev_dense(obs_batch)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out_1 = self.post_dense_1(out)
        out_2 = self.post_dense_2(out)
        out_3 = self.post_dense_3(out)
        #print("out.size()",out.size())
        out = torch.vstack((out_1,out_2,out_3)).reshape(batch_size,3,4)
        return out

class Critic(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agents, args):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agents = num_agents

        self.linear_1 = nn.Linear(obs_dim + act_dim, obs_dim + act_dim)
        self.linear_2 = nn.Linear(obs_dim + act_dim, obs_dim + act_dim)

        self.lstm = nn.LSTM(obs_dim + act_dim, obs_dim + act_dim, 1,batch_first=True)

        self.args = args

        sizes_prev = [obs_dim + act_dim, HIDDEN_SIZE,HIDDEN_SIZE]
        #sizes_prev = [obs_dim + act_dim, HIDDEN_SIZE,HIDDEN_SIZE,HIDDEN_SIZE]

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
        x = torch.cat((obs_batch, action_batch), dim=-1)
        #x = F.relu(self.linear_1(x))
        #x = F.relu(self.linear_2(x))
        #x,_ = self.lstm( x)
        #out = self.prev_dense(x)
        out = self.prev_dense(x)

        if self.args.algo == "bicnet":
            out = self.comm_net(out)

        out_1 = self.post_dense_1(out)
        out_2 = self.post_dense_2(out)
        return out_1,out_2



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
