import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update, device
from algo.network_SAC import *
import torch.nn.functional as F

class SAC:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.total_it = 0 
        
        self.device = device

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon

        # 
        self.critic = QNetwork(obs_dim, act_dim,  num_agent, args).to(device=self.device)
        self.critic_target = QNetwork(obs_dim, act_dim,  num_agent, args).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr*2)
        #
        hard_update(self.critic, self.critic_target)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(3).to(self.device)).item() #action_space.shape
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.a_lr)

            self.actor = GaussianPolicy(obs_dim, act_dim,  num_agent, args).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.actor = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)

        # Initialise replay buffer R
        self.replay_buffer = ReplayBuffer(args.buffer_size, args.batch_size)

        self.c_loss = 0
        self.a_loss = 0     
        self.alpha_loss = 0

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=True):
        obs = torch.Tensor([obs]).to(self.device)

        """if evaluation is False:
            action, _, _ = self.actor.sample(obs)
        else:
            _, _, action = self.actor.sample(obs)"""

        action, _, _ = self.actor.sample(obs)

        #print("action.cpu().detach().numpy()[0]",action.cpu().detach().numpy()[0])

        return action.cpu().detach().numpy()[0]

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))

    def update(self,new_lr):
        self.total_it += 1

        if new_lr != self.a_lr:
            print("new_lr",new_lr)
            self.a_lr = new_lr
            self.c_lr = new_lr
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),  self.a_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  self.c_lr)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.a_lr)

        if len(self.replay_buffer) < self.batch_size:
            return 0, 0

        # Sample a batch from memory
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.replay_buffer.get_batches()

        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.actor.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            #print("qf1_next_target.size()",qf1_next_target.size(),"next_state_log_pi.size()",next_state_log_pi.size())
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            #print("reward_batch.size(),mask_batch.size(),min_qf_next_target.size()",reward_batch.size(),mask_batch.size(),min_qf_next_target.size())
            next_q_value = reward_batch + (1- mask_batch) * self.gamma * (min_qf_next_target)
        
        qf1, qf2 = self.critic(state_batch, action_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
        qf1_loss = F.mse_loss(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf2_loss = F.mse_loss(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
        qf_loss = qf1_loss + qf2_loss

        self.critic_optimizer.zero_grad()
        qf_loss.backward()
        self.critic_optimizer.step()

        pi, log_pi, _ = self.actor.sample(state_batch)

        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)

        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone() # For TensorboardX logs
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs


        if self.total_it % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)

        self.a_loss += policy_loss.item()
        self.c_loss += qf1_loss.item() + qf2_loss.item()
        #self.alpha_loss += alpha_loss.item() + alpha_tlogs.item()
        
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
