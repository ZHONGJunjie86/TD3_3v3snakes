import os
import torch
import numpy as np
from torch.nn.utils import clip_grad_norm_
from pathlib import Path
import sys
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from replay_buffer import ReplayBuffer
from common import soft_update, hard_update#, device
from algo.network_SAC import GaussianPolicy,QNetwork
import torch.nn.functional as F

import torch.nn as nn

class SAC:
    def __init__(self, obs_dim, act_dim, num_agent, args,device):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.total_it = 0 
        
        self.device = device

        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = True #args.automatic_entropy_tuning

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
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        #
        hard_update(self.critic, self.critic_target)

        if self.policy_type == "Gaussian":
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -np.log(1/12)#-3 #np.log(3)#-np.log(1/3)#action_space.shape
                self.log_alpha = torch.tensor(-200.0, requires_grad=True, device=self.device) #-np.log(3) * np.e  -1.3542
                #torch.tensor((-np.log(action_dim) * np.e,), dtype=torch.float32,requires_grad=True, device=self.device)
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

        action, _, _ = self.actor.sample(obs) # action_run, log_prob, action_probs

        #print("action.cpu().detach().numpy()[0]",action.cpu().detach().numpy()[0])

        return action.cpu().detach().numpy()[0]

    def random_action(self):
        if self.output_activation == 'tanh':
            return np.random.uniform(low=-1, high=1, size=(self.num_agent, self.act_dim))
        return np.random.uniform(low=0, high=1, size=(self.num_agent, self.act_dim))

    def update(self,new_lr):
        self.total_it += 1

        if len(self.replay_buffer) < 25e3:#self.batch_size:#
            return 0, 0

        k = 1.0 + len(self.replay_buffer) / 9e5
        #batch_size_ = int(self.batch_size * k)
        train_steps = int(k * 20)

        for i in range(train_steps):

            # Sample a batch from memory
            state_batch, action_batch, reward_batch, next_state_batch, mask_batch = self.replay_buffer.get_batches()

            state_batch = torch.FloatTensor(state_batch).to(self.device)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
            action_batch = torch.FloatTensor(action_batch).to(self.device).reshape(self.batch_size,1,3)
            reward_batch = torch.FloatTensor(reward_batch).to(self.device).reshape(self.batch_size,1,3)
            mask_batch = torch.FloatTensor(mask_batch).to(self.device).reshape(self.batch_size,3)

            with torch.no_grad():  # qf_next 里面带了下一状态的熵
                # action_run, log_prob, action_probs
                _, next_state_log_pi, next_action_probs = self.actor.sample(next_state_batch)
                qf1_next_target, qf2_next_target = self.critic_target(next_state_batch)
                #print("qf1_next_target.size()",qf1_next_target.size(),"next_state_log_pi.size()",next_state_log_pi.size())
                #print("self.alpha * next_state_log_pi.size()",(self.alpha * next_state_log_pi).size())
                min_qf_next_target = next_action_probs*(torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi)#.sum(dim=-1)#,keepdim=False
                #print("min_qf_next_target.size()",min_qf_next_target.size())
                min_qf_next_target = min_qf_next_target.reshape(self.batch_size,3,4).sum(-1)
                #print("min_qf_next_target.size()",min_qf_next_target.size())
                next_q_value = reward_batch +  ((1- mask_batch) * self.gamma*min_qf_next_target).reshape(self.batch_size,1,3)
                #print("next_q_value.size()",next_q_value.size())
                #next_q_value = next_q_value .sum(dim=-1,keepdim=False)
            
            qf1, qf2 = self.critic(state_batch)  # Two Q-functions to mitigate positive bias in the policy improvement step
            qf1 = qf1.gather(-1,action_batch.long())
            qf2 = qf2.gather(-1,action_batch.long())
            #print("qf1.size(),next_q_value.size()",qf1.size(),next_q_value.size())
            qf1_loss = torch.nn.SmoothL1Loss()(qf1, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf2_loss = torch.nn.SmoothL1Loss()(qf2, next_q_value)  # JQ = 𝔼(st,at)~D[0.5(Q1(st,at) - r(st,at) - γ(𝔼st+1~p[V(st+1)]))^2]
            qf_loss = qf1_loss + qf2_loss

            self.critic_optimizer.zero_grad()
            qf_loss.backward()
            self.critic_optimizer.step()
        
            ####################################
            _, log_pi, action_probs = self.actor.sample(state_batch)

            qf1_pi, qf2_pi = self.critic(state_batch)  #因为s是旧策略采集的，所以要reparameterize
            min_qf_pi = torch.sum(torch.min(qf1_pi, qf2_pi)*action_probs,dim=1,keepdim = True)
            #print("min_qf_pi.size()",min_qf_pi.size())
            entropies = -torch.sum(action_probs*log_pi,dim=1,keepdim=True)
            policy_loss = -((self.alpha * entropies) + min_qf_pi).mean() # Jπ = 𝔼st∼D,εt∼N[α * logπ(f(εt;st)|st) − Q(st,f(εt;st))]
            #policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()   # min_qf_pi 里面带了本状态的熵
            self.actor_optimizer.zero_grad()
            policy_loss.backward()
            self.actor_optimizer.step()
            
            #################################
            if self.automatic_entropy_tuning:
                #alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
                alpha_loss = torch.mean(self.log_alpha * ( entropies.detach() - self.target_entropy).detach())#log_pi.detach()
                self.alpha_optim.zero_grad()
                alpha_loss.backward()
                self.alpha_optim.step()

                self.alpha = self.log_alpha.exp()
                alpha_tlogs = self.alpha.clone() # For TensorboardX logs
            else:
                alpha_loss = torch.tensor(0.).to(self.device)
                alpha_tlogs = torch.tensor(self.alpha) # For TensorboardX logs

            #if self.total_it % self.target_update_interval == 0:
            #    soft_update(self.critic_target, self.critic, self.tau)

            self.a_loss += policy_loss.item()
            self.c_loss += qf1_loss.item() + qf2_loss.item()

            #self.alpha_loss += alpha_loss.item() + alpha_tlogs.item()
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def get_actor(self):
        return self.actor
    
    def get_replay_buffer(self):
        return self.replay_buffer
    
    def get_log_alpha(self):
        return self.log_alpha

    def push(self,obs, logits, step_reward,next_obs, done):
        self.replay_buffer.push(obs, logits, step_reward,next_obs, done)
    
    def push_multi(self,tuple_batch):
        self.replay_buffer.push_multi(tuple_batch)
    
    def reset_loss(self):
        self.a_loss = 0
        self.c_loss = 0

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)

        model_actor_path = os.path.join(base_path, "actor_" + str(episode) + ".pth")
        model_critic_path = os.path.join(base_path, "critic_" + str(episode) + ".pth")
        print(f'Actor path: {model_actor_path}')
        print(f'Critic path: {model_critic_path}')

        if os.path.exists(model_critic_path) and os.path.exists(model_actor_path):
            actor_temp = torch.load(model_actor_path, map_location=self.device)
            critic_temp = torch.load(model_critic_path, map_location=self.device)
            self.actor.load_state_dict(actor_temp)
            self.critic.load_state_dict(critic_temp)
            print("Model loaded!")
        else:
            sys.exit(f'Model not founded!')

    def save_model(self, run_dir, episode):
        print("---------------save-------------------")
        base_path = os.path.join(run_dir, 'trained_model')
        print("new_lr: ",self.a_lr)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

        model_actor_path = os.path.join(base_path, "actor_4000"  + ".pth") #+ str(episode)
        torch.save(self.actor.state_dict(), model_actor_path)

        model_critic_path = os.path.join(base_path, "critic_4000" + ".pth") #+ str(episode) 
        torch.save(self.critic.state_dict(), model_critic_path)



class Worker(nn.Module):
    def __init__(self, obs_dim, act_dim, num_agent, args,device): #num_inputs, num_actions, hidden_dim, action_space=None
        super(Worker, self).__init__()
        self.device = device
        self.conv1 = nn.Conv2d(4,16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104
        self.conv3 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104
        self.conv4 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 20104

        self.linear = nn.Linear(1280, 12)

        # action rescaling
        self.action_scale = 2
        self.action_bias = 2

        self.sqrt_2pi_log = np.log(np.sqrt(2 * np.pi))

        self.batch_size = 0
        
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, tensor_cv):
        self.batch_size = tensor_cv.size()[0]
        i_1 = tensor_cv
        # CV
        x = F.relu(self.conv1(i_1))
        #i_2 = i_1 + x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x)).reshape(self.batch_size,1,1280)
        x = self.linear(x)
        
        return x 

    def sample(self, state):
        action_probs = torch.softmax( self.forward(state).reshape(self.batch_size,3,4) ,dim =-1)
        dis = self.Categorical(action_probs)
        action_run = dis.sample()#.view(-1,1)
        z = (action_probs == 0.0).float()*1e-8
        
        log_prob = torch.log(action_probs+z)
        

        return action_run, log_prob.reshape(self.batch_size,1,12), action_probs.reshape(self.batch_size,1,12) #x_t #mean

    def choose_action(self, obs):
            obs = torch.Tensor([obs]).to(self.device)

            action, _, _ = self.sample(obs)

            return action.cpu().detach().numpy()[0]