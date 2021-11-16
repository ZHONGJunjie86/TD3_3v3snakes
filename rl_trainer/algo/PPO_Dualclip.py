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
import torch.nn.functional as F

from torch.distributions import Normal #Multivariate
import torch.nn as nn

torch.set_default_tensor_type(torch.DoubleTensor)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    
    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

class Actor(nn.Module):
    def __init__(self):  #(n+2p-f)/s + 1 
        super(Actor, self).__init__()
        self.conv1 = nn.Conv2d(4,32, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3 = nn.Conv2d(32, 32, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.linear_1 = nn.Linear(2560, 12)

        self.action_scale = 2
        self.action_bias = 2        
        self.epsilon = 1e-6
        #
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, tensor_cv): #,batch_size
        # CV
        batch_size = tensor_cv.size()[0]
        x = F.relu(self.conv1(tensor_cv))
        #x = tensor_cv + x
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x=  x.reshape(batch_size,1,2560) #768

        #x = F.relu(self.linear_1(i))
        #i = i + x
        #x = F.relu(self.linear_2(i))
        #i = i + x
        prob = self.soft_max(self.linear_1(x).reshape(batch_size,3,4))
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
        #entropy = dis.entropy()

        return action,log_prob,entropy#,state_value

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(4,16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2_1 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3_1 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4_1 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 16632 -> 14464
        self.linear_CNN_1_1 = nn.Linear(1280, 3) #14464 = 3584

        ##########################################
        self.conv1_2 = nn.Conv2d(4,16, kernel_size=(6,3), stride=1, padding=1) # 20104 -> 17108
        self.conv2_2 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 17108 -> 141016
        self.conv3_2 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 141016-> 111032
        self.conv4_2 = nn.Conv2d(16, 16, kernel_size=(6,3), stride=1, padding=1) # 16632 -> 14464
        self.linear_CNN_1_2 = nn.Linear(1280, 3) #14464 = 3584


    def forward(self, tensor_cv):
        # CV
        batch_size = tensor_cv.size()[0]
        x = F.relu(self.conv1_1(tensor_cv))
        #x = tensor_cv + x
        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv4_1(x))
        #
        x=  x.reshape(batch_size,1,1280)#.repeat(1,3,1) #768
        out_1 = torch.tanh(self.linear_CNN_1_1(x)).reshape(batch_size,1,3)

        ###################################################################
        # CV
        x = F.relu(self.conv1_2(tensor_cv))
        #x = tensor_cv + x
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv4_2(x))
        #
        x=  x.reshape(batch_size,1,1280)#.repeat(1,3,1)
        out_2 = torch.tanh(self.linear_CNN_1_2(x)).reshape(batch_size,1,3)

       
        return out_1,out_2#,(h_state[0].data,h_state[1].data)

class PPO:
    def __init__(self, obs_dim, act_dim, num_agent, args):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_agent = num_agent
        self.device = device
        self.a_lr = args.a_lr
        self.c_lr = args.c_lr
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.model_episode = args.model_episode
        self.eps = args.epsilon
        self.decay_speed = args.epsilon_speed
        self.output_activation = args.output_activation

        # Initialise actor network and critic network with ξ and θ
        self.actor = Actor().to(self.device)
        self.critic = Critic().to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.a_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.c_lr)
        self.SmoothL1Loss = torch.nn.SmoothL1Loss()

        #
        self.memory = Memory()
        #
        self.c_loss = 0
        self.a_loss = 0
        
        self.eps_clip = 0.1
        self.K_epochs = 4

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):
        self.memory.states.append(obs)
        obs = torch.Tensor([obs]).to(self.device)
        action,action_logprob,_ = self.actor(obs)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob.cpu().detach().numpy()) #[0]
        return action.cpu().detach().numpy()

    def update(self,new_lr):
        if new_lr != self.a_lr:
            print("new_lr",new_lr)
            self.a_lr = new_lr
            self.c_lr = new_lr
            self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),  self.a_lr)
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  self.c_lr)
       
        
        # convert list to tensor. stack: Concatenates sequence of tensors along a new dimension. #stack指定去掉1维
        batch_size = len(self.memory.actions)
        old_states = torch.tensor(self.memory.states).reshape(batch_size,4,20,10).to(device).detach() #torch.squeeze(, 1)
        #old_actions = torch.tensor(self.memory.actions).to(device).detach()
        old_logprobs = torch.tensor(self.memory.logprobs).reshape(batch_size,1,3).to(device).detach()

        
        state_values_1,state_values_2 = self.critic(old_states)

        # Monte Carlo estimate of rewards:
        rewards = []
        GAE_advantage = []
        discounted_reward = 0
        discounted_advatage = 0
        #values_pre = 0
        for reward, is_terminal,values_1,values_2 in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals),
                                       reversed(state_values_1),reversed(state_values_2)): #反转迭代
            #print("1 - is_terminal",(1-is_terminal))
            #if is_terminal==1:
            """buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv"""

            discounted_reward = discounted_reward*(1-is_terminal)
            discounted_reward = reward +  self.gamma *discounted_reward #
            rewards.insert(0, discounted_reward) #插入列表

            values_1,values_2 = values_1.cpu().detach().numpy(),values_2.cpu().detach().numpy()
            #values = (values_1+values_2)/2
            
            #print("values_1,values_2",values_1,values_2,"np.max(values_1,values_2)",np.maximum(values_1,values_2))
            #GAE = reward + vt+1 discounted_advatage - np.maximum(values_1,values_2)
            discounted_advatage = reward + (1-is_terminal)*discounted_advatage - np.maximum(values_1,values_2)#values ##
            GAE_advantage.insert(0, discounted_advatage) #插入列表
            discounted_advatage = self.gamma*self.gamma*discounted_advatage + self.gamma*np.minimum(values_1,values_2)#values # #value
            
            #delta = reward + self.gamma*values_pre*(1-is_terminal) - values
            #discounted_advatage = delta +self.gamma*self.gamma* discounted_advatage *(1-is_terminal) ##np.maximum(values_1,values_2)
            #values_pre = values
            #GAE_advantage.insert(0, discounted_advatage ) #插入列表
            
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards_std = reward.std()
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        GAE_advantage = torch.tensor(GAE_advantage).to(device)
        #print(GAE_advantage.size())
        advantages = (GAE_advantage - GAE_advantage.mean(-1)) / (GAE_advantage.std(-1) + 1e-5) #dim=1


        # importance sampling -> 多次学习
        # Optimize policy for K epochs:

        for _ in range(self.K_epochs):
            # Evaluating old actions and values : 整序列训练...
            _,logprobs, dist_entropy = self.actor(old_states)

            state_values_1,state_values_2 = self.critic(old_states)
            
            # Finding the ratio e^(pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach()) #logprobs.sum(2) - old_logprobs.sum(2)

            #print("ratios.size()",ratios.size(),"advantages.size()",advantages.size())

            # Finding Surrogate Loss:     # Critic    (r+γV(s')-V(s)) 
            #advantages = rewards - torch.max(state_values_1,state_values_2)   #max???  minimize the advatage = underestimating
            surr1 = ratios*advantages.detach()
            #Jθ'(θ) = E min { (P(a|s,θ)/P(a|s,θ') Aθ'(s,a)) , 
            #                         clip(P(a|s,θ)/P(a|s,θ'),1-ε,1+ε)Aθ'(s,a) }              θ' demonstration
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages.detach() 
            
            #Dual_Clip
            surr3 = torch.max(torch.min(surr1, surr2),3*advantages.detach())

            #
            actor_loss = -(surr3 + 5*self.a_lr*dist_entropy).mean()#.sum(-1)   100*self.a_lr 0.01*  .mean()

            
            # take gradient step 
            self.actor_optimizer.zero_grad()
            self.a_loss += actor_loss.item()
            actor_loss.backward()  
            self.actor_optimizer.step()

            critic_loss = torch.nn.SmoothL1Loss()(state_values_1.squeeze(1), rewards) +  torch.nn.SmoothL1Loss()(state_values_2.squeeze(1), rewards)
            #critic_loss = critic_loss / (rewards_std + 1e-5)
            self.critic_optimizer.zero_grad()
            self.c_loss += critic_loss.item()
            critic_loss.backward()
            self.critic_optimizer.step()

        """critic_loss =  self.SmoothL1Loss(state_values, rewards) #0.5*
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()"""

        #self.memory.clear_memory()

        return self.c_loss, self.a_loss

    def get_loss(self):
        return self.c_loss, self.a_loss

    def load_model(self, run_dir, episode):
        print(f'\nBegin to load model: ')
        base_path = os.path.join(run_dir, 'trained_model')
        print("base_path",base_path)

        model_actor_path = os.path.join(base_path, "actor_"  + ".pth")
        model_critic_path = os.path.join(base_path, "critic_"  + ".pth")
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

        self.linear_1 = nn.Linear(800, 800)
        self.linear_2 = nn.Linear(800, 800)

        #self.MU = nn.Linear(800,4)
        #self.STD = nn.Linear(800,4)
        self.linear = nn.Linear(800, 12)



        self.action_scale = 2
        self.action_bias = 2        
        self.epsilon = 1e-6

        
        #
        self.soft_max = nn.Softmax(dim=-1)
        self.Categorical = torch.distributions.Categorical

    def forward(self, tensor_cv): #,batch_size
        # CV
        i_1 = tensor_cv
        batch_size = i_1.size()[0]
        x = F.relu(self.conv1(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3(i_3))
        i_4 = i_3 + x
        i = i_4.reshape(batch_size,1,800)

        #x = F.relu(self.linear_1(i))
        #i = i + x
        #x = F.relu(self.linear_2(i))
        #i = i + x
        log_prob = self.soft_max(self.linear(i).reshape(batch_size,3,4))
        action = self.Categorical(log_prob).sample()
        #samples_2d = torch.multinomial(log_prob.reshape(batch_size,12), num_samples=1, replacement=True)
        #print("samples_2d",samples_2d)
        #action = samples_2d.reshape(i.size(0))

        entropy = -torch.exp(log_prob) * log_prob

        return action,log_prob,entropy#,state_value

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv3_1 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.linear_1_1 = nn.Linear(800, 800)
        self.linear_2_1 = nn.Linear(800, 800)
        self.linear_CNN_1_1 = nn.Linear(800, 3) #14464 = 3584  896
        #
        self.conv1_2 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2_2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv3_2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.linear_1_2 = nn.Linear(800, 800)
        self.linear_2_2 = nn.Linear(800, 800)
        self.linear_CNN_1_2 = nn.Linear(800, 3) #14464 = 3584  896


    def forward(self, tensor_cv):
        i_1 = tensor_cv.detach()
        batch_size = i_1.size()[0]
        # CV
        x = F.relu(self.conv1_1(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2_1(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3_1(i_3))
        i_4 = i_3 + x
        i = i_4.reshape(batch_size,1,800)
        #
        #x = F.relu(self.linear_1_1(i))
        #i = i + x
        #x = F.relu(self.linear_2_1(i))
        #i = i + x
        out_1 = torch.tanh(self.linear_CNN_1_1(i)).reshape(batch_size,1,3) #1

        ###################################################################
        # CV
        i_1 = tensor_cv.detach()
        # CV
        x = F.relu(self.conv1_2(i_1))
        i_2 = i_1 + x
        x = F.relu(self.conv2_2(i_2))
        i_3 = i_2 + x
        x = F.relu(self.conv3_2(i_3))
        i_4 = i_3 + x
        i = i_4.reshape(batch_size,1,800)
        #
        #x = F.relu(self.linear_1_2(i))
        #i = i + x
        #x = F.relu(self.linear_2_2(i))
        #i = i + x
        out_2 = torch.tanh(self.linear_CNN_1_2(i)).reshape(batch_size,1,3)
       
        return out_1,out_2#,(h_state[0].data,h_state[1].data)

"""