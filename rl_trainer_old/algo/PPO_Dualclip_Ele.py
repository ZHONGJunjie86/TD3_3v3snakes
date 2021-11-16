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
        self.conv1 = nn.Conv2d(4,4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv2 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104
        self.conv3 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1) # 20104 -> 20104

        self.linear_1 = nn.Linear(800, 800)
        self.linear_2 = nn.Linear(800, 800)

        self.MU = nn.Linear(800,3)
        self.STD = nn.Linear(800,3)

        self.action_scale = 2
        self.action_bias = 2        
        self.epsilon = 1e-6

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
        
        mean = self.MU(i)
        std = self.STD(i).clamp(-20,2)
        std = std.exp()
        dist = Normal(mean, std)
        
        x_t = dist.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = dist.log_prob(x_t)

        # Enforcing Action Bound
        #log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        #log_prob = log_prob.sum(1, keepdim=True) 
        # 在此基础上，PPO计算新旧策略的熵使用的是经过tanh之前的动作，
        # 而SAC计算策略的熵使用了tanh之后的动作，因而在动作均值接近动作边界时，
        # 即便方差很大，它策略的熵不见得很大。
        # 因此SAC为了抵消tanh的影响，在计算策略熵的时候，添加了 tanh(a) 的导数项 [公式] 作为修正
        # 加上极小值 epsilon 是为了防止log计算溢出。
        # 然而，当动作接近边界值 -1 或 1时，这里的计算误差非常大，甚至会导致梯度方向出现错误。

        entropy = -torch.exp(log_prob) * log_prob
        #log_prob = log_prob.sum(1, keepdim=True)

        return action.clamp(0,3.99),log_prob,entropy#,state_value

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
        out_1 = torch.tanh(self.linear_CNN_1_1(i)).reshape(batch_size,3) #1

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
        out_2 = torch.tanh(self.linear_CNN_1_2(i)).reshape(batch_size,3)
       
        return out_1,out_2#,(h_state[0].data,h_state[1].data)

class PPO:
    def __init__(self, args=None):
        super().__init__(args)
        # could be 0.2 ~ 0.5, ratio.clamp(1 - clip, 1 + clip),
        self.ratio_clip = 0.3 if args is None else args['ratio_clip']
        # could be 0.01 ~ 0.05
        self.lambda_entropy = 0.05 if args is None else args['lambda_entropy']
        # could be 0.95 ~ 0.99, GAE (Generalized Advantage Estimation. ICLR.2016.)
        self.lambda_gae_adv = 0.97 if args is None else args['lambda_gae_adv']
        # if use Generalized Advantage Estimation
        self.if_use_gae = True if args is None else args['if_use_gae']
        # AgentPPO is an on policy DRL algorithm
        self.if_on_policy = True
        self.if_use_dn = False if args is None else args['if_use_dn']

        self.noise = None
        self.optimizer = None
        self.compute_reward = None  # attribution

    def init(self, net_dim, state_dim, action_dim, reward_dim=1, if_per=False):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_reward = self.compute_reward_gae if self.if_use_gae else self.compute_reward_adv

        self.cri = CriticAdv(state_dim, net_dim, self.if_use_dn).to(self.device)
        self.act = ActorPPO(net_dim, state_dim, action_dim, self.if_use_dn).to(self.device)

        self.optimizer = torch.optim.Adam([{'params': self.act.parameters(), 'lr': self.learning_rate},
                                           {'params': self.cri.parameters(), 'lr': self.learning_rate}])
        self.criterion = torch.nn.SmoothL1Loss()
        assert if_per is False  # on-policy don't need PER

    def select_action(state, policy):
        """select action for PPO
       :array state: state.shape==(state_dim, )
       :return array action: state.shape==(action_dim, )
       :return array noise: noise.shape==(action_dim, ), the noise
       """
        states = torch.as_tensor((state,), dtype=torch.float32).detach_()
        action = policy.get_action(states)[0]
        return action.detach().numpy()

    def update_net(self, buffer, _target_step, batch_size, repeat_times=4) -> (float, float):
        buffer.update_now_len_before_sample()
        buf_len = buffer.now_len  # assert buf_len >= _target_step

        '''Trajectory using reverse reward'''
        with torch.no_grad():
            buf_reward, buf_mask, buf_action, buf_state = buffer.sample_all()

            bs = 2 ** 10  # set a smaller 'bs: batch size' when out of GPU memory.
            buf_value = torch.cat([self.cri(buf_state[i:i + bs]) for i in range(0, buf_state.size(0), bs)], dim=0)
            buf_logprob = self.act.compute_logprob(buf_state, buf_action).unsqueeze(dim=1)
            buf_r_ret, buf_adv = self.compute_reward(buf_len, buf_reward, buf_mask, buf_value)
            del buf_reward, buf_mask

        '''PPO: Surrogate objective of Trust Region'''
        obj_critic = None
        for _ in range(int(repeat_times * buf_len / batch_size)):
            indices = torch.randint(buf_len, size=(batch_size,), requires_grad=False, device=self.device)

            state = buf_state[indices]
            action = buf_action[indices]
            r_ret = buf_r_ret[indices]
            logprob = buf_logprob[indices]
            adv = buf_adv[indices]

            new_logprob = self.act.compute_logprob(state, action).unsqueeze(dim=1)  # it is obj_actor
            ratio = (new_logprob - logprob).exp()
            obj_surrogate1 = adv * ratio
            obj_surrogate2 = adv * ratio.clamp(1 - self.ratio_clip, 1 + self.ratio_clip)
            obj_surrogate = -torch.min(obj_surrogate1, obj_surrogate2).mean()
            obj_entropy = (new_logprob.exp() * new_logprob).mean()  # policy entropy
            obj_actor = obj_surrogate + obj_entropy * self.lambda_entropy

            value = self.cri(state)  # critic network predicts the reward_sum (Q value) of state
            obj_critic = self.criterion(value, r_ret)

            obj_united = obj_actor + obj_critic / (r_ret.std() + 1e-5)
            self.optimizer.zero_grad()
            obj_united.backward()
            self.optimizer.step()

        self.update_record(obj_a=obj_surrogate.item(),
                           obj_c=obj_critic.item(),
                           obj_tot=obj_united.item(),
                           a_std=self.act.a_std_log.exp().mean().item(),
                           entropy=(-obj_entropy.item()))
        return self.train_record

    def compute_reward_adv(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return
        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # reward sum
        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
        buf_adv = buf_r_ret - (buf_mask * buf_value)
        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

    def compute_reward_gae(self, buf_len, buf_reward, buf_mask, buf_value) -> (torch.Tensor, torch.Tensor):
        """compute the excepted discounted episode return
        :int buf_len: the length of ReplayBuffer
        :torch.Tensor buf_reward: buf_reward.shape==(buf_len, 1)
        :torch.Tensor buf_mask:   buf_mask.shape  ==(buf_len, 1)
        :torch.Tensor buf_value:  buf_value.shape ==(buf_len, 1)
        :return torch.Tensor buf_r_sum:      buf_r_sum.shape     ==(buf_len, 1)
        :return torch.Tensor buf_advantage:  buf_advantage.shape ==(buf_len, 1)
        """
        buf_r_ret = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # old policy value
        buf_adv = torch.empty(buf_reward.shape, dtype=torch.float32, device=self.device)  # advantage value

        pre_r_ret = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                                device=self.device)  # reward sum of previous step
        pre_adv = torch.zeros(buf_reward.shape[1], dtype=torch.float32,
                              device=self.device)  # advantage value of previous step
        for i in range(buf_len - 1, -1, -1):
            buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]

            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv

        buf_adv = (buf_adv - buf_adv.mean(dim=0)) / (buf_adv.std(dim=0) + 1e-5)
        return buf_r_ret, buf_adv

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

        self.c_loss = 0
        self.a_loss = 0
        
        self.eps_clip = 0.1
        self.K_epochs = 5

    # Random process N using epsilon greedy
    def choose_action(self, obs, evaluation=False):
        self.memory.states.append(obs)
        obs = torch.Tensor([obs]).to(self.device)
        action,action_logprob,_ = self.actor(obs)
        self.memory.actions.append(action)
        self.memory.logprobs.append(action_logprob[0].cpu().detach().numpy())
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
        for reward, is_terminal,values_1,values_2 in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals),
                                       reversed(state_values_1),reversed(state_values_2)): #反转迭代
            #print("1 - is_terminal",(1-is_terminal))
            #if is_terminal==1:
            """buf_r_ret[i] = buf_reward[i] + buf_mask[i] * pre_r_ret
            pre_r_ret = buf_r_ret[i]
            buf_adv[i] = buf_reward[i] + buf_mask[i] * pre_adv - buf_value[i]
            pre_adv = buf_value[i] + buf_adv[i] * self.lambda_gae_adv"""

            discounted_reward = discounted_reward*(1-is_terminal)
            discounted_reward = reward +  discounted_reward #self.gamma *
            rewards.insert(0, discounted_reward) #插入列表

            values_1,values_2 = values_1.cpu().detach().numpy(),values_2.cpu().detach().numpy()
            values = (values_1+values_2)/2
            
            #print("values_1,values_2",values_1,values_2,"np.max(values_1,values_2)",np.maximum(values_1,values_2))
            #GAE = reward + vt+1 discounted_advatage - np.maximum(values_1,values_2)
            discounted_advatage = reward + (1-is_terminal)*(discounted_advatage - values)#np.maximum(values_1,values_2)
            GAE_advantage.insert(0, discounted_advatage) #插入列表
            discounted_advatage = self.gamma*discounted_advatage + values #np.minimum(values_1,values_2) #
        
        # Normalizing the rewards:
        rewards = torch.tensor(rewards).to(device)
        rewards_std = reward.std()
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        GAE_advantage = torch.tensor(GAE_advantage).to(device)
        advantages = (GAE_advantage - GAE_advantage.mean(dim=0)) / (GAE_advantage.std(dim=0) + 1e-5)


        # importance sampling -> 多次学习
        # Optimize policy for K epochs:

        for _ in range(self.K_epochs):
            # Evaluating old actions and values : 整序列训练...
            _,logprobs, dist_entropy = self.actor(old_states)

            state_values_1,state_values_2 = self.critic(old_states)
            
            # Finding the ratio e^(pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:     # Critic    (r+γV(s')-V(s)) 
            #advantages = rewards - torch.max(state_values_1,state_values_2)   #max???  minimize the advatage = underestimating
            surr1 = ratios*advantages.detach()
            #Jθ'(θ) = E min { (P(a|s,θ)/P(a|s,θ') Aθ'(s,a)) , 
            #                         clip(P(a|s,θ)/P(a|s,θ'),1-ε,1+ε)Aθ'(s,a) }              θ' demonstration
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip)*advantages.detach() 
            
            #Dual_Clip
            surr3 = torch.max(torch.min(surr1, surr2),3*advantages.detach())

            #
            actor_loss = -(surr3 + 20*self.a_lr*dist_entropy ).mean() #100*self.a_lr 0.01*

            
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

    
