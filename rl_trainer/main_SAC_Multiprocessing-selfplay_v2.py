import argparse
import datetime
from matplotlib.pyplot import get
import os
import wandb
from tensorboardX import SummaryWriter

from pathlib import Path
import sys

#from torch import manager_path
base_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(base_dir))
from algo.SAC_Multiprocessing import SAC
from algo.SAC_Multiprocessing import Worker
from common import *
from log_path import *
from env.chooseenv import make
from Curve_ import cross_loss_curve
import numpy as np
import multiprocessing
import torch.multiprocessing as mp
from multiprocessing.managers import BaseManager

class MyManager(BaseManager):
    pass

MyManager.register("SAC_Copy",SAC)

os.environ['OMP_NUM_THREADS'] = '1'
import random

Memory_size = 4
#os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class Memory:
    def __init__(self):
        self.m_obs = []
        self.m_obs_next = []
    
    def clear_memory(self):
        del self.m_obs[:]
        del self.m_obs_next[:]
        
def get_enemy_obs(map_agent): # 2 3 4  5 6 7
    map_agent[map_agent==2] = 0.5
    map_agent[map_agent==3] = 0.6
    map_agent[map_agent==4] = 0.7
    map_agent[map_agent==5] = 0.2
    map_agent[map_agent==6] = 0.3
    map_agent[map_agent==7] = 0.4
    map_agent[map_agent==1] = 0.1
    return map_agent
            
def train(rank,args,device,log_dir,run_dir,shared_lock,
            shared_model = None,experiment_share_1 = None,experiment_share_2 = None,experiment_share_3 = None):
    #global shared_lock

    if rank == 0:
        writer = SummaryWriter(str(log_dir))
        save_config(args, log_dir)
    print(f'device: {device}')
    
    env = make(args.game_name, conf=None)
    
    #print("==algo: ", args.algo)
    #print(f'model episode: {args.model_episode}')
    #print(f'save interval: {args.save_interval}')

    num_agents = env.n_player
    #print(f'Total agent number: {num_agents}')

    ctrl_agent_index = [0, 1, 2]
    #print(f'Agent control by the actor: {ctrl_agent_index}')
    ctrl_agent_num = len(ctrl_agent_index)

    width = env.board_width
    #print(f'Game board width: {width}')
    height = env.board_height
    #print(f'Game board height: {height}')

    act_dim = env.get_action_dim()
    #print(f'action dimension: {act_dim}')
    obs_dim = 26
    #print(f'observation dimension: {obs_dim}')
    if rank == 0:
        wandb.init(project="my-test-project", entity="zhongjunjie")
        wandb.config = {
        "learning_rate": 0.0003,
        "batch_size": args.batch_size
        }
    
    model = Worker(obs_dim, 12, 0, args,device).to(device)#SAC(obs_dim*Memory_size, act_dim, ctrl_agent_num, args,device)
    model_enemy = Worker(obs_dim, 12, 0, args,device).to(device)
    shared_lock.acquire()
    
    
    #base_path = os.path.join(run_dir, 'trained_model')
    #base_path =  "/home/j-zhong/work_place/TD3_SAC_PPO_multi_Python/rl_trainer/models/snakes_3v3/run"
    model_actor_path =  "/home/j-zhong/work_place/TD3_SAC_PPO_multi_Python/rl_trainer/models/snakes_3v3/run22/trained_model/actor_4000.pth"
    model_actor_path_new =  "/home/j-zhong/work_place/TD3_SAC_PPO_multi_Python/rl_trainer/models/snakes_3v3/run23/trained_model/actor_4000.pth"
    actor_temp = torch.load(model_actor_path, map_location=device)
    
    model.load_state_dict(actor_temp) # sync with shared model
    model_enemy.load_state_dict(actor_temp)
    
    shared_lock.release()

    save_block = []

    history_reward = [] 
    history_step_reward = []
    history_a_loss = []
    history_c_loss = []
    history_success = []
    history_enemy = {}

    total_step_reward = 0
    c_loss , a_loss = 0,0

    memory  = Memory ()
    memory_enemy  = Memory ()

    training_stage = 2000

    torch.manual_seed(args.seed)

    new_lr = 0.0001

    episode = 0
    episode_enemy_update = 0
    success = 0
    select_pre = False

    while episode < args.max_episodes:

        punishiment_lock = [6,6,6]
        """if episode > training_stage  : 
            try:
                new_lr = sample_lr[int(episode // training_stage)]
            except(IndexError):
                new_lr = 0.000001#* (0.9 ** ((episode-Decay) //training_stage)) 
        """
        new_lr = 0.0003
        # Receive initial observation state s1
        state = env.reset()

        # During training, since all agents are given the same obs, we take the state of 1st agent.
        # However, when evaluation in Jidi, each agent get its own state, like state[agent_index]: dict()
        # more details refer to https://github.com/jidiai/Competition_3v3snakes/blob/master/run_log.py#L68
        # state: list() ; state[0]: dict()
        state_to_training = state[0]

        # ======================= feature engineering =======================
        # since all snakes play independently, we choose first three snakes for training.
        # Then, the trained model can apply to other agents. ctrl_agent_index -> [0, 1, 2]
        # Noted, the index is different in obs. please refer to env description.
        obs = visual_ob(state[0])
        obs_enemy = obs.copy()
        obs = obs/10
        obs_enemy = get_enemy_obs(obs_enemy)

        #print(obs,obs_enemy)
        #Memory-beginning
        for _ in range(Memory_size): 
            memory.m_obs.append(obs)
            memory_enemy.m_obs.append(obs_enemy)
        obs = np.stack(memory.m_obs)
        obs_enemy = np.stack(memory_enemy.m_obs)
        
        episode += 1
        step = 0
        episode_reward = np.zeros(6)

        while True:

            # ================================== inference ========================================
            # For each agents i, select and execute action a:t,i = a:i,θ(s_t) + Nt
            logits = model.choose_action(obs)

            # ============================== add opponent actions =================================
            # we use rule-based greedy agent here. Or, you can switch to random agent.
            """if np.random.uniform()>=0.01:
                logits_enemy = model_enemy.choose_action(obs_enemy)
                #print("logits,logits_enemy",logits,logits_enemy)
                actions = np.array([logits , logits_enemy]).reshape(6)
            else:
                actions = logits_AC(state_to_training, logits, height, width)"""
            logits_enemy = model_enemy.choose_action(obs_enemy)
            actions = np.array([logits , logits_enemy]).reshape(6)
            #print("actions",actions)
            #actions = logits_random(act_dim, logits)

            # Receive reward [r_t,i]i=1~n and observe new state s_t+1
            next_state, reward, done, _, info = env.step(env.encode(actions))
            next_state_to_training = next_state[0]
            
            next_obs = visual_ob(next_state_to_training)
            next_obs_enemy = next_obs.copy()
            next_obs_enemy = get_enemy_obs(next_obs_enemy)
            next_obs = next_obs/10 #get_observations(next_state_to_training, ctrl_agent_index, obs_dim, height, width)/10
            
            #Memory 
            if len(memory.m_obs_next) !=0: 
                del memory.m_obs_next[:1]
                del memory_enemy.m_obs_next[:1]
                memory.m_obs_next.append(next_obs)
                memory_enemy.m_obs_next.append(next_obs_enemy)
            else: 
                memory.m_obs_next = memory.m_obs
                memory_enemy.m_obs_next = memory_enemy.m_obs
                memory.m_obs_next[Memory_size-1] = next_obs
                memory_enemy.m_obs_next[Memory_size-1] = next_obs_enemy
     
            next_obs = np.stack(memory.m_obs_next)
            next_obs_enemy = np.stack(memory_enemy.m_obs_next)
                
            # ================================== reward shaping ========================================
            reward = np.array(reward)
            episode_reward += reward
            """if done:  #结束
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]): #AI赢
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=1)
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):   #random赢
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=2)
                else:
                    step_reward = get_reward(info, ctrl_agent_index, reward, score=0) #平
            else:"""
            if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):   #AI长
                step_reward = get_reward(info, ctrl_agent_index, reward, punishiment_lock,score=3)
            elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):  #random长
                step_reward = get_reward(info, ctrl_agent_index, reward, punishiment_lock,score=4)
            else:                                                          #一样长
                step_reward = get_reward(info, ctrl_agent_index, reward, punishiment_lock,score=0)
            
            total_step_reward += sum(step_reward)

            done = np.array([done] * ctrl_agent_num)

            # ================================== collect data ========================================
            # Store transition in R
            logits[1] = logits[1] + 4
            logits[2] = logits[2] + 8
            #print(logits)
            save_block.append((obs, logits, step_reward,next_obs, done))#[obs,obs,obs][next_obs,next_obs,next_obs]

            obs = next_obs
            obs_enemy = next_obs_enemy
            step += 1
            
            ########## Training
            if args.episode_length <= step: 
            
                if rank == 0:
                    shared_lock.acquire()
                    experiment_share_1.extend(save_block)
                    shared_model.push_multi(experiment_share_1)
                    shared_model.push_multi(experiment_share_2)
                    shared_model.push_multi(experiment_share_3)
                    del experiment_share_1[:]
                    del experiment_share_2[:]
                    del experiment_share_3[:]
                    shared_lock.release()
                      
                    shared_model.update(new_lr)
                    c_loss , a_loss = shared_model.get_loss()
                    shared_model.reset_loss() 
                    model.load_state_dict(shared_model.get_actor().state_dict())
                    shared_model.save_model(run_dir, episode)
                    save_block = []
                else:
                    if episode % rank == 0:
                        shared_lock.acquire()
                        if rank <13:
                            experiment_share_1.extend(save_block)
                        elif 13<=rank and rank<26:
                            experiment_share_2.extend(save_block)
                        else:
                            experiment_share_3.extend(save_block)
                        
                        shared_lock.release()
                        save_block = []
                    if episode % 60 == 0 and episode != 0:
                        actor_temp = torch.load(model_actor_path_new , map_location=device)
                        model.load_state_dict(actor_temp) 

                

                if np.sum(episode_reward[0:3]) > np.sum(episode_reward[3:]): success = 1
                else: success = 0
                history_success.append(success)

                if rank == 0:
                    print(f'[Episode {episode:05d}] total_reward: {np.sum(episode_reward[0:3]):} rank: {rank:.2f}')
                    print(f'[Episode {episode:05d}] Enemy_reward: {np.sum(episode_reward[3:]):}')
                    print(f'\t\t\t\tsnake_1: {episode_reward[0]} '
                      f'snake_2: {episode_reward[1]} snake_3: {episode_reward[2]}')
                    print("log_alpha",shared_model.get_log_alpha())

                    reward_tag = 'reward'
                    loss_tag = 'loss'

                    print("len(shared_model.replay_buffer)",len(shared_model.get_replay_buffer()))
                    writer.add_scalars(reward_tag, global_step=episode,
                                    tag_scalar_dict={'snake_1': episode_reward[0], 'snake_2': episode_reward[1],
                                                        'snake_3': episode_reward[2], 'total': np.sum(episode_reward[0:3])})
                    
                    if c_loss and a_loss :
                        writer.add_scalars(loss_tag, global_step=episode,
                                        tag_scalar_dict={'actor': a_loss, 'critic': c_loss})

                    if c_loss and a_loss :
                        print(f'\t\t\t\ta_loss {a_loss:.3f} c_loss {c_loss:.3f}')
                    

                    history_reward.append(np.sum(episode_reward[0:3]))
                    
                    history_a_loss.append(a_loss/100)
                    history_c_loss.append(c_loss/10)
                    
                    
                    history_step_reward.append(total_step_reward/100)
                    cross_loss_curve(history_reward,history_a_loss,history_c_loss,history_step_reward)
                    wandb.log({"alpha":shared_model.get_log_alpha().item(),"a_loss":a_loss/10,"c_loss":c_loss/100,
                            "reward":np.sum(episode_reward[0:3]),"relative_reward":(np.sum(episode_reward[0:3])-np.sum(episode_reward[3:]))})

                """if episode % args.save_interval == 0 and rank == 0:
                    shared_lock.acquire()
                    shared_model.save_model(run_dir, episode)
                    shared_lock.release()"""
                
                
                if len(history_success)>=60 and sum(history_success[-60:])>=33:
                    print("-------------------------rank",rank,"----------------------------")
                    print("-------------------------Update Enemy!!!----------------------------")
                    print("Success Rate",(sum(history_success[-60:])/60.0)*100,"%")
                    gap = episode - episode_enemy_update
                    if len(history_enemy)<3 and (not select_pre):
                        history_model = Worker(obs_dim, 12, 0, args,device).to(device)
                        history_model.load_state_dict(model_enemy.state_dict())
                        history_enemy[history_model] = gap
                    else:
                        if gap > min(history_enemy.values()) and (not select_pre):  #只会有第一个
                            del history_enemy[min(history_enemy,key = history_enemy.get)]
                            history_model = Worker(obs_dim, 12, 0, args,device).to(device)
                            history_model.load_state_dict(model_enemy.state_dict())
                            history_enemy[history_model] = gap

                    if np.random.uniform()>=0.2:    
                        model_enemy.load_state_dict(model.state_dict())
                        select_pre = False
                    else:
                        # 选个以前牛逼的
                        print("选个以前牛逼的!!!!!")
                        niubi_enemy = random.sample(history_enemy.keys(), 1)[0] 
                        model_enemy.load_state_dict(niubi_enemy.state_dict())
                        select_pre = True
                    
                    episode_enemy_update = episode
                    history_success = []

                total_step_reward = 0

                env.reset()
                memory.clear_memory()
                memory_enemy.clear_memory()
                break


if __name__ == '__main__':# big 21--24   #smaller 22 -- 23 #bigger 25
    parser = argparse.ArgumentParser()
    parser.add_argument('--game_name', default="snakes_3v3", type=str)
    parser.add_argument('--algo', default="ddpg", type=str, help="bicnet/ddpg")
    parser.add_argument('--max_episodes', default=500000, type=int) #50000
    parser.add_argument('--episode_length', default=200, type=int)
    parser.add_argument('--output_activation', default="softmax", type=str, help="tanh/softmax")

    parser.add_argument('--buffer_size', default=int(5e6), type=int) #1e5
    parser.add_argument('--tau', default=0.4, type=float) #0.005
    parser.add_argument('--gamma', default=0.95, type=float) #0.95
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--a_lr', default=0.0003, type=float)#0.0001
    parser.add_argument('--c_lr', default=0.0003, type=float)
    parser.add_argument('--batch_size', default=32768 , type=int)#16384 8192 4096
    parser.add_argument('--epsilon', default=0.5, type=float)
    parser.add_argument('--epsilon_speed', default=0.993, type=float) #0.99998

    parser.add_argument("--save_interval", default=20, type=int) #1000
    parser.add_argument("--model_episode", default=0, type=int)
    parser.add_argument('--log_dir', default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    
    #SAC
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')

    parser.add_argument("--load_model", action='store_true')  # 加是true；不加为false
    parser.add_argument("--load_model_run", default=22, type=int)
    parser.add_argument("--load_model_run_episode", default=4000, type=int)
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α (default: False)')
    
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')

    #Multiprocessing
    parser.add_argument('--processes', default=40, type=int, help='number of processes to train with')

    args = parser.parse_args()

    if sys.version_info[0] > 2:
        mp.set_start_method('spawn') # this must not be in global scope
    elif sys.platform == 'linux' or sys.platform == 'linux2':
        raise "Must be using Python 3 with linux!" # or else you get a deadlock in conv2d

    
    manager = MyManager()
    manager.start()

    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    model_share = manager.SAC_Copy(1*Memory_size, 12, 3, args,device)
    experiment_share_1 = mp.Manager().list([])
    experiment_share_2 = mp.Manager().list([])
    experiment_share_3 = mp.Manager().list([])

    # 定义保存路径
    run_dir, log_dir = make_logpath(args.game_name, args.algo)
    #writer = SummaryWriter(str(log_dir))
    #save_config(args, log_dir)

    if True:#args.load_model:#
        load_dir = os.path.join(os.path.dirname(run_dir), "run" + str(args.load_model_run))
        model_share.load_model(load_dir, episode=args.load_model_run_episode)
    
    shared_lock = multiprocessing.Manager().Lock()

    processes = []
    for rank in range(args.processes):# #rank 编号
        if rank == 0:
            device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
            p = mp.Process(target=train, args=( rank, args,device,log_dir,run_dir,shared_lock,
                            model_share ,experiment_share_1 ,experiment_share_2 ,experiment_share_3 ))
        elif rank <13:
            device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
            p = mp.Process(target=train, args=( rank, args,device,log_dir,run_dir,shared_lock,
                            None ,experiment_share_1 ,None ,None)) #rank 编号
        elif 13<=rank and rank<26:
            device = torch.device("cuda:2") if torch.cuda.is_available() else torch.device("cpu")
            p = mp.Process(target=train, args=( rank, args,device,log_dir,run_dir,shared_lock,
                            None ,None ,experiment_share_2 ,None)) #rank 编号
        else:
            device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")
            p = mp.Process(target=train, args=( rank, args,device,log_dir,run_dir,shared_lock,
                            None ,None ,None ,experiment_share_3)) #rank 编号

        p.start() ; processes.append(p)
    for p in processes: p.join()
