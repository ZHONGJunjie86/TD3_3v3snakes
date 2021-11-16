import numpy as np
import torch
import random
from agent.rl.submission_image import agent, visual_ob
from env.chooseenv import make
from tabulate import tabulate
import argparse
from torch.distributions import Categorical
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

def get_actions(state, algo, indexs,memory):

    # random agent
    actions = np.random.randint(4, size=3)

    # rl agent
    if algo == 'rl':
        obs = visual_ob(state)/10
        #get_observations(state, indexs, obs_dim=26, height=10, width=20)/10
        #Memory
        if len(memory) !=0: 
            del memory[:1]
            memory.append(obs)
        else: 
            for _ in range(4): 
                memory.append(obs)
        obs = np.stack(memory)
        logits = agent.choose_action(obs)
        actions = np.array([out for out in logits])

    return actions


def get_join_actions(obs, algo_list,memory):
    obs_2_evaluation = obs[0]
    
    indexs = [0,1,2,3,4,5]

    first_action = get_actions(obs_2_evaluation, algo_list[0], indexs[:3],memory)
    
    second_action = get_actions(obs_2_evaluation, algo_list[1], indexs[3:],memory)
    actions = np.zeros(6)
    actions[:3] = first_action[:]
    actions[3:] = second_action[:]
    return actions


def run_game(env, algo_list, episode, verbose=False):

    total_reward = np.zeros(6)
    num_win = np.zeros(3)
    
    memory = []

    for i in range(1, episode + 1):
        episode_reward = np.zeros(6)

        state = env.reset()

        step = 0

        while True:
            joint_action = get_join_actions(state, algo_list,memory)

            next_state, reward, done, _, info = env.step(env.encode(joint_action))
            reward = np.array(reward)
            episode_reward += reward

            if done:
                if np.sum(episode_reward[:3]) > np.sum(episode_reward[3:]):
                    num_win[0] += 1
                elif np.sum(episode_reward[:3]) < np.sum(episode_reward[3:]):
                    num_win[1] += 1
                else:
                    num_win[2] += 1

                if not verbose:
                    print('.', end='')
                    if i % 100 == 0 or i == episode:
                        print()
                break

            state = next_state
            step += 1
            memory = []

        total_reward += episode_reward

    # calculate results
    total_reward /= episode
    print("total_reward: ", total_reward)
    print(f'\nResult base on {episode} ', end='')
    print('episode:') if episode == 1 else print('episodes:')

    header = ['Name', algo_list[0], algo_list[1]]
    data = [['score', np.round(np.sum(total_reward[:3]), 2), np.round(np.sum(total_reward[3:]), 2)],
            ['win', num_win[0], num_win[1]]]
    print(tabulate(data, headers=header, tablefmt='pretty'))


if __name__ == "__main__":
    env_type = 'snakes_3v3'

    game = make(env_type, conf=None)
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)
    parser = argparse.ArgumentParser()
    parser.add_argument("--my_ai", default="rl", help="rl/random")
    parser.add_argument("--opponent", default="random", help="rl/random")
    parser.add_argument("--episode", default=100)
    args = parser.parse_args()

    agent_list = [args.my_ai, args.opponent]
    run_game(game, algo_list=agent_list, episode=args.episode, verbose=False)
