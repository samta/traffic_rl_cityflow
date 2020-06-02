import argparse
import json
import logging
import numpy as np
from datetime import datetime
from cityflow_env import CityFlowEnv
from utility import parse_roadnet
from tqdm import tqdm
from agents.ql_agent import QLAgent
from exploration.epsilon_greedy import EpsilonGreedy


logging.basicConfig(filename='1rl.log', format='%(asctime)s,%(msecs)d [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

def main():
    date = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/global_config.json')
    parser.add_argument('--num_step', type=int, default=3000,
                        help='number of timesteps for one episode, and for inference')
    parser.add_argument('--algo', type=str, default='DQN',
                        choices=['DQN', 'DDQN', 'DuelDQN'], help='choose an algorithm')
    parser.add_argument('--inference', action="store_true", help='inference or training')
    parser.add_argument('--ckpt', type=str, help='inference or training')
    parser.add_argument('--epoch', type=int, default=30, help='number of training epochs')
    parser.add_argument('--save_freq', type=int, default=100, help='model saving frequency')

    args = parser.parse_args()


    # preparing config
    # # for environment
    config = json.load(open(args.config))
    config["num_step"] = args.num_step
    # config["replay_data_path"] = "replay"
    cityflow_config = json.load(open(config['cityflow_config_file']))
    roadnetFile = cityflow_config['dir'] + cityflow_config['roadnetFile']
    config["lane_phase_info"] = parse_roadnet(roadnetFile)

    # # for agent
    intersection_id = list(config['lane_phase_info'].keys())[0]
    phase_list = config['lane_phase_info'][intersection_id]['phase']
    logging.info(phase_list)
    state_size = config["state_size"] = len(config['lane_phase_info'][intersection_id]['start_lane']) + 1
    #state_size = config["state_size"] = 25
    # the single dimension appended to the tail is for the current phase.
    # [vehicle_count for each start lane] + [current_phase]
    logging.info('state size:%s' % state_size)
    config["action_size"] = len(phase_list)
    phase_list = [1,2,3,4,5,6,7,8]
    # build cityflow environment
    env = CityFlowEnv(config)
    EPISODES = 1
    num_step = config['num_step']
    state_size = config['state_size']
    total_step = 0
    #num_step = 10
    with tqdm(total=EPISODES*args.num_step) as pbar:
        for i in range(1, EPISODES+1):
            logging.info('EPISODE >>:%s' % i)
            episode_length = 1
            env.reset()
            t=0
            state = env.get_state()
            state = np.array(list(state['start_lane_vehicle_count'].values()) + [
                state['current_phase']])  # a sample state definition
            # print ('state1:', state)
            state = np.reshape(state, [1, state_size])
            print('state2:', state)
            agent = QLAgent(starting_state=env.get_rl_state(),
                                 state_space=1,
                                 action_space=env.action_space,
                                 alpha=0.1,
                                 gamma=0.99,
                                 exploration_strategy=EpsilonGreedy(initial_epsilon=0.05,
                                                                    min_epsilon=0.005,
                                                                    decay=1.0))

            last_action = phase_list[agent.act(state)]
            print('last action:', last_action)

            print('episode_length:{}, num_step:{}'.format(episode_length, num_step))
            while episode_length < num_step:
                #logging.info('current state:%s' % state)
                logging.info('EPISODE LENGTH >>%s' % episode_length)
                action = agent.act(state)  # index of action
                logging.info('new action:%s' % action)
                action_phase = phase_list[action]  # actual action
                logging.info('action phase:>>%s' % action_phase)
                next_state, reward = env.step(action_phase)  # one step
                logging.info('STATE>>:%s' % next_state)
                logging.info('ACTION PHASE:{}'.format(action_phase))
                logging.info('ELAPSED TIME ON PHASE {} is {}'.format(env.current_phase, env.current_phase_time))
                logging.info('NORM ELAPSED TIME ON PHASE {} is {}'.format(env.current_phase, env.get_elapsed_time()))
                #for n_s in next_state.iteritems():
                #    logging.info(n_s)
                logging.info('REWARD:%s' % reward)

                # last_action_phase = action_phase
                episode_length += 1
                total_step += 1

                pbar.update(1)
                # store to replay buffer
                # prepare state
                agent.learn(new_state=env.get_rl_state(), reward=reward)

                env._compute_step_info()

                state = next_state
                logging.info("episode:{}/{}, total_step:{}, action:{}, reward:{}"
                             .format(i, EPISODES, total_step, action, reward))
                pbar.set_description("total_step:{total_step}, episode:{i}, episode_step:{episode_length}, "
                                     "reward:{reward}")

    env.save_csv()
    # save simulation replay
    # env.log()
    # automatically


if __name__ == '__main__':
    main()
