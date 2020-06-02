import cityflow
import pandas as pd
import os
import math
# from sim_setting import sim_setting_control
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import logging
from gym import spaces
import numpy as np
import copy

class CityFlowEnv(object):
    def __init__(self, config):
        self.step_ount = 0
        # cityflow_config['rlTrafficLight'] = rl_control # use RL to control the light or not
        self.eng = cityflow.Engine(config['cityflow_config_file'], thread_num=config['thread_num'])

        self.config = config
        self.num_step = config['num_step']
        self.lane_phase_info = config['lane_phase_info']  # "intersection_1_1"

        assert len(self.lane_phase_info.keys()) == 1, "Check out intersection numbers!"
        self.intersection_id = list(self.lane_phase_info.keys())[0]

        #logging.info('self.lane_phase_info:%s' % self.lane_phase_info)
        #logging.info('self.intersection_id:%s' % self.intersection_id)

        self.start_lane = self.lane_phase_info[self.intersection_id]['start_lane']
        self.phase_list = self.lane_phase_info[self.intersection_id]["phase"]
        self.phase_startLane_mapping = self.lane_phase_info[self.intersection_id]["phase_startLane_mapping"]

        self.current_phase = self.phase_list[0]
        self.current_phase_time = 0
        self.yellow_time = 2

        self.phase_log = []
        self.last_vehicle_count = 0

        self.num_green_phases = 4
        self.lane_per_ts = 8
        self.num_phase = 8

        self.radix_factors = [10 for _ in range(self.lane_per_ts)]

        self.lane_length = 300
        self.vehicle_gap = 2.5
        self.vehicle_size = 5

        self.min_green_time = 5
        self.max_green_time = 50.0

        self.lane_waiting_time = {}
        self.episode = 1
        self.last_waiting_time = 0.0
        self.total_wait_time = 0.0
        self.metric = []

        self.num_phases = 4
        self.lane_count = 8
        self.delta_time = 5
        self.time = 0


        self.observation_space = spaces.Box(low=np.zeros(self.num_phases + 1 + self.lane_count),
                                            high=np.ones(self.num_phases + 1 + self.lane_count))
        self.discrete_observation_space = spaces.Tuple((
            spaces.Discrete(self.num_phases),  # Green Phase
            spaces.Discrete(self.max_green_time // self.delta_time),  # Elapsed time of phase
            *(spaces.Discrete(4) for _ in range(self.lane_count))  # Density for each incoming lanes
        ))
        self.action_space = spaces.Discrete(self.num_phases)
        #self.action_space = 8

    def reset(self):
        self.eng.reset()
        self.last_vehicle_count = 0

    def get_elapsed_time(self):
        return self.current_phase_time/self.max_green_time

    def get_lanes_density(self):
        lane_density = []
        for lane in self.start_lane:
            #logging.info('lane:{} vehicle count: {}'.format(lane, self.eng.get_lane_vehicle_count()[lane]))
            lane_density.append(self.eng.get_lane_vehicle_count()[lane]/(300/7.5))

        #logging.info('lane_density: %s' % lane_density)
        return lane_density

    def get_queue_length(self):
        queue_length = []
        lane_vehicle_waiting = self.eng.get_lane_waiting_vehicle_count()
        for lane in self.start_lane:
            #logging.info('lane:{} waiting vehicle count: {}'.format(lane, lane_vehicle_waiting[lane]))
            queue_length.append(lane_vehicle_waiting[lane] / (300/7.5))
        #logging.info('queue length: %s' % queue_length)
        return queue_length

    def step(self, next_phase):
        if self.current_phase == next_phase:
            self.current_phase_time += 1
        else:
            self.current_phase = next_phase
            self.current_phase_time = 1

        #logging.info('self.current_phase_time:%s' % self.current_phase_time)
        #logging.info('self.intersection_id:{} self.current_phase:{}'.format(self.intersection_id, self.current_phase))

        self.eng.set_tl_phase(self.intersection_id, self.current_phase) # set phase of traffic light
        for i in range(self.min_green_time-1):
            print ('time step>>', self.time)
            self.eng.next_step()
            self.current_phase_time += 1
            self.time += 1

        self.phase_log.append(self.current_phase)

        return self.get_state(), self.get_reward_1() # return next_state and reward

    def get_state(self):
        state = {}
        print('self.start_lane:', self.start_lane)
        print('self.eng.get_lane_vehicle_count():', self.eng.get_lane_vehicle_count())
        state['lane_vehicle_count'] = self.eng.get_lane_vehicle_count()  # {lane_id: lane_count, ...}
        state['start_lane_vehicle_count'] = {lane: self.eng.get_lane_vehicle_count()[lane] for lane in self.start_lane}
        state[
            'lane_waiting_vehicle_count'] = self.eng.get_lane_waiting_vehicle_count()  # {lane_id: lane_waiting_count, ...}
        state['lane_vehicles'] = self.eng.get_lane_vehicles()  # {lane_id: [vehicle1_id, vehicle2_id, ...], ...}
        state['vehicle_speed'] = self.eng.get_vehicle_speed()  # {vehicle_id: vehicle_speed, ...}
        state['vehicle_distance'] = self.eng.get_vehicle_distance()  # {vehicle_id: distance, ...}
        state['current_time'] = self.eng.get_current_time()
        state['current_phase'] = self.current_phase
        state['current_phase_time'] = self.current_phase_time
        #print('<<state>>', state)
        return state

    def get_reward(self):
         # a sample reward function which calculates the total of waiting vehicles
         lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
         reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
         return reward

    def get_reward_2(self):
        # reward function
        # reward = prev_wait_time-current_wait_time
        # wait_time = total time all the vehicles spent in the traffic signal
        #logging.info('calculate the reward !')
        veh_count = 0
        for lane in self.start_lane:
            #logging.info('lane>>:%s' % lane)
            ##logging.info('vehicle count on the lane>>:%s' % self.eng.get_lane_vehicle_count()[lane])
            #logging.info('lane: {}, vehicle waiting count on the lane>> {}'.format(lane, self.eng.get_lane_waiting_vehicle_count()[lane]))
            veh_count = veh_count + self.eng.get_lane_vehicle_count()[lane]


        ##logging.info('self.last_wait_vehicle_count:{}, waiting_veh_count:{}'.format(self.last_vehicle_count,
        #                                                                      veh_count))
        r = self.last_vehicle_count - veh_count
        ##logging.info('reward based on veh count:%s' % r)

        lane_vehicle_count = self.eng.get_lane_vehicle_count()
        vehicle_velocity = self.eng.get_vehicle_speed()
        #logging.info('vehicle_velocity:%s' % vehicle_velocity)
        #logging.info('lane_vehicle_count:%s' % lane_vehicle_count)
        reward = sum(list(vehicle_velocity.values())) / (sum(list(lane_vehicle_count.values())) + 2)
        #logging.info('Reward: %s' % reward)
        #return reward
        self.get_reward_1()
        return r

    def get_reward_1(self):
        #logging.info('start calculating reward!')
        lane_waiting_time_list = {}
        road_waiting_time_list = []
        #logging.info('iterarte over lane_waiting_time:%s' % self.lane_waiting_time)

        """for lane in self.start_lane:
            #logging.info('vehicle in {} -> {}'.format(lane, self.eng.get_lane_vehicles()[lane]))
            for vehicle in self.lane_waiting_time[lane].iteritems():
                #logging.info('waiting vehicle {}'.format(vehicle))
                #logging.info('lane {} vehiciles {}'.format(lane, vehicle))
                if vehicle not in self.eng.get_lane_vehicles()[lane] and vehicle in self.lane_waiting_time[lane]:
                    del self.lane_waiting_time[lane][vehicle]"""

        for lane in self.start_lane:
            wait_time = 0
            logging.info('lane>>:%s' % lane)
            lane_vehicle = self.eng.get_lane_vehicles()[lane]
            logging.info('lane_vehicle:%s' % lane_vehicle)
            if lane not in self.lane_waiting_time.keys():
                self.lane_waiting_time[lane] = {}

            #logging.info('self.lane_waiting_time1:%s' % self.lane_waiting_time)
            #logging.info('self.lane_waiting_time2:%s' % self.lane_waiting_time[lane].keys())

            # remove those vehicle from  lane_waiting_time which are not in lane any more
            lane_waiting_time = copy.deepcopy(self.lane_waiting_time)
            for veh in lane_waiting_time[lane].keys():
                #logging.info('veh %s' % veh)
                #logging.info('lane_vehicle %s' % lane_vehicle)
                v_s = float(self.eng.get_vehicle_info(veh).get('speed', 0.0))
                #logging.info('vehicle speed:%s' % v_s)
                if veh not in lane_vehicle or v_s != 0.0:
                    #logging.info('veh {} not in lane_vehicle {} or '.format(veh, lane_vehicle))
                    del self.lane_waiting_time[lane][veh]

            #logging.info('self.lane_waiting_time now :%s' % self.lane_waiting_time)
            for vehicle in lane_vehicle:
                logging.info('vehicle:%s' % vehicle)
                vehicle_info = self.eng.get_vehicle_info(vehicle)
                logging.info('vehicle_info:%s' % vehicle_info)
                current_time = self.eng.get_current_time()
                #logging.info('current_time:%s' % current_time)
                if vehicle not in self.lane_waiting_time[lane].keys() and float(vehicle_info.get('speed')) == 0:
                    self.lane_waiting_time[lane][vehicle] = current_time
                elif vehicle in self.lane_waiting_time[lane].keys() and float(vehicle_info.get('speed')) == 0:
                    #self.lane_waiting_time[lane][vehicle] = current_time - self.lane_waiting_time[lane][vehicle]
                    wait_time  = wait_time + (current_time - self.lane_waiting_time[lane][vehicle])
                #if vehicle in self.lane_waiting_time[lane].keys() and float(vehicle_info.get('speed')) != 0
                #    del self.lane_waiting_time[lane][vehicle]

            #logging.info('self.lane_waiting_time3:%s' % self.lane_waiting_time)
            #logging.info('wait_time:%s' % wait_time)
            lane_waiting_time_list[lane] = wait_time
            #logging.info('EPISODE-LANE-WAIT::{}-{}-{}'.format(self.episode*4,lane, wait_time))



        self.episode += 1
        #logging.info('lane_waiting_time_list:%s' % lane_waiting_time_list)
        logging.info('lane_waiting_time_list:%s' % lane_waiting_time_list)
        logging.info('final self.lane_waiting_time:%s' % self.lane_waiting_time)
        self.total_wait_time = sum(lane_waiting_time_list.values())
        logging.info('self.total_wait_time:%s' % self.total_wait_time)
        reward = self.last_waiting_time - self.total_wait_time
        #logging.info('REWARD:%s' % reward)
        self.last_waiting_time = self.total_wait_time
        return reward

    def get_rl_state(self):
        lane_density = self.get_lanes_density()
        queue_length = self.get_queue_length()
        elapsed_time = self.get_elapsed_time()
        phase = self.get_phase_list()
        rl_state = self.encode(phase + [elapsed_time] + lane_density + queue_length)
        #rl_state = self.encode(phase + [elapsed_time] + queue_length)
        print('rl state', rl_state)
        return rl_state

    def get_score(self):
        lane_waiting_vehicle_count = self.eng.get_lane_waiting_vehicle_count()
        reward = -1 * sum(list(lane_waiting_vehicle_count.values()))
        metric = 1/((1 + math.exp(-1 * reward)) * self.config["num_step"])
        return metric

    def log(self):
        if not os.path.exists(self.config['replay_data_path']):
            os.makedirs(self.config["replay_data_path"])
        
        # self.eng.#logging.info_log(self.config['replay_data_path'] + "/replay_roadnet.json",
        #                    self.config['replay_data_path'] + "/replay_flow.json")

        df = pd.DataFrame({self.intersection_id: self.phase_log[:self.num_step]})
        df.to_csv(os.path.join(self.config['replay_data_path'], 'signal_plan.txt'), index=None)

    def get_phase_list(self):
        #logging.info('get phase list:%s' % self.current_phase)
        phase_list = [1 if self.current_phase // 2 == i else 0 for i in
                    range(self.num_green_phases)]  # one-hot encoding
        #phase_list = [1 if self.current_phase-1 == i else 0 for i in
        #              range(self.num_phase)]  # one-hot encoding

        #logging.info('phase list:%s' % phase_list)
        return phase_list

    """def encode(self, state):
        print('state:', state)
        print('self.num_green_phases:', self.num_green_phases)
        phase = state[:self.num_green_phases].index(1)
        elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        print('state[self.num_green_phases]:',state[self.num_green_phases])
        density_queue = [self._discretize_density(d) for d in state[self.num_green_phases + 1:]]
        return self.radix_encode([phase, elapsed] + density_queue)"""

    def encode(self, state):
        print('state:', state)
        print('self.num_phase:', self.num_phase)
        phase = state[:self.num_green_phases].index(1)
        print('phase>>', phase)
        elapsed = self._discretize_elapsed_time(state[self.num_green_phases])
        print('state[self.num_phase]:',state[self.num_green_phases+1:])
        density_queue = [self._discretize_density(d) for d in state[self.num_green_phases + 1:]]
        print ('coded density_queue:', density_queue)
        return self.radix_encode([phase, elapsed] + density_queue)


    def _discretize_density(self, density):
        if density < 0.1:
            return 1
        elif density < 0.2:
            return 2
        elif density < 0.3:
            return 3
        elif density < 0.4:
            return 4
        elif density < 0.5:
            return 5
        elif density < 0.6:
            return 6
        elif density < 0.7:
            return 7
        elif density < 0.8:
            return 8
        elif density < 0.9:
            return 9
        else:
            return 1.0

    def _discretize_elapsed_time(self, elapsed):
        elapsed *= self.max_green_time
        delta_time = 5
        for i in range(int(self.max_green_time)//delta_time):
            if elapsed <= delta_time + i*delta_time:
                return i
        return self.max_green_time//delta_time -1

    def radix_encode(self, values):
        print('values:', values)
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        print('res:', res)
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res

    def _compute_step_info(self):
        logging.info(self.eng.get_lane_waiting_vehicle_count())
        logging.info(self.lane_waiting_time)
        m = {
            'step_time': self.episode,
            'total_stopped': sum([s_h for s_h in self.eng.get_lane_waiting_vehicle_count().values()]),
            'total_wait_time': self.total_wait_time
        }
        self.episode += 1
        self.metric.append(m)


    def save_csv(self):
        import time
        df = pd.DataFrame(self.metric)
        out_filename = "output"
        df.to_csv(out_filename + '.csv' + str(time.time()), index=False)
        df.plot(kind='line', x='step_time', y='total_stopped')
        plt.show()
        plt.savefig(out_filename + '.jpg')

    """
    def encode(self, state):
        #logging.info('state_before_encode:%s' % state)
        #logging.info('self.num_green_phases:%s' % self.num_green_phases)
        #phase = state[:self.num_green_phases].index(1)
        phase = state[-1]
        #logging.info('phase:%s' % phase)
        #logging.info('state[self.num_green_phases]:%s' % state[self.num_green_phases])
        #logging.info('state[self.num_green_phases + 1:]:%s' % state[self.num_green_phases + 1:])
        density_queue = [self._discretize_density(d) for d in state[0:len(state)-1]]
        #logging.info('density_queue:%s' %  density_queue)
        e = self.radix_encode(density_queue + [phase])
        #logging.info('state_after_encode:%s' %  e)
        return e

    def _discretize_density(self, density):
        if density < 0.1:
            return 0
        elif density < 0.2:
            return 1
        elif density < 0.3:
            return 2
        elif density < 0.4:
            return 3
        elif density < 0.5:
            return 4
        elif density < 0.6:
            return 5
        elif density < 0.7:
            return 6
        elif density < 0.8:
            return 7
        elif density < 0.9:
            return 8
        else:
            return 9


    def radix_encode(self, values):
        res = 0
        for i in range(len(self.radix_factors)):
            res = res * self.radix_factors[i] + values[i]
        return int(res)

    def radix_decode(self, value):
        res = [0 for _ in range(len(self.radix_factors))]
        for i in reversed(range(len(self.radix_factors))):
            res[i] = value % self.radix_factors[i]
            value = value // self.radix_factors[i]
        return res"""

    def norm(self, value):
        if value < 0.1:
            return 0
        elif value < 0.2:
            return 1
        elif value < 0.3:
            return 2
        elif value < 0.4:
            return 3
        elif value < 0.5:
            return 4
        elif value < 0.6:
            return 5
        elif value < 0.7:
            return 6
        elif value < 0.8:
            return 7
        elif value < 0.9:
            return 8
        else:
            return 9

    def encode_1(self, lanes_count):
        b = 0
        a = 10
        lane_length = len(lanes_count)
        for i in range(lane_length-1):
            j = self.norm(lanes_count[i]/60.0)
            b = b + j*a ** (lane_length -2 - i)

        b = b*10+lanes_count[-1]
        return b