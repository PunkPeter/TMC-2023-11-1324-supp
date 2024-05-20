import csv
import random

from ..multiagentenv import MultiAgentEnv
import numpy as np
from collections import namedtuple
from .config import *
from .system_model import SystemModel


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class EVSEnv(MultiAgentEnv):
    def __init__(self, kwargs):
        args = kwargs

        if isinstance(args, dict):
            args = convert(args)
        self.args = args

        self.n_agents = N_AGENT
        self.action_space = action_size
        self.observation_space = state_size

        # 初始化state
        self.S_decision = np.zeros(self.n_agents)
        self.S_channel = np.zeros(self.n_agents)
        self.S_power = np.zeros(self.n_agents)
        self.S_gain = np.zeros((self.n_agents, K_CHANNEL))
        for n in range(self.n_agents):
            self.S_gain[n] = np.random.exponential(1, size=K_CHANNEL)
        self.S_size = np.zeros(self.n_agents)
        self.S_cycle = np.zeros(self.n_agents)
        self.S_resolu = np.zeros(self.n_agents)
        self.S_ddl = np.zeros(self.n_agents)
        self.S_res = np.zeros(self.n_agents)
        self.S_com = np.zeros(self.n_agents)
        self.S_epsilon = np.zeros(self.n_agents)
        self.S_res_delta = np.zeros(self.n_agents)

        self.action_lower_bound = [0, 1, 0.01, MIN_RES, MIN_COM, MIN_POWER]
        self.action_higher_bound = [1, K_CHANNEL, 0.99, MAX_RES, MAX_COM, MAX_POWER]

        self.epoch = 0

    # 重置
    def reset(self):
        global CAPABILITY_E

        self.epoch = 0
        CAPABILITY_E = 50
        self.S_res_delta = np.zeros(self.n_agents)

        for n in range(self.n_agents):
            self.S_size[n] = np.random.normal(S_SIZE, 1)
            self.S_cycle[n] = np.random.normal(S_CYCLE, 0.1)
            self.S_ddl[n] = S_DDL
            self.S_epsilon[n] = S_EPSILON

            self.S_channel[n] = np.random.choice(range(1, K_CHANNEL + 1))

        return self.state

    def step(self, action):
        print(action)
        for n in range(self.n_agents):
            self.S_decision[n] = action[n][0]
            self.S_channel[n] = action[n][1]
            self.S_resolu[n] = action[n][2]
            self.S_res[n] = np.max((action[n][3] - self.S_res_delta[n], MIN_RES))
            self.S_com[n] = action[n][4]
            self.S_power[n] = action[n][5]

        # 求reward
        system_model = SystemModel(self.n_agents, self.S_channel, self.S_power, self.S_gain, self.S_size, self.S_cycle,
                                   self.S_resolu, self.S_ddl, self.S_res, self.S_com, self.S_epsilon, self.S_decision)

        self.epoch += 1
        done = False
        if self.epoch > 5:
            self.reset()
            done = True

        return self.state, system_model.Reward, done, None

    def get_avail_actions(self):
        """ return available actions for all agents """
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_actions.append(self.get_avail_agent_actions(agent_id))

        return np.expand_dims(np.array(avail_actions), axis=0)

    def get_avail_agent_actions(self, agent_id):
        """ return the available actions for agent_id """
        action = np.array([1] * 6)
        return action

    def get_num_of_agents(self):
        """ Return the number of agents """
        return self.n_agents

    def get_obs(self):
        """ Return the obs for each agent in the power system
           the default obs: voltage, active power of generators, bus state, load active power, load reactive power
           each agent can only observe the state within the zone where it belongs """
        return self.state

    def get_obs_size(self):
        """ Return the observation size
        """
        return self.observation_space

    def get_total_actions(self):
        """ Return the total number of actions an agent could ever take
        """
        return self.action_space

    @property
    def state(self):
        """ Return all observation
        """
        state_ = [[self.S_size[n], self.S_cycle[n], self.S_res[n], CAPABILITY_E] for n in range(self.n_agents)]
        state_ = np.array(state_)
        return state_
