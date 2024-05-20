""" Formulas to calculate """
import sys

import numpy as np

from .ENV_EVS import *
from .config import *


class SystemModel:
    def __init__(self, n_agents, S_channel, S_power, S_gain, S_size, S_cycle, S_resolu, S_ddl, S_res, S_com, S_epsilon,
                 S_decision):
        # Parameters
        self.S_channel = S_channel
        self.S_power = S_power
        self.S_gain = S_gain
        self.S_size = S_size
        self.S_cycle = S_cycle
        self.S_resolu = S_resolu
        self.S_ddl = S_ddl
        self.S_res = S_res
        self.S_com = S_com
        self.S_epsilon = S_epsilon
        self.S_decision = S_decision

        self.Phi_local = V_L * np.log(1 + self.S_resolu / THETA_L)
        self.Phi_off = V_E * np.log(1 + self.S_resolu / THETA_E)

        DataRate = np.zeros(n_agents)
        for n in range(n_agents):
            SNR = np.sum([0 if (n_ == n or int(self.S_channel[n_]) != int(self.S_channel[n])) else self.S_decision[n_] *
                                                                                                   self.S_power[n_] *
                                                                                                   self.S_gain[n_][int(self.S_channel[n_]) - 1] for n_ in range(n_agents)])

            DataRate[n] = W_BANDWIDTH * np.log(1 + self.S_power[n] * self.S_gain[n][int(self.S_channel[n]) - 1] /
                                               (NOISE_VARIANCE + SNR)) / np.log(2)

        self.Time_proc = self.S_resolu * self.S_cycle * self.S_size / CAPABILITY_E
        self.Time_local = self.S_resolu * self.S_cycle * self.S_size / self.S_res
        self.Time_off = self.S_resolu * self.S_size / DataRate

        self.Energy_local = K_ENERGY_LOCAL * self.S_size * self.S_cycle * self.S_resolu * (self.S_res ** 2)
        self.Energy_off = self.S_power * self.Time_off

        self.total_com = np.sum(self.S_com)
        self.T_mean = np.mean(self.Time)
        self.Energy_mine = OMEGA * self.S_com
        self.R_mine = KSI * self.S_com / self.total_com * np.exp(-LAMBDA * self.T_mean / S_ddl)

    @property
    def Phi(self):
        return (1 - self.S_decision) * self.Phi_local + self.S_decision * self.Phi_off

    @property
    def Phi_penalty(self):
        return np.maximum((self.S_epsilon - self.Phi) / self.S_epsilon, 0)

    @property
    def Time(self):
        return (1 - self.S_decision) * self.Time_local + self.S_decision * (self.Time_off + self.Time_proc)

    @property
    def Time_penalty(self):
        return np.maximum((self.Time - self.S_ddl) / self.S_ddl, 0)

    @property
    def Energy(self):
        return (1 - self.S_decision) * self.Energy_local + self.S_decision * self.Energy_off

    @property
    def Utility_vt(self):
        return LAMBDA_E * ((self.Energy_local - self.Energy) / self.Energy_local) + LAMBDA_PHI * (
                (self.Phi - self.Phi_local) / self.Phi_local)

    @property
    def Utility_mine(self):
        return self.R_mine - self.Energy_mine

    @property
    def Reward(self):
        return (MU_1 * self.Utility_vt + MU_2 * self.Utility_mine - BETA * self.Phi_penalty - ALPHA * self.Time_penalty) * 1000