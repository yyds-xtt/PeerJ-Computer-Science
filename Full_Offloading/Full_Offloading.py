'''
--- Full Offloading (offloaing ratio = 1) ----
-------- Autor: Xi Hu and Yang Huang ---------
--- Northeastern University at Qinhuangdao ---
'''

import math
import numpy as np
from  tqdm import  tqdm
from random import choice

class MEC_env(object):
    height = 100
    ground_length = ground_width = 1000
    loc_MEC = [500, 500]
    bandwidth_nums = 600
    B = bandwidth_nums * 10 ** 6
    p_noisy_los = 10 ** (-13)
    p_noisy_nlos = 10 ** (-11)
    f_ve = 1.2e10
    f_MEC = 5e10
    s = 1000
    p_uplink = 0.1
    alpha0 = 1e-5
    slot_num = 180
    MEC_Computing = 10 * 1024 * 1048576 * 8
    v_ve = 10
    M_i = choice([(1 * 1048576 * 8, 2 * 1048576 * 8), (20 * 1048576 * 8, 30 * 1048576 * 8), (200 * 1048576 * 8, 250 * 1048576 * 8)])
    M = 10
    block_flag_list = np.random.randint(0, 2, M)
    loc_ve_list = np.random.randint(0, 1001, size=[M, 2])
    task_list = np.random.randint(*M_i, M)

    def __init__(self):
        self.start_state = np.append(self.MEC_Computing, np.ravel(self.loc_ve_list))
        self.start_state = np.append(self.start_state, self.task_list)
        self.start_state = np.append(self.start_state, self.block_flag_list)
        self.state = self.start_state

    def reset(self):
        self.reset_env()
        self.state = np.append(self.MEC_Computing, np.ravel(self.loc_ve_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self._get_obs()

    def reset_env(self):
        self.MEC_Computing = 10 * 1024 * 1048576 * 8
        self.loc_ve_list = np.random.randint(0, 1001, size=[self.M, 2])
        self.reset_step()

    def reset_step(self):
        M_i = choice([(1 * 1048576 * 8, 2 * 1048576 * 8), (20 * 1048576 * 8, 30 * 1048576 * 8), (200 * 1048576 * 8, 250 * 1048576 * 8)])
        self.task_list = np.random.randint(*M_i, self.M)
        self.block_flag_list = np.random.randint(0, 2, self.M)

    def _get_obs(self):
        self.state = np.append(self.MEC_Computing, np.ravel(self.loc_ve_list))
        self.state = np.append(self.state, self.task_list)
        self.state = np.append(self.state, self.block_flag_list)
        return self.state

    def time_delay(self, loc_ve, offloading_ratio, task_size, block_flag):
        loc_MEC = [500, 500]
        dx = loc_MEC[0] - loc_ve[0]
        dy = loc_MEC[1] - loc_ve[1]
        dh = self.height
        dist_mec_vehicle = dx * dx + dy * dy + dh * dh
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_mec_vehicle = abs(self.alpha0 / dist_mec_vehicle)
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_mec_vehicle / p_noise)
        t_tr = offloading_ratio * task_size / trans_rate
        t_edge = offloading_ratio * task_size / (self.f_MEC / self.s)
        t_local = (1 - offloading_ratio) * task_size / (self.f_ve / self.s)
        if t_tr < 0 or t_edge < 0 or t_local < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        t_MEC = t_tr + t_edge
        time_delay = max(t_MEC, t_local)
        return time_delay

    def step(self):
        step_redo = False
        is_terminal = False
        ve_id = np.random.randint(0, self.M)
        offloading_ratio = 1
        task_size = self.task_list[ve_id]
        block_flag = self.block_flag_list[ve_id]
        if self.MEC_Computing == 0:
            is_terminal = True
            reward = 0
        elif self.MEC_Computing - self.task_list[ve_id] < 0:
            self.task_list = np.ones(self.M) * self.MEC_Computing
            reward = 0
            step_redo = True
        elif self.time_delay(self.loc_ve_list[ve_id], offloading_ratio, task_size, block_flag) >= 1000:
            print('任务超过最大时延，跳过本次迭代！！！')
            reward = 0
        else:
            delay = self.com_delay(self.loc_ve_list[ve_id], offloading_ratio, task_size, block_flag)
            reward = -(delay/100)

            self.reset2(offloading_ratio, task_size, ve_id)
        return reward, is_terminal, step_redo


    def reset2(self, offloading_ratio, task_size, ve_id):
        self.MEC_Computing -= self.task_list[ve_id]
        for i in range(self.M):
            tmp = np.random.rand(2)
            theta_ve = tmp[0] * np.pi * 2
            block_flag = self.block_flag_list[ve_id]
            t = self.time_delay(self.loc_ve_list[ve_id], offloading_ratio, task_size, block_flag)
            dis_ve = tmp[1] * t * self.v_ve
            self.loc_ve_list[i][0] = self.loc_ve_list[i][0] + math.cos(theta_ve) * dis_ve
            self.loc_ve_list[i][1] = self.loc_ve_list[i][1] + math.sin(theta_ve) * dis_ve
            self.loc_ve_list[i] = np.clip(self.loc_ve_list[i], 0, self.ground_width)
        self.reset_step()

    def com_delay(self, loc_ve, offloading_ratio, task_size, block_flag):
        loc_uav = [500, 500]
        p1 = 0.2
        p2 = 0.3
        p3 = 0.5
        p = 0.5
        q = 0.5
        dx = loc_uav[0] - loc_ve[0]
        dy = loc_uav[1] - loc_ve[1]
        dh = self.height
        dist_mec_vehicle = dx * dx + dy * dy + dh * dh
        p_noise = self.p_noisy_los
        if block_flag == 1:
            p_noise = self.p_noisy_nlos
        g_mec_vehicle = abs(self.alpha0 / dist_mec_vehicle)
        trans_rate = self.B * math.log2(1 + self.p_uplink * g_mec_vehicle / p_noise)
        t_tr = offloading_ratio * task_size / trans_rate
        t_edge = offloading_ratio * task_size / (self.f_MEC / self.s)
        t_local = (1 - offloading_ratio) * task_size / (self.f_ve / self.s)
        if t_tr < 0 or t_edge < 0 or t_local < 0:
            raise Exception(print("+++++++++++++++++!! error !!+++++++++++++++++++++++"))
        t_MEC = t_tr + t_edge
        time_delay = max(t_MEC, t_local)
        energy_consumption = t_local * p1 + t_tr * p2 + t_edge * p3
        s = p * time_delay + q * energy_consumption
        return s


def Full_Offloading():
    n = 100
    s2 = 0
    for _ in tqdm(range(n)):
        env = MEC_env()
        env.reset()
        s1 = 0
        i = 0
        while i < env.slot_num:
            delay, is_terminal, step_redo = env.step()
            s1 += delay
            i = i + 1
        s2 += s1
    print((s2/n)*100)

if __name__ == '__main__':
    Full_Offloading()
