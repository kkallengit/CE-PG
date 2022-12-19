#   Museum environment setup
import numpy as np
from random import choice
class env_multi_robot:
    def __init__(
            self,
            graph,
            trans_prob_matrix,
            stateSpace,
            robotState,
            targetState,
            robotNum,
    ):
        self.reset_robot_state_ = robotState
        self.reset_target_state_ = targetState
        self.robot_num_ = robotNum
        self.graph_ = graph
        self.trans_prob_matrix = trans_prob_matrix
        self.state_space_ = stateSpace
        self.robot_states_ = [robotState for i in range(robotNum)]
        self.target_state_ = targetState
        self.done_ = False
        self.eta_fp_ = 0.0
        self.eta_fn_ = 0.0

    

    def hmm_simulator(self):
        hmm_state = np.zeros(self.state_space_, dtype=float, order='C')
        for i in range(self.state_space_):
            if i == self.target_state_:
                hmm_state[i] = 0.5
                # hmm_state[i] = 1.0
            else:
                hmm_state[i] = 0.5 / (self.state_space_ - 1)
                # hmm_state[i] = 0.
        return hmm_state

    def update_env(self, action):
        # 更新robot位置
        rewards = np.zeros(self.robot_num_, dtype=int, order='C')
        for i in range(self.robot_num_):
            if action[i] in self.graph_[self.robot_states_[i]]:
                self.robot_states_[i] = action[i]
                if self.robot_states_[i]==self.target_state_:
                    rewards[i] = 5
                else:rewards[i]=-1
            else:
                rewards[i] = -2
        traget_next_state = np.random.choice(range(len(self.trans_prob_matrix[self.target_state_])), p=self.trans_prob_matrix[self.target_state_]) #lazy target transition matrix
        if traget_next_state != 0:
            self.target_state_ = traget_next_state
        return rewards

    def generate_robot_obs(self):
        obs_robot = []
        for i in range(self.robot_num_):
            cur = np.zeros(self.state_space_, dtype=float, order='C')
            cur[self.robot_states_[i]] = 1.0
            obs_robot.append(cur)
        return obs_robot

    def update_env_obs(self):
        obs_env = []
        for robot_state in self.robot_states_:
            if robot_state == self.target_state_:
                a = np.random.rand()
                if a < self.eta_fn_:
                    obs_env.append(0)
                else:
                    obs_env.append(1)
            else:
                a = np.random.rand()
                if a < self.eta_fp_:
                    obs_env.append(1)
                else:
                    obs_env.append(0)
        return obs_env


    def step(self, action):
        reward = self.update_env(action)
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()
        for i in range(len(obs_env)):
            if obs_env[i] == 1 and self.robot_states_[i] == self.target_state_:
                self.done_ = True
        return obs_robot, obs_env, reward, self.target_state_, self.done_

    def reset(self):
        self.done_ = False
        # self.graph_ = self.graphdict_setup()
        self.robot_states_ = [self.reset_robot_state_ for i in range(self.robot_num_)]
        self.target_state_ = self.reset_target_state_
        true_hmm = self.hmm_simulator()
        obs_robot = self.generate_robot_obs()
        obs_env = self.update_env_obs()

        return obs_robot, obs_env, self.target_state_

        #represent robot location with ROBOT_NUM 


