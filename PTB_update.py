import numpy as np

class PTB:
    def __init__(self, graph, start_prob, transititon_prob, fp_prob, fn_prob):
        self.graph_ = graph
        self.state_space_ = len(graph)
        self.ptb_ = start_prob
        self.gamma_ = transititon_prob
        self.eta_fp_ = fp_prob
        self.eta_fn_ = fn_prob
        self.obss_ = []
        self.poss_ = []
        self.lambda_ = np.zeros((self.state_space_, self.state_space_))

    def update_lambda_matrix(self, robot_pos, observation):
        for i in range(len(self.lambda_)):
            if i == robot_pos:
                if observation == 1:
                    self.lambda_[i][i] = 1 - self.eta_fn_
                else:   self.lambda_[i][i] = self.eta_fn_
            else:
                if observation == 1:
                    self.lambda_[i][i] = self.eta_fp_
                else:   self.lambda_[i][i] = 1 - self.eta_fp_

    def store_trajectory(self, robot_pos, observation):
        self.obss_.append(observation)
        self.poss_.append(robot_pos)

    def update_ptb(self, robot_pos, observation):
        self.store_trajectory(robot_pos, observation)
        self.update_lambda_matrix(robot_pos, observation)
        buffer = np.matmul(self.lambda_, self.gamma_)
        self.ptb_ = np.matmul(buffer, self.ptb_)
        sum = 0.
        for i in self.ptb_:
            sum += i
        weight = 1 / sum
        self.ptb_ = self.ptb_ * weight 
        return self.ptb_

    def get_current_ptb(self):
        return self.ptb_


