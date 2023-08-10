import torch
import torch.nn as nn
import torch.nn.functional as F



class Policy(nn.Module):
    def __init__(self,state_dim,action_dim):
        super(Policy,self).__init__()

        self.input_dim=state_dim
        self.output_dim=action_dim

        self.l1 = nn.Linear(self.input_dim, 128)
        self.l2 = nn.Linear(128,256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128,self.output_dim)

        self.saved_log_probs = []
        self.saved_all_log_probs=[]
        self.saved_all_probs = []
        self.rewards = []
        self.returns=[]


    def forward(self, x): 
        '''the output is a probability distribution with STATE_SPACE dimension'''
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x=self.l4(x)

        return F.softmax(x,dim=1) 
        '''It is applied to all slices along dim, 
        and will re-scale them so that the elements lie in the range [0, 1] and sum to 1'''

import torch
import torch.nn as nn
import torch.nn.functional as F



class Policy2(nn.Module):  
    def __init__(self,ptb_dim,action_dim):
        super(Policy2,self).__init__()

        self.input1_dim=7                    
        self.input2_dim=ptb_dim
        self.output_dim=action_dim

        self.l1 = nn.Linear(self.input1_dim, 128)
        self.l1_1 = nn.Linear(self.input2_dim, 128)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 128)
        self.l4 = nn.Linear(128, self.output_dim)

        self.saved_log_probs = []
        self.saved_all_log_probs=[]
        self.saved_all_probs = []
        self.rewards = []
        self.returns=[]

        '''the output is a probability distribution with STATE_SPACE dimension'''
    def forward(self, x, u):
        x = F.relu(self.l1(x))
        u = F.relu(self.l1_1(u))
        x = F.relu(self.l2(torch.cat([x, u], dim=1)))
        x = F.relu(self.l3(x))
        x = self.l4(x)

        return F.softmax(x,dim=1) 
        '''It is applied to all slices along dim, 
        and will re-scale them so that the elements lie in the range [0, 1] and sum to 1'''

    
