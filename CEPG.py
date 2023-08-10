import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import random
import math


from func import env_multi_robot
from PTB_update import PTB
from env import graphdict_setup
from network import Policy, Policy2

from tensorboardX import SummaryWriter


directory = './maddpg/'
writer=SummaryWriter(directory)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

env_name="test_env" # test_env  office    museum
robot_num=2


ROBOT_NUM = robot_num





if env_name=="test_env":
    STATE_SPACE = 11
    INIT_Searcher_State=5
elif env_name=='office':
    STATE_SPACE = 61
    INIT_Searcher_State=43
elif env_name=='museum':
    STATE_SPACE = 71
    INIT_Searcher_State=1
else:raise NameError("Env is wrong!! Please choose right env_name")

gamma = 0.95
state_dim=STATE_SPACE*2
action_dim=STATE_SPACE
ptb_dim=STATE_SPACE
policy = [Policy2(ptb_dim,action_dim) for i in range(ROBOT_NUM)]

optimizer = []
for i in range(ROBOT_NUM):
    optimizer.append(optim.Adam(policy[i].parameters(), lr=0.001)) #lr：learning rate
beta = [0.7 for i in range(ROBOT_NUM)]
eps = 0.01


def manualInitEmm(graph):
    lazy_gumma = np.zeros((STATE_SPACE, STATE_SPACE), dtype=float) #
    for i in range(len(lazy_gumma)):
        size = len(graph[i])
        for j in range(len(lazy_gumma)):
            if i == j:
                lazy_gumma[i][j] = 0.9
            elif j in graph[i]:
                lazy_gumma[i][j] = 0.1 / (size - 1)
            else:
                lazy_gumma[i][j] = 0.

    random_gumma=np.zeros((STATE_SPACE, STATE_SPACE), dtype=float)  #
    for i in range(len(random_gumma)):
        size = len(graph[i])
        for j in range(len(random_gumma)):
            if j in graph[i]:
                random_gumma[i][j] = 1 / (size)
            else:
                random_gumma[i][j] = 0.

    gumma=lazy_gumma
    return gumma


def get_robot_state(observation):
    for i in range(len(observation)):
        if observation[i] == 1.:
            return i

def get_network_input(state):
    robot_state=[get_robot_state(state)]
    ptb=state[-STATE_SPACE:]



    robot_state_bina=bin(robot_state[0])[2:].rjust(7,'0')
    robot_state_bina=[int(robot_state_bina[i]) for i in range(len(robot_state_bina))]  #

    ptb_10times=[elem*10 for elem in ptb]
    ptb_expontial=[math.exp(elem) for elem in ptb]
    ptb_all=[math.exp(elem*2) for elem in ptb]

    network_input1=np.array(robot_state_bina)
    network_input2=np.array(ptb_all)

    return network_input1,network_input2

def select_action(state, num):
    robot_state = get_robot_state(state)
    possible_action_line = graph_[robot_state]
    input1,input2=get_network_input(state)

    input1=torch.from_numpy(input1).float().unsqueeze(0)
    input2=torch.from_numpy(input2).float().unsqueeze(0)
    #state = torch.from_numpy(state).float().unsqueeze(0) #change  the dimension (from 122 to [1，122])

    probs = policy[num](input1,input2) #probs is the output of neural net（probability distribution） [1，61]dimension tensor
    prob_weights = probs.clone()
    prob_sum = 0.
    for i in graph_[robot_state]:
        prob_sum += probs[0][i]   # where torch.Size([1,61])，prob_sum represent the total probability 
    if prob_sum == 0.:
        length = len(graph_[robot_state])
        for i in range(len(probs[0])):
            if i in graph_[robot_state]:
                prob_weights[0][i] = 1.0 / length #Uniform selection probability
            else:
                prob_weights[0][i] = 0.
    else:
        for i in range(len(probs[0])):
            if i in graph_[robot_state]:
                prob_weights[0][i] = probs[0][i] / prob_sum
            else:
                prob_weights[0][i] = 0.
    '''at this time, only the location connected with current node have non-zero elements '''
    m = Categorical(prob_weights)      # distribution
    action = m.sample()           #  sample
    policy[num].saved_log_probs.append(m.log_prob(action))    # calculate  logp(a∣πθ(s)), and make it easy to calculate gradient  https://www.cnblogs.com/pprp/p/14285062.html
    
    for act in possible_action_line:
        policy[num].saved_all_log_probs.append(m.log_prob(torch.tensor(act)))
    #policy[num].saved_probs.append(prob_weights[0][action])   # restore action's weight 
    return action.item()


def save_other_robot(state,trajectory,num):
    robot_list=[i for i in range(ROBOT_NUM)]
    robot_state=get_robot_state(state)
    #state = torch.from_numpy(state).float().unsqueeze(0) 

    for robot_idx in range(ROBOT_NUM):
        if robot_idx!=num:
            prob_set = []

            input1,input2=get_network_input(state)
            input1=torch.from_numpy(input1).float().unsqueeze(0)
            input2=torch.from_numpy(input2).float().unsqueeze(0)

            probs = policy[robot_idx](input1,input2)
            prob_weights = probs.clone()
            possible_action_line = graph_[robot_state]
            prob_sum = 0.
            for i in possible_action_line:
                prob_sum += probs[0][i]
            if prob_sum == 0.:
                length = len(graph_[robot_state])
                for i in range(len(probs[0])):
                    if i in graph_[robot_state]:
                        prob_weights[0][i] = 1.0 / length #Uniform selection probability
                    else:
                        prob_weights[0][i] = 0.
            else:
                for i in range(len(probs[0])):
                    if i in graph_[robot_state]:
                        prob_weights[0][i] = probs[0][i] / prob_sum
                    else:
                        prob_weights[0][i] = 0.

            for act in possible_action_line:
                if robot_state in trajectory[robot_idx]: #if this state in other robots' trajectory 
                    policy[robot_idx].saved_all_probs.append(prob_weights[0][act])
                else:
                    policy[robot_idx].saved_all_probs.append(0)

def finish_episode(num):
    R = 0
    pg_loss = []
    deterministic_loss=[]
    ce_loss=[]
    returns = []
    for r in policy[num].rewards[::-1]:
        R = r + gamma * R
        returns.insert(0,R)
    returns = torch.tensor(returns)
    if len(returns) <= 1:
        std = 0.
    else:
        std = returns.std()
    returns = (returns - returns.mean()) / (std + eps)
    for log_prob, R in zip(policy[num].saved_log_probs, returns):
        pg_loss.append(-log_prob * R)

    for robot_idx in range(ROBOT_NUM):
        T = len(policy[num].saved_all_probs)
        if robot_idx != num:
            cur_loss = []            
            for other_all_prob, self_all_log_prob in zip(policy[num].saved_all_probs,policy[robot_idx].saved_all_log_probs):
                cur_loss.append(other_all_prob*self_all_log_prob)
            ce_loss.append(sum(cur_loss)/T)

    optimizer[num].zero_grad()
    a = torch.cat(pg_loss).sum() * beta[num]
    b = torch.cat(ce_loss).sum() * ((1 - beta[num]) / (ROBOT_NUM - 1))
    loss = a + b
    # loss = torch.cat(pg_loss).sum() * beta[num] + torch.cat(ce_loss).sum() * ((1 - beta[num]) / (ROBOT_NUM - 1))         #
    if num == ROBOT_NUM - 1:
        loss.backward()
    else:
        loss.backward(retain_graph=True)
    optimizer[num].step()


graph_ = graphdict_setup(env_name,STATE_SPACE)
transition_prob_ = manualInitEmm(graph_)
fp_prob_ = 0.0
fn_prob_ = 0.0
start_prob_ = np.zeros(STATE_SPACE, dtype=float)
for i in range(len(start_prob_)):
    start_prob_[i] = 1.0 / (STATE_SPACE-1)



def main():
    ReturnsCollector = []
    steps_list=[]
    limit = 1000
    for i_episode in range(60000):
        initRobotState =INIT_Searcher_State    #43(office)  5(test_env)
        target_init_list=list(range(1,STATE_SPACE))
        target_init_list.remove(initRobotState)
        initTargetState = random.choice(target_init_list)
        initTargetState=1

        states = [[] for i in range(ROBOT_NUM)]
        returns=[[]for i in range(ROBOT_NUM)]
        trajectory = [[] for i in range(ROBOT_NUM)]

        env = env_multi_robot(graph_, transition_prob_, STATE_SPACE, initRobotState, initTargetState, ROBOT_NUM)
        one_hot_pose, obs, trueTargetState = env.reset()  #
        estimator = [PTB(graph_, start_prob_, transition_prob_, fp_prob_, fn_prob_) for i in range(ROBOT_NUM)]

        start_prob_[initRobotState]=0
        steps=0

        for i in range(ROBOT_NUM):
            states[i] = np.hstack((one_hot_pose[i], start_prob_))
            trajectory[i].append(initRobotState)

        for t in range(1, limit):
            actions = [[] for i in range(ROBOT_NUM)]
            robot_poses = [[] for i in range(ROBOT_NUM)]
            for num in range(ROBOT_NUM):
                robot_poses[num] = get_robot_state(one_hot_pose[num])
                actions[num] = select_action(states[num], num)
                trajectory[num].append(actions[num])
                save_other_robot(states[num],trajectory,num)                
            one_hot_pose, obs, rewards, trueTargetState, done = env.step(actions)
            for num in range(ROBOT_NUM):
                policy[num].rewards.append(rewards[num])
            steps+=1
            if done:
                #print('i_episode:',i_episode ,'  steps',steps)
                break
            for num in range(ROBOT_NUM):
                robot_poses[num] = get_robot_state(one_hot_pose[num])
                ptb_= estimator[num].update_ptb(robot_poses[num], obs)
                states[num] = np.hstack((one_hot_pose[num], ptb_))


        for num in range(ROBOT_NUM):
            finish_episode(num)
        for num in range(ROBOT_NUM):
            del policy[num].saved_log_probs[:]
            del policy[num].saved_all_log_probs[:]
            del policy[num].saved_all_probs[:]
            del policy[num].rewards[:]
            del policy[num].returns[:]

        if i_episode<=1000:
            steps_list.append(steps)
        else:
            if steps<100:
                steps_list.append(steps)  

        # print('steps_list: ',steps_list)
        # print('episode_index',i_episode)

        if len(steps_list)%100==0:

            average_steps=sum(steps_list[-100:])/100
            ReturnsCollector.append(average_steps)
            print('ReturnsCollector',ReturnsCollector)
            print('---------------------------------------------------------------------')
    for i in range(ROBOT_NUM):
        name = '.\model' +str(ROBOT_NUM)+ str(i) + '.pkl'
        torch.save(policy[i], name)
    print(ReturnsCollector)

if __name__ == '__main__':
    main()