import numpy as np
import matplotlib.pyplot as plt
import gym
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import collections
original_stdout = sys.stdout

#define global parameters
seed = 12345
gamma = 0.95
env = gym.make('CartPole-v0')
np.random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#state space dimension
dim_input=env.observation_space.sample().size
#action space dimension
dim_output=env.action_space.n
#fully connected neural network
dim_1=16
learning_rate=3e-3


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.fc1=nn.Linear(dim_input,dim_1)
        self.fc2=nn.Linear(dim_1,dim_output)
        self.fc3=nn.Linear(dim_1,1)
    def forward(self,x):
        x=F.relu(self.fc1(x))
        action_probs=F.softmax(self.fc2(x),dim=1)
        state_values=self.fc3(x)
        return action_probs,state_values

policy = Policy().to(device)
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)

#container for useful information during training
Experience=collections.namedtuple('Experience',field_names=['state','action','prob','state_value','reward','done','new_state'])
#buffer
class ExperienceBuffer:
    def __init__(self,capacity):
        self.buffer=collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.buffer)
    def append(self,experience):
        self.buffer.append(experience)
    def clear(self):
        self.buffer.clear()

def select_action(state):
    state = torch.from_numpy(state).float().unsqueeze(0).to(device)
    probs,state_value = policy(state)
    action=np.random.choice(np.arange(dim_output),p=probs.detach().numpy()[0])
    return action.item(),torch.unsqueeze(torch.log(probs[0][action]),0),state_value

def cal_loss(episodes_buffer):
    BATCH_SIZE=episodes_buffer.__len__()
    all_loss=[]
    for episode in episodes_buffer.buffer:
        R=0
        returns, policy_loss, state_values, value_loss, buffer_rewards, buffer_prob = [], [], [], [],[],[]
        for step in episode:
            buffer_rewards.append(step.reward)
            buffer_prob.append(step.prob)
            state_values.append(step.state_value)
        for r in buffer_rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(device)
        for log_prob, R, v in zip(buffer_prob, returns, state_values):
            advantage=R-v
            policy_loss.append(-log_prob * advantage)
            value_loss.append(F.smooth_l1_loss(v,torch.tensor([np.float(R)])).unsqueeze(0))
        policy_loss=torch.cat(policy_loss).sum()
        value_loss=torch.cat(value_loss).sum()
        all_loss.append(policy_loss+value_loss*0.5)
    total_loss=torch.tensor(0)
    for i in all_loss:
        total_loss=total_loss+i
    return total_loss/BATCH_SIZE


f=open('1_cartpole.txt', 'w')
sys.stdout=f
BATCH_SIZE=5
episodes_buffer=ExperienceBuffer(BATCH_SIZE)
running_reward = 10
i_episode=0
Epochs=640
while i_episode/BATCH_SIZE<Epochs:
    i_episode=i_episode+1
    state= env.reset()
    done = False
    cur_episode=[]
    while not done:
        action,prob,state_value=select_action(state)
        new_state,new_reward,new_done,_=env.step(action)
        cur_episode.append(Experience(state,action,prob,state_value,new_reward,new_done,new_state))
        state,done=new_state,new_done
    episodes_buffer.append(cur_episode)
    ep_reward=0
    for step in cur_episode:
        ep_reward=ep_reward+step.reward
    running_reward = 0.1 * ep_reward + (1-0.1) * running_reward
    #optimization
    if episodes_buffer.__len__()==BATCH_SIZE:
        optimizer.zero_grad()
        loss = cal_loss(episodes_buffer)
        loss.backward()
        optimizer.step()
        episodes_buffer.clear()
        print('Episode {}\tLoss :{:.2f}\tLast reward: {:.2f}\tLast 10 Average reward: {:.2f}'.format(i_episode/BATCH_SIZE, loss, ep_reward, running_reward))
sys.stdout=original_stdout
