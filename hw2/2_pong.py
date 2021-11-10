import gym
import sys
import os
import numpy as np
import logging
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
#os.chdir('/Users/lesliezhao/Dropbox/nu_course/2021_fall/DP/IEMS469_HW_LWZ/hw2/model/')
logging.basicConfig(filename='ac_base.log', filemode='w', level=logging.INFO)


#global parameters
GAMMA = 0.99
EPISODE = 5000
BATCH_SIZE=32
seed=1234
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
env = gym.make('Pong-v0')
np.random.seed(seed)
env.seed(seed)
torch.manual_seed(seed)
learning_rate=1e-5
#state space dimension
dim_input=(1,80,80)
#action space dimension
dim_output=2
eps = np.finfo(np.float32).eps.item()


def preprocess(image):
    image = image[35:195]  # crop
    image = image[::2, ::2, 0]  # downsample by factor of 2
    image[image == 144] = 0  # erase background (background type 1)
    image[image == 109] = 0  # erase background (background type 2)
    image[image != 0] = 1  # everything else (paddles, ball) just set to 1
    return np.reshape(image, dim_input[1:])

class Policy(nn.Module):
    def __init__(self,dropout=0.5):
        super(Policy,self).__init__()
        #in_channels,out_channels,kernel_size
        self.con1 = nn.Conv2d(4, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.con2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.con3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.fc1 = nn.Linear(1568,256)
        self.fc2 = nn.Linear(256,dim_output)
        self.fc3 = nn.Linear(256,1)
        self.saved_log_probs=[]
        self.saved_state_values=[]
    def forward(self,x):
        x = F.relu(self.bn1((self.con1(x))))
        x = F.relu(self.bn2((self.con2(x))))
        x = F.relu(self.bn3((self.con3(x))))
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        action_probs=F.softmax(self.fc2(x), dim=1)
        state_values=self.fc3(x)
        return action_probs,state_values
    def select_action(self, state):
        probs,state_value = self(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        self.saved_state_values.append(state_value)
        return action.item()+2

try:
    policy = torch.load('policy_gradient_base.pt',map_location=torch.device(device)).to(device)
    logging.info('use a saved model')
except:
    policy=Policy().to(device)
    logging.info('start a new model')
optimizer = optim.Adam(policy.parameters(), lr=learning_rate)


def run_episode(env,policy):
    state = env.reset()
    rewards = []
    done= False
    states_stack=np.zeros((4,80,80))
    while not done:
        state = preprocess(state)
        states_stack[0], states_stack[1], states_stack[2], states_stack[3] = state, states_stack[0], states_stack[1], states_stack[2]
        action = policy.select_action(torch.tensor(states_stack).to(device).unsqueeze(0).float())
        state, reward, done, info = env.step(action)
        rewards.append(reward)
    return rewards

def discount_rewards(r, gamma=0.99):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        if r[t] != 0:
            running_add = 0
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    #standardize
    discounted_r -= np.mean(discounted_r)
    discounted_r /= np.std(discounted_r) + eps
    return discounted_r



losses,cur_rewards,running_rewards=[],[],[]
running_reward=-21
episodes_rewards=[]
policy_loss=[]
value_loss=[]
logging.basicConfig(filename='base.log', filemode='w', level=logging.INFO)
for episode in range(EPISODE):
    # Gather samples
    rewards = run_episode(env, policy)
    running_reward=0.1*np.sum(rewards)+0.9*running_reward
    episodes_rewards.append(discount_rewards(rewards))
    if episode%BATCH_SIZE==0:
        # Update policy network
        flatten_episodes_rewards=[item for items in episodes_rewards for item in items]
        #calculate loss function
        for log_prob,state_value,reward in zip(policy.saved_log_probs,policy.saved_state_values,flatten_episodes_rewards):
            advantage=reward-state_value
            advantage=advantage.detach()
            policy_loss.append(-log_prob*advantage)
            value_loss.append(F.smooth_l1_loss(state_value.reshape(-1),torch.tensor([np.float(reward)]).to(device)))

        # loss = [-log_prob * reward for log_prob, reward in zip(policy.saved_log_probs,flatten_episodes_rewards)]
        optimizer.zero_grad()
        policy_loss=torch.stack(policy_loss).sum()
        value_loss=torch.stack(value_loss).sum()
        loss=policy_loss+value_loss*0.5
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(),2)
        optimizer.step()
        #clean up the space
        del policy.saved_log_probs[:]
        del policy.saved_state_values[:]
        del episodes_rewards[:]
        policy_loss = []
        value_loss = []
    #collect numbers and print current progress
    output_string="Episode {}: current reward is {}, running reward is {}".format(episode, np.sum(rewards),running_reward)
    logging.info(output_string)
    # losses.append(loss.item())
    # cur_rewards.append(np.sum(rewards))
    # running_rewards.append(running_reward)
    if episode%100==0 and episode!=100:
        print('save model')
        torch.save(policy,'policy_gradient_base.pt')
