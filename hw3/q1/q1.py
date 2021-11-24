import os
import logging
import gym
import random
import numpy as np
from itertools import count
from collections import deque
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
logging.basicConfig(filename='cartpole.txt', filemode='w', level=logging.INFO)

#global variables
# seed=42
env = gym.make('CartPole-v0')
# env.seed(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
gamma=0.95
batch_size=256
eps_start=1
eps_end=0
eps_decay=(eps_start-eps_end)/300000
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim_input=env.observation_space.sample().size
dim_output=env.action_space.n
training=True
if not training:
    eps_start=0


#buffer
class ReplayMemory(object):
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    def push(self,m):
        self.buffer.append(m)
    def __len__(self):
        return len(self.buffer)
    def sample(self,num):
        batch=random.sample(self.buffer,num)
        #batch_state,batch_action,batch_reward,batch_next_state,batch_done
        batch_state=torch.stack([i[0] for i in batch])
        batch_action=torch.stack([i[1] for i in batch])
        batch_reward=torch.stack([i[2] for i in batch])
        batch_next_state=torch.stack([i[3] for i in batch])
        batch_done=torch.stack([i[4] for i in batch])
        return batch_state,batch_action,batch_reward,batch_next_state,batch_done

buffer_size=5000
memory=ReplayMemory(buffer_size)

#a dqn neural network
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1=nn.Linear(dim_input,64)
        self.fc2=nn.Linear(64,dim_output)
    def forward(self, x):
        x = F.relu((self.fc1(x)))
        return self.fc2(x)
try:
    net_train=torch.load('dqn_cartpole.pt',map_location=torch.device(device)).to(device)
    net_eval=torch.load('dqn_cartpole.pt',map_location=torch.device(device)).to(device)
    logging.info('use existing model')
except:
    net_train=DQN().to(device)
    net_eval=DQN().to(device)
    logging.info('use new model')
net_train.train()
net_eval.eval()
optimizer = optim.Adam(net_train.parameters(),lr=1e-3)
crit=nn.MSELoss()

#a function to select actions
def select_action(state,eps):
    action_prob=np.zeros(dim_output,np.float32)
    action_prob.fill(eps/dim_output)
    max_q,max_q_index=net_train(Variable(state.to(device))).data.cpu().max(),net_train(Variable(state.to(device))).data.cpu().argmax()
    action_prob[max_q_index]+=1-eps
    action=np.random.choice(np.arange(dim_output),p=action_prob)
    next_state,reward,done,_=env.step(action)
    next_state=torch.FloatTensor(next_state)
    memory.push((state,torch.LongTensor([int(action)]),torch.Tensor([reward]),next_state,torch.Tensor([done])))
    return next_state,reward,done,max_q


eps=eps_start
i_frame,running_reward,running_q=0,0,0
state=torch.FloatTensor(env.reset())
#fill half of the buffer with some random experience
for _ in range(buffer_size//2):
    next_state,reward,done,_=select_action(state,eps)
    if done:
        state=torch.FloatTensor(env.reset())
    else:
        state=next_state
for i_episode in range(10000):
    cur_reward,cur_q,ep_len,done=0,0,0,False
    state=torch.FloatTensor(env.reset())
    while not done:
        eps=max(eps_end,eps-eps_decay)
        next_state,reward,done,qval=select_action(state,eps)
        cur_reward,cur_q=cur_reward+reward,max(cur_q,qval)
        state=next_state
        i_frame,ep_len=i_frame+1,ep_len+1
        if ep_len%4==0 and training:
            batch_state,batch_action,batch_reward,batch_next_state,batch_done=memory.sample(batch_size)
            batch_q=net_train(batch_state).gather(1,batch_action).squeeze(1)
            batch_next_q=net_eval(batch_next_state).detach().max(1)[0]*gamma*(1-batch_done.squeeze(1))
            loss=crit(batch_q,batch_reward.squeeze(1)+batch_next_q)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net_train.parameters(),2)
            optimizer.step()
    #print results and save model
    running_reward=0.1*cur_reward+0.9*running_reward
    running_q=0.1*cur_q+0.9*running_q
    logging.info('episode %d episode length %d frame %d cur_rew %.3f mean_rew %.3f cur_maxq %.3f mean_maxq %.3f' % (i_episode,ep_len,i_frame,cur_reward,running_reward,cur_q,running_q))
    net_eval.load_state_dict(net_train.state_dict())
    if i_episode%100==0 and training:
        torch.save(net_train,'dqn_cartpole.pt')
