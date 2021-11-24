import os
import logging
import gym
import random
import numpy as np
from collections import deque
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
torch.set_default_tensor_type('torch.FloatTensor')
logging.basicConfig(filename='pacman.log', filemode='w', level=logging.INFO)

#global variables
# seed=1234
env = gym.make('MsPacman-v0')
# env.seed(seed)
# random.seed(seed)
# np.random.seed(seed)
# torch.manual_seed(seed)
gamma=0.99
batch_size=256
eps_start=1
eps_end=0.01
eps_decay=(eps_start-eps_end)/10000000
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dim_output=env.action_space.n
num_iterations=50000
#whether we want to train the model or test the model
training=True


#a function to process the image into 88*80
mspacman_color=210+164+74
def preprocess_observation(obs):
    img = obs[1:176:2, ::2]  # crop and downsize
    img = img.sum(axis=2)  # to greyscale
    img[img == mspacman_color] = 0  # Improve contrast
    img = (img // 3 - 128).astype(np.int8)  # normalize from -128 to 127
    #return img.reshape(88, 80, 1)
    return torch.FloatTensor(img.reshape([88, 80])).unsqueeze(0).unsqueeze(0)

#buffer
class Replay_buffer(object):
    def __init__(self,capacity):
        self.buffer=deque(maxlen=capacity)
    def append(self,m):
        self.buffer.append(m)
    def __len__(self):
        return len(self.buffer)
    def sample(self,num):
        batch=random.sample(self.buffer,num)
        return map(lambda x: Variable(torch.cat(x,0).to(device)),zip(*batch))
buffer_size=50000
buffer=Replay_buffer(buffer_size)

#a dqn neural network---seems to me we are able to determine the direction based on only one screenshot
#but the difference is pretty small, so I still use the same strategy as HW2 and stack 4 figures together
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1=nn.Conv2d(4,32,kernel_size=5,stride=2)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,64,kernel_size=5,stride=2)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3=nn.Conv2d(64,32,kernel_size=5,stride=2)
        self.bn3=nn.BatchNorm2d(32)
        self.fc1=nn.Linear(1792,512)
        self.fc2=nn.Linear(512,dim_output)
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)
try:
    net_train=torch.load('dqn_pacman.pt',map_location=torch.device(device)).to(device)
    net_eval=torch.load('dqn_pacman.pt',map_location=torch.device(device)).to(device)
    logging.info('use existing model')
except:
    net_train=DQN().to(device)
    net_eval=DQN().to(device)
    logging.info('use new model')
net_train.train()
net_eval.eval()
optimizer = optim.Adam(net_train.parameters(),lr=1e-5)
criterion=nn.MSELoss()

#a function to select actions
def select_action(state,eps):
    action_prob=np.ones(dim_output,np.float32)*eps/dim_output
    max_q,max_q_index=net_train(Variable(state.to(device))).data.cpu().max(),net_train(Variable(state.to(device))).data.cpu().argmax()
    action_prob[max_q_index]=(1-eps)+action_prob[max_q_index]
    action=np.random.choice(np.arange(dim_output),p=action_prob)
    next_state,reward,done,_=env.step(action)
    next_state=torch.cat([state.narrow(1,1,3), preprocess_observation(next_state)], 1)
    buffer.append((state,torch.LongTensor([int(action)]),torch.Tensor([reward]),next_state,torch.Tensor([done])))
    return next_state,reward,done,max_q


eps=eps_start
i_frame,running_reward,running_q=0,0,0
state=torch.cat([preprocess_observation(env.reset())]*4,1)
#fill the buffer with some random experience
for _ in range(buffer_size//5):
    next_state,reward,done,_=select_action(state,eps)
    if done:
        state=torch.cat([preprocess_observation(env.reset())]*4,1)
    else:
        state=next_state
#train the model
for i_episode in range(num_iterations):
    cur_reward,cur_q_sum,ep_len,done=0,0,0,False
    state=torch.cat([preprocess_observation(env.reset())]*4,1)
    while not done:
        eps=max(eps_end,eps-eps_decay)
        next_state,reward,done,qval=select_action(state,eps)
        cur_reward,cur_q_sum=cur_reward+reward,cur_q_sum+qval
        state=next_state
        i_frame,ep_len=i_frame+1,ep_len+1
        if training:
            if ep_len%5==0:
                batch_state,batch_action,batch_reward,batch_next_state,batch_done=buffer.sample(batch_size)
                batch_q=net_train(batch_state).gather(1,batch_action.unsqueeze(1)).squeeze(1)
                batch_next_q=net_eval(batch_next_state).detach().max(1)[0]*gamma*(1-batch_done)
                loss=criterion(batch_q,batch_reward+batch_next_q)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net_train.parameters(),2)
                optimizer.step()
    #print results and save model
    running_reward=0.1*cur_reward+0.9*running_reward
    running_q=0.1*cur_q_sum/ep_len+0.9*running_q
    logging.info('episode %d episode length %d frame %d cur_rew %.3f mean_rew %.3f cur_maxq %.3f mean_maxq %.3f' % (i_episode,ep_len,i_frame,cur_reward,running_reward,cur_q_sum/ep_len,running_q))
    net_eval.load_state_dict(net_train.state_dict())
    if i_episode%100==0:
        torch.save(net_train,'dqn_pacman.pt')
