import os
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import random
import logging
import argparse
import matplotlib.pyplot as plt

#global variables
np.random.seed(42)
alpha=1   #weight for bonus
n_episode=1000


def sample_jester_data(file_name, context_dim = 32, num_actions = 8, num_contexts = 19181,
    shuffle_rows=True, shuffle_cols=False):
    """Samples bandit game from (user, joke) dense subset of Jester dataset.
    Args:
       file_name: Route of file containing the modified Jester dataset.
       context_dim: Context dimension (i.e. vector with some ratings from a user).
       num_actions: Number of actions (number of joke ratings to predict).
       num_contexts: Number of contexts to sample.
       shuffle_rows: If True, rows from original dataset are shuffled.
       shuffle_cols: Whether or not context/action jokes are randomly shuffled.
    Returns:
       dataset: Sampled matrix with rows: (context, rating_1, ..., rating_k).
       opt_vals: Vector of deterministic optimal (reward, action) for each context.
    """
    np.random.seed(0)
    with tf.gfile.Open(file_name,'rb') as f:
        dataset=np.load(f)
    if shuffle_cols:
        dataset=dataset[:,np.random.permutation(dataset.shape[1])]
    if shuffle_rows:
        np.random.shuffle(dataset)
    dataset=dataset[:num_contexts,:]
    assert context_dim+num_actions==dataset.shape[1],'Wrong data dimensions.'
    opt_actions=np.argmax(dataset[:, context_dim:], axis=1)
    opt_rewards=np.array([dataset[i, context_dim + a] for i, a in enumerate(opt_actions)])
    return dataset,opt_rewards,opt_actions
#(19181,40) for 40 jokes, (19181,) for reward, (19181,) for action
dataset,rewards,actions = sample_jester_data('jester_data_40jokes_19181users.npy')

#initialize covariance matrix as identity matrix
mat_A=np.zeros((8,32,32))
for a in range(8):
    mat_A[a]=np.identity(32)
b = np.zeros((8,32))
p = np.zeros(8)

#ucb for one user
def ucb(ind,training=True):
    x=dataset[ind][:32]
    for a in range(8):
        inv_A=np.linalg.inv(mat_A[a])
        theta=np.dot(inv_A,b[a])
        p[a]=np.dot(theta,x)+alpha*np.sqrt(np.dot(np.dot(x,inv_A),x))+np.random.rand()/1000000
    cur_regret=rewards[ind]-dataset[ind][int(np.argmax(p))+32]
    if training:
        mat_A[int(np.argmax(p))]+=np.expand_dims(x,1).dot(np.expand_dims(x,0))
        b[int(np.argmax(p))]+=dataset[ind][int(np.argmax(p))+32]*x
    return cur_regret
#training
total_regret=0
for eps in range(n_episode):
    print(eps)
    for user in range(18000):
        cur_regret=ucb(user,training=True)
        total_regret+=cur_regret
    print(total_regret,total_regret/((eps+1)*18000))
#testing
collector_regret=[]
total_regret=0
for user in range(18000, 19181):
    cur_regret=ucb(user,training=False)
    total_regret+=cur_regret
    collector_regret.append(total_regret)
print(collector_regret[-1])
plt.plot(collector_regret,label='Regret')
plt.xlabel('Users')
plt.ylabel('Cumulative Regret')
plt.legend()
plt.savefig('results.pdf')
plt.show()