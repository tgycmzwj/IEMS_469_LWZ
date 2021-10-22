"""
Code for RL Assignment 1
Author: Weijia Zhao @ Kellogg Finance


Additional assumption:
time t events: dispatch shuttle->new consumers come->calculate waiting cost
Under my assumption, all consumers have to wait at least one period
One can easily modify the code for alternative timeline: new consumers come->dispatch shuttle->calculate waiting cost
"""

import random
import numpy as np
import matplotlib.pyplot as plt
import itertools

import time



#define global variables
#capacity of shuttle if it is dispatched
K=15
#cost of dispatching a shuttle
c_f=100
#cost per customer left waiting per time period, assume it is already sorted
c_h=[2,2.5,3]
#number of customers arriving during time interval t
A_min=1
A_max=5
#upper bound of num of people of each type in station
p_max=30
#discount rate
gamma=0.95
#time period for enumeration method
T=500
#possible actions
action=[0,1]
#convergence criteria
max_value_iter=10**7
toler_value_iter=10**-6
max_policy_iter=10**7
toler_policy_iter=10**-6

#c_h=[1,1.5,2,2.5,3]
#p_max=100

#define some supporting function as the state space becomes complicated in multi case
def reward(s,a,c_h,c_f):
    """
    :param s: a tuple of the number of consumers in each type
    :param a: 0/1 no_dispatch/dispath
    :param c_f: cost of dispath
    :param c_h: a vector of the waiting cost for consumers of each type
    :return: one period reward
    """
    reward=0-a*c_f
    for i in range(len(s)):
        reward=reward-c_h[i]*s[i]
    return reward
def state_after_dispatch(s,a,K):
    """
    :param s: a tuple of the number of consumers in each type
    :param a: 0/1 no_dispatch/dispatch
    :return: a tuple of number of consumers in each type after the bus pick up
    """
    #Always better to first dispatch the consumers with larger waiting cost
    if a==0:
        return s
    space_left=K
    s_vector=list(s)
    for i in range(len(s)-1,-1,-1):
        num_pickup=min(space_left,s[i])
        space_left=space_left-num_pickup
        s_vector[i]=s_vector[i]-num_pickup
        if space_left==0:
            break
    return tuple(s_vector)
def states_after_arrival(s,p_max,A_min,A_max):
    """
    :param s: a tuple of the number of existing consumers in each type
    :param p_max: upper bound of num of people of each type in station
    :param A_min,A_max: number of customers arriving during time interval t
    :return: all possibilities of the new state after arrival of new consumers in a vector, with each case as a tuple
    """
    new_states=[]
    new_comings=list(itertools.product(list(range(A_min,A_max+1)),repeat=len(s)))
    for new_coming in new_comings:
        temp=list(s)
        new_state=[min(temp[j]+new_coming[j],p_max) for j in range(len(temp))]
        new_states.append(tuple(new_state))
    return new_states
def all_states(c_h,p_max):
    """
    :param c_h: a vector of cost per customer left waiting per time period
    :param p_max: upper bound of num of people of each type in station
    :return: all possible states
    """
    return list(itertools.product(list(range(p_max+1)),repeat=len(c_h)))



#enumeration method
def enumeration(K,c_f,c_h,A_min,A_max,p_max,gamma,T,action):
    #initialize value function, key is the current state, len(c_h) length tuple, value is [value_no_dispatch,value_dispatch]
    all_states_avail = all_states(c_h, p_max)
    V={k:[0,0] for k in all_states_avail}
    V_pre={k:[0,0] for k in all_states_avail}
    for t in range(T,-1,-1):
        print(t)
        for s in all_states_avail:
            for a in action:
                s_after_dispatch=state_after_dispatch(s,a,K)
                ss_after_arrival=states_after_arrival(s_after_dispatch,p_max,A_min,A_max)
                probability=1/len(ss_after_arrival)
                for s_after_arrival in ss_after_arrival:
                    cur_reward=reward(s_after_arrival,a,c_h,c_f)
                    V[s][a]=V[s][a]+probability*(cur_reward+gamma*max(V_pre[s_after_arrival]))
        V_pre=V.copy()
        V={k:[0,0] for k in all_states_avail}
    V_results=V_pre.copy()
    for s in V_results.keys():
        V_results[s]=max(V_results[s])
    return V_results
results_enumeration=enumeration(K,c_f,c_h,A_min,A_max,p_max,gamma,T,action)
plt.plot(list(results_enumeration.values()))
plt.title('enumeration')
plt.savefig('q2_enumeration.pdf')
plt.clf()


#value iteration method
def value_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_iter,tolerance):
    # initialize value function, key is the current state, len(c_h) length tuple, value is [value_no_dispatch,value_dispatch]
    all_states_avail = all_states(c_h, p_max)
    V = {k: [0, 0] for k in all_states_avail}
    V_pre = {k: [0, 0] for k in all_states_avail}
    V_results_pre = V_pre.copy()
    for s in V_results_pre.keys():
        V_results_pre[s] = max(V_results_pre[s])

    cur_iter=0
    cur_error=np.abs(tolerance)*10
    while cur_iter<max_iter and cur_error>tolerance:
        cur_iter=cur_iter+1
        print('cur_iter=',cur_iter,', cur_error=',cur_error)
        for s in all_states_avail:
            for a in action:
                s_after_dispatch=state_after_dispatch(s,a,K)
                ss_after_arrival=states_after_arrival(s_after_dispatch,p_max,A_min,A_max)
                probability = 1 / len(ss_after_arrival)
                for s_after_arrival in ss_after_arrival:
                    cur_reward = reward(s_after_arrival, a, c_h, c_f)
                    V[s][a] = V[s][a] + probability * (cur_reward + gamma * V_results_pre[s_after_arrival])
        V_pre=V.copy()
        V={k:[0,0] for k in all_states_avail}
        V_results = V_pre.copy()
        for s in V_results.keys():
            V_results[s] = max(V_results[s])
        cur_error=np.linalg.norm([V_results[key]-V_results_pre[key] for key in V_results.keys()])
        V_results_pre=V_results.copy()
    return V_results_pre
results_value_iter=value_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_value_iter,toler_value_iter)
plt.plot(list(results_value_iter.values()))
plt.title('value_iteration')
plt.savefig('q2_value_iteration.pdf')
plt.clf()


#policy iteration method
def policy_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_iter_out,tolerance_out,max_iter_in,tolerance_in):
    # initialize value function, key is the current state, len(c_h) length tuple, value is [value_no_dispatch,value_dispatch]
    all_states_avail = all_states(c_h, p_max)
    V = {k: [0, 0] for k in all_states_avail}
    V_pre = {k: [0, 0] for k in all_states_avail}
    V_results = {k: 0 for k in all_states_avail}
    V_results_pre = {k: 0 for k in all_states_avail}
    for s in V_results_pre.keys():
        V_results_pre[s] = max(V_pre[s])
    pi={k: 0 for k in all_states_avail}
    pi_pre={k: 0 for k in all_states_avail}

    cur_iter_out = 0
    cur_error_out = np.abs(tolerance_out) * 10
    while cur_iter_out < max_iter_out and cur_error_out > tolerance_out:
        cur_iter_out = cur_iter_out + 1
        cur_iter_in = 0
        cur_error_in = np.abs(tolerance_in) * 10
        print('cur_iter_out=', cur_iter_out, ', cur_error_out=', cur_error_out)
        #compute converged value function
        while cur_iter_in < max_iter_in and cur_error_in > tolerance_in:
            cur_iter_in = cur_iter_in + 1
            print('cur_iter_in=', cur_iter_in, ', cur_error_in=', cur_error_in)
            for s in all_states_avail:
                a=pi_pre[s]
                s_after_dispatch=state_after_dispatch(s,a,K)
                ss_after_arrival=states_after_arrival(s_after_dispatch,p_max,A_min,A_max)
                probability = 1/len(ss_after_arrival)
                for s_after_arrival in ss_after_arrival:
                    cur_reward = reward(s_after_arrival, a, c_h, c_f)
                    V[s][a]=V[s][a]+probability*(cur_reward+gamma*V_results_pre[s_after_arrival])
            V_pre = V.copy()
            V = {k: [0, 0] for k in all_states_avail}
            for s in V_results.keys():
                V_results[s] = V_pre[s][pi_pre[s]]
            cur_error_in = np.linalg.norm([V_results[key] - V_results_pre[key] for key in V_results.keys()])
            V_results_pre = V_results.copy()
            V_results={k: 0 for k in all_states_avail}

        #compute policy function
        for s in all_states_avail:
            total_reward=np.zeros(len(action))
            for a in action:
                s_after_dispatch=state_after_dispatch(s,a,K)
                ss_after_arrival=states_after_arrival(s_after_dispatch,p_max,A_min,A_max)
                probability = 1 / len(ss_after_arrival)
                for s_after_arrival in ss_after_arrival:
                    cur_reward = reward(s_after_arrival, a, c_h, c_f)
                    total_reward[a]=total_reward[a]+probability*(cur_reward+gamma*V_results_pre[s_after_arrival])
            pi[s]=np.argmax(total_reward)
        cur_error_out = np.linalg.norm([pi[key] - pi_pre[key] for key in pi.keys()])
        pi_pre=pi.copy()
        pi={k: 0 for k in all_states_avail}
    return pi_pre
results_policy_iter=policy_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_policy_iter,toler_policy_iter,max_value_iter,toler_value_iter)
plt.plot(list(results_policy_iter.values()))
plt.title('policy iteration')
plt.show()
plt.savefig('q2_policy_iteration.pdf')
plt.clf()
