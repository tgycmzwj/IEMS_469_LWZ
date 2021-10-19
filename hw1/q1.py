"""
Code for RL Assignment 1, part 1
Author: Weijia Zhao @ Kellogg Finance


Additional assumption:
time t events: dispatch shuttle->new consumers come->calculate waiting cost
Under my assumption, all consumers have to wait at least one period
One can easily modify the code for alternative timeline: new consumers come->dispatch shuttle->calculate waiting cost
"""

import random
import numpy as np
import matplotlib.pyplot as plt



#define global variables
#capacity of shuttle if it is dispatched
K=15
#cost of dispatching a shuttle
c_f=100
#cost per customer left waiting per time period
c_h=2
#number of customers arriving during time interval t
A_min=1
A_max=5
#upper bound of num of people in station
p_max=200
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


#enumeration method
def enumeration(K,c_f,c_h,A_min,A_max,p_max,gamma,T,action):
    #initialize value vector V(s,a)
    V=np.zeros((p_max+1,len(action)))
    V_prev=np.zeros((p_max+1,len(action)))
    for t in range(T,-1,-1):
        print(t)
        for s in range(p_max+1):
            for a in action:
                #shuttle come and pick up consumers
                s_remaining=max(0,s-a*K)
                #new customers come
                for i in range(A_min,A_max+1):
                    #if s_remaining+i>p_max, over p_max consumers waiting, cap at p_max
                    s_remaining_new=min(p_max,s_remaining+i)
                    # reward in one period
                    reward=0-(s_remaining_new*c_h+a*c_f)
                    V[s][a]=V[s][a]+float(1/(A_max+1-A_min))*(reward+gamma*max(V_prev[s_remaining_new]))
        V_prev=V.copy()
        V=np.zeros((p_max+1,len(action)))
    return np.max(V_prev,axis=1)
results_enumeration=enumeration(K,c_f,c_h,A_min,A_max,p_max,gamma,T,action)
plt.plot(results_enumeration)
plt.title('enumeration')
plt.savefig('q1_enumeration.pdf')
plt.show()


#value iteration method
def value_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_iter,tolerance):
    # initialize value vector V(s)
    V=np.zeros(p_max+1)
    V_pre=np.zeros(p_max+1)
    cur_iter=0
    cur_error=np.abs(tolerance)*100
    while cur_iter<max_iter and cur_error>tolerance:
        print('cur_iter=',cur_iter,'cur_error=',cur_error)
        cur_iter=cur_iter+1
        for s in range(p_max+1):
            #collect the reward
            total_reward=np.zeros(len(action))
            for a in action:
                s_remaining=max(0,s-a*K)
                for i in range(A_min,A_max+1):
                    s_remaining_new=min(p_max,s_remaining+i)
                    residual_value=V_pre[s_remaining_new]
                    reward=0-(s_remaining_new*c_h+a*c_f)
                    total_reward[a]=total_reward[a]+float(1/(A_max+1-A_min))*(reward+gamma*residual_value)
            V[s]=np.max(total_reward)
        cur_error=np.linalg.norm(V-V_pre)
        V_pre=V.copy()
        V=np.zeros(p_max+1)
    return V_pre

results_value_iter=value_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_value_iter,toler_value_iter)
plt.plot(results_value_iter)
plt.title('value iteration')
plt.savefig('q1_value_iteration.pdf')
plt.show()


#policy iteration
def policy_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_iter_out,tolerance_out,max_iter_in,tolerance_in):
    # initialize value vector V(s)
    V=np.zeros(p_max+1)
    V_pre=np.zeros(p_max + 1)
    pi=np.zeros(p_max+1).astype(int)
    pi_pre=np.zeros(p_max+1).astype(int)
    cur_iter_out=0
    cur_error_out=np.abs(tolerance_out)*10
    while cur_iter_out<max_iter_out and cur_error_out>tolerance_out:
        cur_iter_out=cur_iter_out+1
        cur_iter_in = 0
        cur_error_in = np.abs(tolerance_in) * 10
        print('cur_iter=',cur_iter_out,'cur_error=',cur_error_out)
        #compute the converged value function
        while cur_iter_in<max_iter_in and  cur_error_in>tolerance_in:
            cur_iter_in=cur_iter_in+1
            for s in range(p_max+1):
                total_reward=0
                a=pi_pre[s]
                s_remaining=max(0,s-a*K)
                for i in range(A_min,A_max+1):
                    s_remaining_new=min(p_max,s_remaining+i)
                    residual_value=V_pre[s_remaining_new]
                    reward=0-(s_remaining_new*c_h+a*c_f)
                    total_reward=total_reward+float(1/(A_max+1-A_min))*(reward+gamma*residual_value)
                V[s]=total_reward
            cur_error_in=np.linalg.norm(V-V_pre)
            V_pre=V.copy()
            V=np.zeros(p_max+1)
        #compute policy function
        for s in range(p_max+1):
            total_reward=np.zeros(len(action))
            for a in action:
                s_remaining=max(0,s-a*K)
                for i in range(A_min, A_max + 1):
                    s_remaining_new=min(p_max,s_remaining+i)
                    residual_value=V_pre[s_remaining_new]
                    reward=0-(s_remaining_new*c_h+a*c_f)
                    total_reward[a]=total_reward[a]+float(1/(A_max+1-A_min))*(reward+gamma*residual_value)
            pi[s]=np.argmax(total_reward)
        cur_error_out = np.linalg.norm(pi - pi_pre)
        pi_pre=pi.copy()
        pi=np.zeros(p_max+1).astype(int)
    return pi_pre
results_policy_iter=policy_iter(K,c_f,c_h,A_min,A_max,p_max,gamma,action,max_policy_iter,toler_policy_iter,max_value_iter,toler_value_iter)
plt.plot(results_policy_iter)
plt.title('policy iteration')
plt.savefig('q1_policy_iteration.pdf')
plt.show()



print('hehe')




















