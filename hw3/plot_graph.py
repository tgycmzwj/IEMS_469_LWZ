import matplotlib.pyplot as plt
import numpy as np
import re
import random
import sys

# plot question 1
# length,frame,cur_reward,ave_reward,cur_maxq,mean_maxq=[],[],[],[],[],[]
# with open('/Users/lesliezhao/Dropbox/nu_course/2021_fall/DP/IEMS469_HW_LWZ/hw3/cartpole_training.txt','r') as file:
#     for line in file:
#         all_info=re.split(' ',line)
#         try:
#             length.append(int(all_info[4]))
#             frame.append(int(all_info[6]))
#             cur_reward.append(np.float(all_info[8]))
#             ave_reward.append(np.float(all_info[10]))
#             cur_maxq.append(np.float(all_info[12]))
#             mean_maxq.append(np.float(all_info[14]))
#         except:
#             continue
# num=[i+1 for i in range(len(cur_reward))]
# plt.plot(num,cur_maxq,label='current maxq')
# plt.plot(num,mean_maxq,label='average maxq')
# plt.legend(loc='lower right')
# plt.xlabel('episode')
# plt.savefig('1_cartpole_qvalue.pdf')
# plt.show()

# #
# length,frame,cur_reward,ave_reward,cur_maxq,mean_maxq=[],[],[],[],[],[]
# with open('/Users/lesliezhao/Dropbox/nu_course/2021_fall/DP/IEMS469_HW_LWZ/hw3/cartpole_testing.txt','r') as file:
#     for line in file:
#         all_info=re.split(' ',line)
#         try:
#             length.append(int(all_info[4]))
#             frame.append(int(all_info[6]))
#             cur_reward.append(np.float(all_info[8]))
#             ave_reward.append(np.float(all_info[10]))
#             cur_maxq.append(np.float(all_info[12]))
#             mean_maxq.append(np.float(all_info[14]))
#         except:
#             continue
# num=[i+1 for i in range(len(cur_reward))]
# plt.plot(num,cur_reward,label='current maxq')
# plt.plot(num,ave_reward,label='average maxq')
# plt.legend(loc='lower right')
# plt.xlabel('episode')
# plt.savefig('1_cartpole_reward.pdf')
# plt.show()


#plot for quesiton 2
# length,cur_rew,mean_rew,cur_maxq,mean_maxq=[],[],[],[],[]
# with open('/Users/lesliezhao/Dropbox/nu_course/2021_fall/DP/IEMS469_HW_LWZ/hw3/pacman_training.txt','r') as file:
#     for line in file:
#         try:
#             all_info=[item for item in re.split('(episode)|(length)|(frame)|(cur_rew)|(mean_rew)|(cur_maxq)|(mean_maxq)|(eps)',line) if item]
#             length.append(int(all_info[4]))
#             cur_rew.append(int(float(all_info[8].strip())))
#             mean_rew.append(float(all_info[10].strip()))
#             cur_maxq.append(float(all_info[12].strip()))
#             mean_maxq.append(float(all_info[14].strip()))
#         except:
#             continue
# num=[i+1 for i in range(len(cur_rew))]
# plt.plot(num,cur_maxq,label='current maxq')
# plt.plot(num,mean_maxq,label='mean maxq')
# plt.legend(loc='lower right')
# plt.xlabel('episode')
# plt.savefig('2_pacman_q.pdf')
# plt.show()



ave_reward=[]
mean_reward=[]
length,frame,cur_reward,ave_reward,cur_maxq,mean_maxq=[],[],[],[],[],[]
with open('/Users/lesliezhao/Dropbox/nu_course/2021_fall/DP/IEMS469_HW_LWZ/hw3/pacman_testing.txt','r') as file:
    for line in file:
        try:
            all_info=[item for item in re.split('(episode)|(length)|(frame)|(cur_rew)|(mean_rew)|(cur_maxq)|(mean_maxq)|(eps)',line) if item]
            length.append(int(all_info[4]))
            cur_reward.append(int(float(all_info[8].strip())))
            mean_reward.append(float(all_info[10].strip()))
        except:
            continue
num=[i+1 for i in range(len(cur_reward))]

plt.plot(num,cur_reward,label='current reward')
plt.plot(num,mean_reward,label='mean reward')
plt.legend(loc='lower right')
plt.xlabel('episode')
plt.savefig('2_pacman_reward.pdf')
plt.show()

print('finished')