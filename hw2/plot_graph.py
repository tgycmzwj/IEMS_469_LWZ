import matplotlib.pyplot as plt
import numpy as np
import re
import random

#plot question 1
# cur_reward,ave_reward=[],[]
# with open('1_cartpole.txt','r') as file:
#     for line in file:
#         all_info=re.split('\t',line)
#         try:
#             cur_reward.append(np.float(all_info[2][13:]))
#             ave_reward.append(np.float(all_info[3][24:-1]))
#         except:
#             break
# num=[i+1 for i in range(len(cur_reward))]
# plt.plot(num,cur_reward,label='current reward')
# plt.plot(num,ave_reward,label='average reward')
# plt.legend(loc='lower right')
# plt.xlabel('epochs')
# plt.savefig('1_cartpole.pdf')
# plt.show()



#
# #plot question 2
# cur_reward,ave_reward=[],[]
# with open('base_new.log','r') as file:
#     for line in file:
#         all_info=re.split('(,)|(:)',line)
#         if len(all_info)>0:
#             cur_reward.append(np.float(all_info[9][19:]))
#             ave_reward.append(np.float(all_info[12][19:-1]))
# num=[i+1 for i in range(len(cur_reward))]
# plt.plot(num,cur_reward,label='current reward')
# plt.plot(num,ave_reward,label='average reward')
# plt.legend(loc='lower right')
# plt.xlabel('epochs')
# plt.savefig('2_pong.pdf')
# plt.show()
# print('finished')
