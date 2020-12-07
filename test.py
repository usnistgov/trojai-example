import numpy as np
import pickle

md_name = 'id-00000258'

data_path = md_name+'_mean_channel_max.pkl'
with open(data_path,'rb') as f:
    benign = pickle.load(f)
data_path = md_name+'_poisoned_mean_channel_max.pkl'
with open(data_path,'rb') as f:
    poison = pickle.load(f)


rst=list()
n_conv = len(benign)
cnt = [0]*n_conv
for k in range(n_conv):
    n_chann = len(benign[k])
    for i in range(n_chann):
        if (poison[k][i] > benign[k][i]):
            cnt[k] += 1
            rst.append((k,i,benign[k][i],poison[k][i]))
            #print(k,i, 'b:',benign[k][i], 'p:',poison[k][i])

#rst.sort(key=lambda t: t[3]-t[2],reverse=True)
#rst.sort(key=lambda t: t[3]/t[2],reverse=True)
#rst.sort(key=lambda t: t[3],reverse=True)
for t in rst:
    print(t[0],t[1],'b:',t[2],'p:',t[3],'ratio=',t[3]/t[2], 'diff=',t[3]-t[2])

for k,ct in enumerate(cnt):
    print(k,ct,len(benign[k]))
