import sklearn.metrics
import matplotlib.pyplot as plt


fpr = list()
tpr = list()
with open('roc_results.txt','r') as f:
    for l in f:
        rst = l.split()
        if (len(rst) == 2):
            a, b = rst
            tpr.append(float(a))
            fpr.append(float(b))

print(tpr)
print(fpr)

roc_auc = sklearn.metrics.auc(fpr,tpr)
print(roc_auc)

plt.figure()
plt.plot(fpr,tpr)
plt.show()


