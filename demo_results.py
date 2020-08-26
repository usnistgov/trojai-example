import os
import numpy as np
from utils import mad_detection

def read_gt(filepath):
  rst = dict()
  with open(filepath,'r') as f:
    for l in f:
      sl = l.split()
      if len(sl) == 3:
        rst[sl[0]] = (sl[1],sl[2])
  return rst


def draw_roc(out_dir, gt_dict):
  lb_list = []
  sc_list = []


  files = os.listdir(out_dir)
  for fn in files:
    if not fn.endswith('.npy'):
      continue

    na = fn.split('.')[0]
    if na not in gt_dict:
      continue

    if gt_dict[na][0].lower() == 'true':
      true_lb = 1
    else:
      true_lb = 0

    raw_list = np.load(os.path.join(out_dir,fn))
    score = np.min(raw_list)

    lb_list.append(true_lb)
    sc_list.append(score)


  from sklearn.metrics import roc_curve, auc
  import matplotlib.pyplot as plt

  print(sum(lb_list))

  tpr, fpr, thr = roc_curve(lb_list,sc_list)
  print(fpr)
  print(tpr)
  print(thr)
  print(auc(fpr,tpr))
  plt.figure()
  plt.plot(fpr,tpr)
  plt.show()

if __name__ == '__main__':
    rst = read_gt('/home/tdteach/data/trojai-round0-dataset/round0_train_gt.txt')
    nrst = dict()
    for key in rst:
      nrst[key] = rst[key]
      #if 'resnet' in rst[key][1]:
      #  nrst[key] = rst[key]
    draw_roc('output', nrst)











