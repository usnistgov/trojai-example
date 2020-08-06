import os
import numpy as np
from utils import mad_detection


def draw_roc(out_dir, model_dir):
  lb_list = []
  sc_list = []

  files = os.listdir(out_dir)
  for fn in files:
    if not fn.endswith('.npy'):
      continue

    na = fn.split('.')[0]
    csv_fn = os.path.join(model_dir,na,'ground_truth.csv')
    with open(csv_fn,'r') as f:
      true_lb = int(f.readline())

    raw_list = np.load(os.path.join(out_dir,fn))

    score = np.min(raw_list)

    '''
    #original nc
    l1_norm_list = raw_list[0]
    crosp_lb = list(range(5))
    min_idx, a_idx = mad_detection(l1_norm_list, [0,1,2,3,4])
    '''
    '''
    # NN nc
    l1_norm_list = []
    crosp_lb = []
    for sc in range(5):
      for tg in range(5):
        if sc == tg:
          continue
        l1_norm_list.append(raw_list[sc][tg])
        crosp_lb.append('%d-%d'%(sc,tg))
    min_idx, a_idx = mad_detection(l1_norm_list, crosp_lb)
    '''

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
    draw_roc('output', '/home/tdteach/data/trojai-round0-dataset')











