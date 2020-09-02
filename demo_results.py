import os
import csv
import numpy as np
from utils import mad_detection

def read_gt(filepath):
  rst = list()
  with open(filepath,'r',newline='') as csvfile:
    csvreader = csv.DictReader(csvfile)
    for row in csvreader:
      rst.append(row)
  return rst


def trim_gt(gt_csv, t_dict):
  rst = list()
  for row in gt_csv:
    ok = True
    for key in t_dict:
      ok = False
      for d in t_dict[key]:
        if d in row[key]:
          ok = True
      if not ok:
        break
    if ok:
      rst.append(row)
  print(rst[0])
  return rst


def draw_roc(out_dir, gt_dict):
  lb_list = list()
  sc_list = list()

  for row in gt_dict:
    fn = row['model_name']
    full_fn = os.path.join(out_dir,fn+'.npy')
    if not os.path.exists(full_fn):
      continue

    print(fn)

    lb = row['poisoned']
    if lb.lower() == 'true':
      lb_list.append(1)
    else:
      lb_list.append(0)

    raw_list = np.load(full_fn)
    score = np.min(raw_list)
    sc_list.append(score)


  from sklearn.metrics import roc_curve, auc
  import matplotlib.pyplot as plt

  print(sum(lb_list))

  print(lb_list)
  print(sc_list)

  gt_pos = 0
  cover_loss = 0

  for x,y in zip(lb_list,sc_list):
      #if x == 0 and y > 0:
      #    print((x,y))
      if x > 0:
          gt_pos += 1
          if y < 1-1e-3:
              cover_loss += 1
              print((x,y))
  print(cover_loss/gt_pos)


  tpr, fpr, thr = roc_curve(lb_list,sc_list)
  print(fpr)
  print(tpr)
  print(thr)
  print(auc(fpr,tpr))
  plt.figure()
  plt.plot(fpr,tpr)
  plt.show()

if __name__ == '__main__':
    gt_csv = read_gt('/home/tdteach/data/round2-dataset-train/METADATA.csv')
    ac_list = ['resnet','inception','densenet']
    rst_csv = trim_gt(gt_csv, {'model_architecture':ac_list})
    draw_roc('output', rst_csv)











