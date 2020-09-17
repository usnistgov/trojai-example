import os
import csv
import numpy as np
from utils import mad_detection
import sklearn.metrics


def gen_confusion_matrix(targets, predictions):
    targets = np.asarray(targets)
    predictions = np.asarray(predictions)
    TP_counts = list()
    TN_counts = list()
    FP_counts = list()
    FN_counts = list()
    TPR = list()
    FPR = list()
    thresholds = np.arange(0.0, 1.01, 0.01)
    nb_condition_positive = np.sum(targets == 1)
    nb_condition_negative = np.sum(targets == 0)
    for t in thresholds:
        detections = predictions >= t
        # both detections and targets should be a 1d numpy array
        TP_count = np.sum(np.logical_and(detections == 1, targets == 1))
        FP_count = np.sum(np.logical_and(detections == 1, targets == 0))
        FN_count = np.sum(np.logical_and(detections == 0, targets == 1))
        TN_count = np.sum(np.logical_and(detections == 0, targets == 0))
        TP_counts.append(TP_count)
        FP_counts.append(FP_count)
        FN_counts.append(FN_count)
        TN_counts.append(TN_count)
        if nb_condition_positive > 0:
            TPR.append(TP_count / nb_condition_positive)
        else:
            TPR.append(np.nan)
        if nb_condition_negative > 0:
            FPR.append(FP_count / nb_condition_negative)
        else:
            FPR.append(np.nan)
    TP_counts = np.asarray(TP_counts).reshape(-1)
    FP_counts = np.asarray(FP_counts).reshape(-1)
    FN_counts = np.asarray(FN_counts).reshape(-1)
    TN_counts = np.asarray(TN_counts).reshape(-1)
    TPR = np.asarray(TPR).reshape(-1)
    FPR = np.asarray(FPR).reshape(-1)
    thresholds = np.asarray(thresholds).reshape(-1)
    return TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds


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
  print('totoal :', len(rst))
  return rst


def draw_roc(out_dir, gt_dict):
  lb_list = list()
  sc_list = list()
  fn_list = list()

  for row in gt_dict:
    fn = row['model_name']
    full_fn = os.path.join(out_dir,fn+'.npy')
    if not os.path.exists(full_fn):
      continue

    fn_list.append(fn)

    lb = row['poisoned']
    if lb.lower() == 'true':
      lb_list.append(1)
    else:
      lb_list.append(0)

    raw_list = np.load(full_fn)
    score = np.min(raw_list)

    if (score > 1):
        z = np.log(score)
        w = np.tanh(1.0*z-5)
        w = w/2+0.5
        w = max(0.0, min(w,1.0))
        w = w+0.01
        w = w*99.0/101.0
        score = w

    sc_list.append(score)


  from sklearn.metrics import roc_curve, auc
  import matplotlib.pyplot as plt

  print(sum(lb_list))

  print(lb_list)
  print(sc_list)

  gt_pos = 0
  gt_neg = 0
  f_neg = 0
  f_pos = 0

  fneg_list = list()
  fpos_list = list()

  my_thr = 0.5
  for fn,x,y in zip(fn_list,lb_list,sc_list):
      num = int(fn.split('-')[1])
      if x > 0.5:
          gt_pos += 1
      else:
          gt_neg += 1
      if x > 0 and y < my_thr:
          f_neg += 1
          print(fn, (x,y))
          fneg_list.append(num)
      if x == 0 and y > my_thr:
          f_pos += 1
          print(fn, (x,y))
          fpos_list.append(num)
  print(gt_pos, gt_neg)
  print('false negative rate (cover rate)', f_neg/gt_pos)
  print(fneg_list)
  print('false positive rate', f_pos/gt_neg)
  print(fpos_list)


  TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = gen_confusion_matrix(lb_list, sc_list)
  print(TP_counts)
  roc_auc = sklearn.metrics.auc(FPR,TPR)
  print('auc: ', roc_auc)

  min_rr = 10
  min_rr_tpr = None
  min_rr_fpr = None
  for f,t in zip(FPR,TPR):
      w = f+(1-t)
      if w < min_rr:
          min_rr = w
          min_rr_tpr = t
          min_rr_fpr = f
  print('min error: ({},{},{})'.format(min_rr, min_rr_fpr, min_rr_tpr))


  #plt.figure()
  #plt.plot(FPR,TPR)
  #plt.show()
  #exit(0)

  tpr, fpr, thr = roc_curve(lb_list,sc_list)
  #print(fpr)
  #print(tpr)
  #print(thr)
  print(auc(fpr,tpr))
  #plt.figure()
  #plt.plot(fpr,tpr)
  #plt.show()

if __name__ == '__main__':
    gt_csv = read_gt('/home/tdteach/data/round2-dataset-train/METADATA.csv')
    #ac_list = ['googlenet']
    ac_list = ['resnet18']
    #ac_list = ['resnet','inception','densenet']
    rst_csv = trim_gt(gt_csv, {'model_architecture':ac_list})
    #rst_csv = trim_gt(gt_csv, {})
    draw_roc('output', rst_csv)











