import os
import csv
import numpy as np
import utils
import sklearn.metrics
import pickle


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


def trim_gt(gt_dict, t_dict):
  rst = list()
  for md_name in gt_dict:
    row=gt_dict[md_name]
    ok = True
    for key in t_dict:
      value=t_dict[key]
      if type(value) is list:
        if row[key] not in value: ok=False
      elif value != row[key]:
        ok=False

      if not ok: break

    if ok:
      rst.append(row)

  rst_dict=dict()
  for row in rst:
    rst_dict[row['model_name']]=row
  return rst_dict



def _deal_rst_data(data):
    rst, ch_rst=data['rst'], data['ch_rst']
    max_acc, max_ch_acc=0,0
    for key in rst:
        max_acc=max(max_acc,rst[key]['acc'])
        max_ch_acc=max(max_ch_acc,ch_rst[key]['acc'])
    return max(max_acc, max_ch_acc)


def deal_data(data):
  return _deal_rst_data(data)



def linear_adjust(lb_list, sc_list):
  import torch
  from torch.autograd import Variable

  a=np.ones(1)
  b=np.zeros(1)
  va=Variable(torch.from_numpy(a), requires_grad=True)
  vb=Variable(torch.from_numpy(b), requires_grad=True)
  opt = torch.optim.Adam([va,vb],lr=0.1)

  lb_list=np.asarray(lb_list)
  lb=torch.from_numpy(lb_list)

  sc_list=np.asarray(sc_list)
  sc_list=sc_list*2-1
  sc_list=np.maximum(sc_list,-1+1e-12)
  sc_list=np.minimum(sc_list,+1-1e-12)
  sc_list=np.arctanh(sc_list)
  sc=torch.from_numpy(sc_list)

  max_step=500
  for step in range(max_step):
    #if step%100==0: print(step)
    cg=torch.tanh(sc*va+vb)
    cg=cg/2+0.5
    s = torch.log(cg)*lb+(1-lb)*torch.log(1-cg)
    loss = -torch.mean(s)

    opt.zero_grad()
    loss.backward()
    opt.step()
  print(loss, va, vb)


def draw_roc(out_dir, gt_dict,suffix):

  suffix=suffix+'.pkl'

  fns=os.listdir(out_dir)
  rst_fns=list()
  for fn in fns:
    md_name=fn.split('_')[0]
    if fn != md_name+'_'+suffix: continue

    if not md_name in gt_dict: continue
    rst_fns.append(fn)
  rst_fns.sort()

  print('total :', len(rst_fns))

  rst_dict=dict()
  for fn in rst_fns:
    md_name = fn.split('_')[0]

    lb = gt_dict[md_name]['poisoned']
    if lb.lower() == 'true':
      lb=1
    else:
      lb=0

    full_fn=os.path.join(out_dir,fn)
    with open(full_fn,'rb') as f:
      data_dict=pickle.load(f)

    rst_dict[md_name]=dict()
    rst_dict[md_name]['label']=lb
    rst_dict[md_name]['data']=data_dict

  lb_list = list()
  sc_list = list()
  fn_list = list()
  for md_name in rst_dict:
    fn_list.append(md_name)
    lb_list.append(rst_dict[md_name]['label'])
    sc_list.append(deal_data(rst_dict[md_name]['data']))

  lb_list = np.asarray(lb_list)
  sc_list = np.asarray(sc_list)

  from sklearn.metrics import roc_curve, auc
  import matplotlib.pyplot as plt

  print('total positive:', sum(lb_list))

  '''
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
  '''


  TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = gen_confusion_matrix(lb_list, sc_list)
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


  '''
  plt.figure()
  plt.plot(FPR,TPR)
  plt.show()
  exit(0)
  #'''

  #for fn,lb,sc in zip(fn_list,lb_list,sc_list):
  #  print(fn,lb,sc)

  tpr, fpr, thr = roc_curve(lb_list,sc_list)
  #print(fpr)
  #print(tpr)
  #print(thr)
  print(auc(fpr,tpr))
  print(len(lb_list))
  plt.figure()
  plt.plot(fpr,tpr)
  plt.show()


  #'''
  k=0
  for md_name,sc,lb in zip(rst_dict.keys(), sc_list, lb_list):
    if sc >= 0.0:
      print(k,md_name, sc, lb)
      k+=1
  #'''

  sc_list=linear_adjust(lb_list,sc_list)

if __name__ == '__main__':
    home = os.environ['HOME']
    csv_path = os.path.join(home,'data/round7-train-dataset/METADATA.csv')
    gt_dict = utils.read_gt_csv(csv_path)
    filter_dict=dict()
    #filter_dict['model_architecture']=['GruLinear','LstmLinear']
    #filter_dict['model_architecture']=['FCLinear']
    #filter_dict['embedding']=['GPT-2']

    rst_dict = trim_gt(gt_dict, filter_dict)
    draw_roc('scratch', rst_dict, suffix='rst')











