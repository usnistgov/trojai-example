import os
import pickle
import numpy as np
from batch_run import gt_csv


def prepare_data():
    gt_lb = dict()
    for row in gt_csv:
        md_name = row['model_name']
        poisoned = row['poisoned']
        lb = 0
        if poisoned=='True': 
            lb = 1
        gt_lb[md_name] = {'lb': lb}

    pkl_fo = 'record_results'
    fns = os.listdir(pkl_fo)
    for fn in fns:
        if not fn.endswith('.pkl'): continue
        md_name = fn.split('.')[0]
        if md_name not in gt_lb: continue
        gt_lb[md_name]['rd_path']=os.path.join(pkl_fo,fn)

    del_list = list()
    for md_name in gt_lb:
        if 'rd_path' not in gt_lb[md_name]:
            del_list.append(md_name)
    for md_name in del_list:
        del gt_lb[md_name]

    for md_name in gt_lb:
        path = gt_lb[md_name]['rd_path']

        with open(path,'rb') as f:
            data = pickle.load(f)

        gt_lb[md_name]['raw']=data
        data_keys = list(data.keys())
        data_keys.sort()
        a = [data[k] for k in data_keys]
        b = a.copy()
        b.append(np.max(a))
        b.append(np.mean(a))
        b.append(np.std(a))
        gt_lb[md_name]['probs'] = np.asarray(b)

        print(gt_lb[md_name]['lb'], gt_lb[md_name]['probs'])

    return gt_lb



def train_rf(gt_lb):
  X = [gt_lb[k]['probs'] for k in gt_lb]
  Y = [gt_lb[k]['lb'] for k in gt_lb]
  X = np.asarray(X)
  Y = np.asarray(Y)


  from sklearn.model_selection import KFold
  from sklearn.model_selection import cross_val_score
  from sklearn.ensemble import RandomForestClassifier as RFC
  from sklearn.linear_model import LinearRegression as LR
  from sklearn.linear_model import LogisticRegression
  from sklearn.pipeline import make_pipeline
  from mlxtend.classifier import StackingCVClassifier, StackingClassifier
  from sklearn.metrics import roc_auc_score

  from lightgbm import LGBMClassifier

  auc_list=list()
  rf_auc_list=list()
  best_test_acc=0
  kf=KFold(n_splits=5,shuffle=True)
  for train_index, test_index in kf.split(Y):

    Y_train, Y_test = Y[train_index], Y[test_index]

    #rf_clf=RFC(n_estimators=200, max_depth=11, random_state=1234)
    #rf_clf=RFC(n_estimators=200)
    rf_clf=LGBMClassifier(num_leaves=100)

    X_train, X_test = X[train_index], X[test_index]
    rf_clf.fit(X_train,Y_train)

    preds=rf_clf.predict(X_train)
    train_acc=np.sum(preds==Y_train)/len(Y_train)
    print(' train acc:', train_acc)

    score=rf_clf.score(X_test, Y_test)
    preds=rf_clf.predict(X_test)
    probs=rf_clf.predict_proba(X_test)
    test_acc=np.sum(preds==Y_test)/len(Y_test)
    auc=roc_auc_score(Y_test, probs[:,1])
    print(' test acc:', test_acc, 'auc:',auc)


    
  rf_clf=LGBMClassifier(num_leaves=100)
  rf_clf.fit(X,Y)

  import joblib
  joblib.dump(rf_clf, 'lgbm.joblib')

  score=rf_clf.score(X, Y)
  preds=rf_clf.predict(X)
  probs=rf_clf.predict_proba(X)
  test_acc=np.sum(preds==Y)/len(Y)
  auc=roc_auc_score(Y, probs[:,1])
  print(' train on all cc:', test_acc, 'auc:',auc)


if __name__ == '__main__':
    gt_lb = prepare_data()
    train_rf(gt_lb)

