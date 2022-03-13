import os
import pickle
import numpy as np
from example_trojan_detector import get_feature
from batch_run_trojai import gt_csv
import copy

model_architecture = ['roberta-base', 'google/electra-small-discriminator', 'distilbert-base-cased']


def prepare_data():
    gt_lb = dict()
    for row in gt_csv:
        md_name = row['model_name']
        poisoned = row['poisoned']
        lb = 0
        if poisoned == 'True':
            lb = 1
        for arch, ar in enumerate(model_architecture):
            if ar == row['model_architecture']:
                break
        gt_lb[md_name] = {'lb': lb, 'arch': arch}

    pkl_fo = 'record_results'
    fns = os.listdir(pkl_fo)
    for fn in fns:
        if not fn.endswith('.pkl'): continue
        md_name = fn.split('.')[0]
        if md_name not in gt_lb: continue
        gt_lb[md_name]['rd_path'] = os.path.join(pkl_fo, fn)

    del_list = list()
    for md_name in gt_lb:
        if 'rd_path' not in gt_lb[md_name]:
            del_list.append(md_name)
    for md_name in del_list:
        del gt_lb[md_name]

    for md_name in gt_lb:
        path = gt_lb[md_name]['rd_path']

        with open(path, 'rb') as f:
            data = pickle.load(f)

        gt_lb[md_name]['raw'] = data
        feat = get_feature(data)
        gt_lb[md_name]['probs'] = feat

        print(md_name, gt_lb[md_name]['lb'], gt_lb[md_name]['probs'])

    return gt_lb


def linear_adjust(X, Y):
    lr = 0.1
    alpha = 1.0
    beta = 0.0

    sc = X
    sigmoid_sc = 1.0 / (1.0 + np.exp(-sc))
    sigmoid_sc = np.minimum(1.0 - 1e-12, np.maximum(0.0 + 1e-12, sigmoid_sc))
    loss = -(Y * np.log(sigmoid_sc) + (1 - Y) * np.log(1 - sigmoid_sc))

    print('init loss:', np.mean(loss))

    for step in range(10000):
        g_beta = sigmoid_sc - Y
        g_alpha = g_beta * X

        alpha -= lr * np.mean(g_alpha)
        beta -= lr * np.mean(g_beta)

        sc = X * alpha + beta
        sigmoid_sc = 1.0 / (1.0 + np.exp(-sc))
        sigmoid_sc = np.minimum(1.0 - 1e-12, np.maximum(0.0 + 1e-12, sigmoid_sc))
        loss = -(Y * np.log(sigmoid_sc) + (1 - Y) * np.log(1 - sigmoid_sc))

    print('loss:', np.mean(loss))
    # calc_auc(Y,sigmoid_sc)

    print(alpha, beta)

    return {'alpha': alpha, 'beta': beta}


def train_rf(gt_lb):
    if gt_lb is not None:
        X = [gt_lb[k]['probs'] for k in gt_lb]
        Y = [gt_lb[k]['lb'] for k in gt_lb]
        A = [gt_lb[k]['arch'] for k in gt_lb]
        X = np.asarray(X)
        Y = np.asarray(Y)
        A = np.asarray(A)

        A = np.expand_dims(A, axis=1)
        # X = np.concatenate([A,X],axis=1)
        # X = X[:,[0,7]]

        out_data = {'X': X, 'Y': Y, 'A': A}
        with open('train_data.pkl', 'wb') as f:
            pickle.dump(out_data, f)
        print('writing to train_data.pkl')

        print('X shape:', X.shape)
        print('Y shape:', Y.shape)
        print('A shape:', A.shape)

    else:
        rd_path = ['record_results_0/train_data_0.pkl',
                   'record_results_1/train_data_1.pkl',
                   'record_results_2/train_data_2.pkl']
        Xs, Ys, As = list(), list(), list()
        for p in rd_path:
            print(p)
            with open(p, 'rb') as f:
                data = pickle.load(f)
            print('load from ', p)
            Xs.append(data['X'])
            Ys.append(data['Y'])
            As.append(data['A'])
        X = np.concatenate(Xs)
        Y = np.concatenate(Ys)
        A = np.concatenate(As)

    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from sklearn.ensemble import RandomForestClassifier as RFC
    from sklearn.linear_model import LinearRegression as LR
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    # from mlxtend.classifier import StackingCVClassifier, StackingClassifier
    from sklearn.metrics import roc_auc_score

    from lightgbm import LGBMClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    best_auc = 0
    auc_list = list()
    rf_auc_list = list()
    best_test_acc = 0
    kf = KFold(n_splits=4, shuffle=True)

    # X = np.concatenate([X,A],axis=1)

    for train_index, test_index in kf.split(Y):

        Y_train, Y_test = Y[train_index], Y[test_index]

        # rf_clf=RFC(n_estimators=200, max_depth=11, random_state=1234)
        # rf_clf=RFC(n_estimators=200)
        # rf_clf = make_pipeline(StandardScaler(), SVC(gamma='auto',kernel='sigmoid',probability=True))
        rf_clf = LGBMClassifier(num_leaves=20)

        X_train, X_test = X[train_index], X[test_index]
        A_train, A_test = A[train_index], A[test_index]

        # X_train, Y_train = X_train[A_train==2], Y_train[A_train==2]

        rf_clf.fit(X_train, Y_train)

        preds = rf_clf.predict(X_train)
        train_acc = np.sum(preds == Y_train) / len(Y_train)
        # print(' train acc:', train_acc)

        # X_test, Y_test = X_test[A_test==2], Y_test[A_test==2]
        print('n test:', len(X_test))

        score = rf_clf.score(X_test, Y_test)
        preds = rf_clf.predict(X_test)
        probs = rf_clf.predict_proba(X_test)
        test_acc = np.sum(preds == Y_test) / len(Y_test)
        auc = roc_auc_score(Y_test, probs[:, 1])
        lr_param = linear_adjust(probs[:, 1], Y_test)
        print(' test acc: %.4f' % (test_acc), 'auc: %.4f' % (auc))

        if auc > best_auc:
            best_auc = auc
            best_clf = copy.deepcopy(rf_clf)
            best_lr_param = copy.deepcopy(lr_param)
            print('best model <------------------->')

    import joblib
    joblib.dump(best_clf, 'lgbm.joblib')
    print('dump to lgbm.joblib')

    adj_param = {'lr_param': best_lr_param, 'hash_map': global_hash_map}
    outpath = 'adj_param.pkl'
    with open(outpath, 'wb') as f:
        pickle.dump(adj_param, f)
    print('dump to', outpath)

    '''
  rf_clf=LGBMClassifier(num_leaves=100)
  rf_clf.fit(X,Y)
  score=rf_clf.score(X, Y)
  preds=rf_clf.predict(X)
  probs=rf_clf.predict_proba(X)
  test_acc=np.sum(preds==Y)/len(Y)
  auc=roc_auc_score(Y, probs[:,1])
  linear_adjust(probs[:,1], Y)
  print(' train on all cc:', test_acc, 'auc:',auc)
  '''


if __name__ == '__main__':
    gt_lb = prepare_data()
    train_rf(gt_lb)

    # train_rf(None)
