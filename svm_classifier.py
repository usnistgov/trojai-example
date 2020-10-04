import sklearn
from demo_results import gen_confusion_matrix
from sklearn.svm import LinearSVC
import pickle
import csv
import utils
import os
import numpy as np
import pickle

from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


svm_folder = 'svm_models'

def get_data(arch):
    home = os.environ['HOME']
    gt_csv = utils.read_gt_csv(os.path.join(home,'data/round2-dataset-train/METADATA.csv'))

    scratch_folder = 'scratch'

    data_list = list()
    na_list = list()

    for row in gt_csv:
        if not row['model_architecture'] == arch:
            continue

        md_name = row['model_name']
        full_path = os.path.join(scratch_folder,md_name+'.pkl')
        #print(md_name)

        if not os.path.exists(full_path):
            continue

        with open(full_path,'rb') as f:
            data = pickle.load(f)

        if row['poisoned'] == 'True':
            lb = 1
        else:
            lb = 0

        data_list.append((lb, data))
        na_list.append(row['model_name'])
        #print(md_name, len(data['candi_ct']))


    X = list()
    Y = list()

    min_fts = -1
    for d in data_list:
        candi_ct= d[1]['candi_ct']
        if min_fts < 0 or len(candi_ct) < min_fts:
            min_fts = len(candi_ct)

    for d in data_list:
        candi_ct= d[1]['candi_ct']
        candi_ct = candi_ct[:min_fts]
        X.append(candi_ct)
        Y.append(d[0])

    X = np.asarray(X)
    Y = np.asarray(Y)


    for na,x,y in zip(na_list,X,Y):
        if na=='id-00000035':
            print(na,x,y)
    exit(0)

    return X,Y,na_list


def split_training_set(X,Y):
    nX = len(X)
    id_list = list(range(nX))
    id_list = np.asarray(id_list)

    pos_idx = Y>0.5
    neg_idx = np.logical_not(pos_idx)

    pos_list = id_list[pos_idx]
    neg_list = id_list[neg_idx]
    np.random.shuffle(pos_list)
    np.random.shuffle(neg_list)

    n_tr_pos = int(0.8*len(pos_list))
    n_tr_neg = int(0.8*len(neg_list))
    tr_pos_list = pos_list[:n_tr_pos]
    tr_neg_list = neg_list[:n_tr_neg]
    te_pos_list = pos_list[n_tr_pos:]
    te_neg_list = neg_list[n_tr_neg:]

    tr_idx = np.concatenate([tr_pos_list,tr_neg_list])
    te_idx = np.concatenate([te_pos_list,te_neg_list])

    tr_X = np.copy(X[tr_idx])
    tr_Y = np.copy(Y[tr_idx])
    te_X = np.copy(X[te_idx])
    te_Y = np.copy(Y[te_idx])

    return tr_X, tr_Y, te_X, te_Y


def cross_validate(X,Y, ft_idx, split=5):
    nX = len(X)
    id_list = list(range(nX))
    id_list = np.asarray(id_list)
    np.random.shuffle(id_list)
    XX = X[id_list]
    YY = Y[id_list]

    n_tr = nX-int(nX/split)
    rd_idx = list()
    for i in range(n_tr, nX):
        rd_idx.append(i)
    for i in range(n_tr):
        rd_idx.append(i)
    rd_idx = np.asarray(rd_idx)

    te_acc_list = list()
    for z in range(split):
        clf = LinearSVC()
        clf.max_iter = 1000000
        clf.fit(XX[:n_tr,ft_idx],YY[:n_tr])
        te_acc = clf.score(XX[n_tr:,ft_idx],YY[n_tr:])
        te_acc_list.append(te_acc)

        XX = XX[rd_idx]
        YY = YY[rd_idx]

    return te_acc_list




@ignore_warnings(category=ConvergenceWarning)
def try_svm(arch):
    X, Y, _ = get_data(arch)
    print(X.shape)

    candi_ft = list()
    col_sum = np.sum(X,0)
    for j in range(len(col_sum)):
        if col_sum[j] > 0:
            candi_ft.append(j)
    candi_ft = np.asarray(candi_ft,dtype=np.int32)

    tr_X, tr_Y, te_X, te_Y = split_training_set(X,Y)

    def _next_fts(cur, n, m):
        st_k = -1
        for i in range(m):
            if cur[m-1-i] < n-1-i:
                st_k = m-1-i
                break
        if st_k < 0:
            return None
        cur[st_k] += 1
        for i in range(st_k+1, m):
            cur[i] = cur[i-1]+1
        return cur

    m_ft = len(candi_ft)
    best_all_acc = 0
    best_ft_idx = None
    has_found = False

    '''
    for ft_len in range(4, m_ft):
        candi_ft_idx = list(range(ft_len))

        while candi_ft_idx is not None:
            ft_idx = candi_ft[candi_ft_idx]

            clf = LinearSVC()
            XX = X[:,ft_idx]
            clf.fit(XX,Y)
            all_acc = clf.score(XX,Y)

            if all_acc > best_all_acc:
                best_all_acc = all_acc
                print(XX[0])
                print(np.sum(XX))
                print(ft_idx,all_acc)
                print(clf.coef_[0], clf.intercept_)
                print(clf)

                zz=np.matmul(XX,clf.coef_[0])+clf.intercept_
                ans = 0
                for j in range(len(zz)):
                    if zz[j] > 0 and Y[j] > 0.5:
                        ans += 1
                    elif zz[j] < 0 and Y[j] < 0.5:
                        ans += 1
                print(ans/len(zz))


            if all_acc > 0.8:
                clf = LinearSVC()
                clf.max_iter = 1000000
                XX = tr_X[:,ft_idx]
                clf.fit(XX,tr_Y)
                tr_acc = clf.score(XX,tr_Y)

                XXX = te_X[:,ft_idx]
                te_acc = clf.score(XXX,te_Y)

                print(tr_acc,te_acc,ft_idx,all_acc)

                if tr_acc > 0.8 and te_acc > 0.8:
                    print(tr_acc, te_acc, ft_idx)
                    has_found = True
                    break

            candi_ft_idx = _next_fts(candi_ft_idx,m_ft,ft_len)

        if has_found:
            break

    exit(0)
    #'''


    best_all_acc = 0
    best_all_idx = None
    step = 0

    while best_all_acc<0.7 and step<1000:
      step += 1
      best_te_acc = 0
      best_ft_idx = None

      ft_idx = candi_ft
      while True:

        coef_list = list()
        te_acc_list = list()
        for _ in range(5):
          tr_X, tr_Y, te_X, te_Y = split_training_set(X,Y)

          clf = LinearSVC()
          clf.max_iter = 1000000
          clf.fit(tr_X[:,ft_idx],tr_Y)
          te_acc = clf.score(te_X[:,ft_idx],te_Y)

          coef_list.append(clf.coef_[0])
          te_acc_list.append(te_acc)

        coef_mat = np.asarray(coef_list)
        coef_mat = np.abs(coef_mat)
        coef_sum = np.sum(coef_mat, axis=1, keepdims=True)
        coef_mat = coef_mat/coef_sum
        coef_wet = np.sum(coef_mat,axis=0)

        coef_ord = np.argsort(coef_wet)
        coef_ord = np.flip(coef_ord)
        coef_sum = np.sum(coef_wet)
        cur_s = 0
        new_ft_idx = list()
        for k in coef_ord:
            new_ft_idx.append(k)
            cur_s += coef_wet[k]
            if cur_s/coef_sum > 0.9:
                break
        new_ft_idx = np.asarray(new_ft_idx)
        new_ft_idx = ft_idx[new_ft_idx]
        new_ft_idx = np.sort(new_ft_idx)

        if len(new_ft_idx) == len(ft_idx):
            break

        acc_list = cross_validate(X,Y,new_ft_idx,split=4)
        te_acc = np.min(acc_list)

        if te_acc > best_te_acc:
            best_te_acc = te_acc
            best_ft_idx = new_ft_idx.copy()

        ft_idx = new_ft_idx

      print(ft_idx, acc_list)
      if best_te_acc > best_all_acc:
        best_all_acc = best_te_acc
        print('update', best_all_acc)
        best_all_idx = best_ft_idx.copy()
        print(best_all_acc, best_all_idx)
      elif best_te_acc > best_all_acc-(1e-9) and len(best_all_idx) > len(best_ft_idx):
          best_all_idx = best_ft_idx.copy()
          print('update', best_all_acc, best_all_idx)

    return best_all_acc, best_all_idx



def save_svm_model(arch, ft_idx):
    X, Y, _ = get_data(arch)

    clf = LinearSVC()

    XX = X[:,ft_idx]
    print(XX[0])
    print(np.sum(XX))
    clf.fit(XX,Y)
    acc = clf.score(XX,Y)
    print('all acc: ', acc)


    model = {'layer_candi':ft_idx, 'svm_model':clf}
    save_name = os.path.join(svm_folder,arch+'_svm.pkl')
    print('save to '+save_name)
    if os.path.exists(save_name):
        os.remove(save_name)
        print('overwrite the old model')
    with open(save_name,'wb') as f:
        pickle.dump(model,f)

    model = None
    with open(save_name,'rb') as f:
        model = pickle.load(f)
    clf = model['svm_model']
    ft_idx = model['layer_candi']

    acc = clf.score(X[:,ft_idx],Y)
    print('loading acc: ', acc)


    coef = clf.coef_[0]
    sc = np.matmul(XX, coef)+clf.intercept_
    print(ft_idx)
    print(coef, clf.intercept_)


def calc_auc(y,x):
    TP_counts, FP_counts, FN_counts, TN_counts, TPR, FPR, thresholds = gen_confusion_matrix(y,x)
    roc_auc = sklearn.metrics.auc(FPR,TPR)
    print('auc: ', roc_auc)


def build_loss_model(arch_lsit):
    all_x = list()
    all_y = list()
    for arch in arch_list:
        print(arch)
        X, Y, _ = get_data(arch)

        model_path = os.path.join('svm_models',arch+'_svm.pkl')
        with open(model_path,'rb') as f:
            model = pickle.load(f)
        clf = model['svm_model']
        ft_idx = model['layer_candi']

        coef = clf.coef_[0]
        intr = clf.intercept_

        print(ft_idx)
        XX = X[:,ft_idx]
        sc = np.matmul(XX,coef)+intr

        all_x.append(sc)
        all_y.append(Y)


    all_coef = {}
    for arch,X,Y in zip(arch_list, all_x, all_y):
        lr = 0.1
        alpha = 1.0
        beta = 0.0

        sc = X
        sigmoid_sc = 1.0/(1.0+np.exp(-sc))
        sigmoid_sc = np.minimum(1.0-1e-12, np.maximum(0.0+1e-12, sigmoid_sc))
        loss = -(Y*np.log(sigmoid_sc)+(1-Y)*np.log(1-sigmoid_sc))

        print(arch)
        print('init loss:', np.mean(loss))


        for step in range(10000):
            g_beta = sigmoid_sc-Y
            g_alpha = g_beta*X

            alpha -= lr*np.mean(g_alpha)
            beta -= lr*np.mean(g_beta)

            sc = X*alpha+beta
            sigmoid_sc = 1.0/(1.0+np.exp(-sc))
            sigmoid_sc = np.minimum(1.0-1e-12, np.maximum(0.0+1e-12, sigmoid_sc))
            loss = -(Y*np.log(sigmoid_sc)+(1-Y)*np.log(1-sigmoid_sc))


        print('loss:', np.mean(loss))
        calc_auc(Y,sigmoid_sc)

        all_coef[arch] = (alpha,beta)


    print(all_coef)


    all_loss = list()
    all_sc = list()
    for arch, X,Y in zip(arch_list, all_x,all_y):
        alpha, beta = all_coef[arch]
        sc = X*alpha+beta
        sigmoid_sc = 1.0/(1.0+np.exp(-sc))
        sigmoid_sc = np.minimum(1.0-1e-12, np.maximum(0.0+1e-12, sigmoid_sc))
        loss = -(Y*np.log(sigmoid_sc)+(1-Y)*np.log(1-sigmoid_sc))

        all_loss.append(loss)
        all_sc.append(sigmoid_sc)

    all_loss = np.concatenate(all_loss)
    all_y = np.concatenate(all_y)
    all_sc = np.concatenate(all_sc)
    print('all loss:', np.mean(all_loss))
    calc_auc(all_y,all_sc)


    model_path = os.path.join(svm_folder,'loss_model.pkl')
    with open(model_path,'wb') as f:
        pickle.dump(all_coef, f)






if __name__ == '__main__':
    #'''
    arch = 'resnet18'
    print(arch)
    #ft_idx = [19,20,85,86,89,91]
    acc, ft_idx = try_svm(arch)
    save_svm_model(arch,ft_idx)
    exit(0)
    #'''

    arch_list = ['resnet18','resnet34','resnet50','resnet101', 'resnet152','wideresnet50','wideresnet101','densenet121','densenet161','densenet169','densenet201','googlenet','inceptionv3','squeezenetv1_0','squeezenetv1_1','mobilenetv2','shufflenet1_0','shufflenet1_5','shufflenet2_0','vgg11bn','vgg13bn','vgg16bn','vgg19bn']
    build_loss_model(arch_list)











