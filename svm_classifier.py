import sklearn
from demo_results import gen_confusion_matrix
from sklearn.svm import LinearSVC
import pickle
import csv
import utils
import os
import numpy as np
import pickle

svm_folder = 'svm_models'

def get_data(arch):
    home = os.environ['HOME']
    gt_csv = utils.read_gt_csv(os.path.join(home,'data/round2-dataset-train/METADATA.csv'))

    scratch_folder = 'scratch'

    data_list = list()

    for row in gt_csv:
        if not row['model_architecture'] == arch:
            continue

        md_name = row['model_name']
        full_path = os.path.join(scratch_folder,md_name+'.pkl')

        if not os.path.exists(full_path):
            continue

        with open(full_path,'rb') as f:
            data = pickle.load(f)
        if row['poisoned'] == 'True':
            lb = 1
        else:
            lb = 0
        data_list.append((lb, data))


    X = list()
    Y = list()

    for d in data_list:
        X.append(d[1]['candi_ct'])
        Y.append(d[0])
    X = np.asarray(X)
    Y = np.asarray(Y)

    return X,Y


def try_svm(arch):
    X, Y = get_data(arch)

    nX = len(X)
    split_p = int(nX*0.8)
    print(nX)


    best_all_acc = 0
    best_ft_idx = None

    while best_all_acc < 0.8:
      best_te_acc = 0

      while best_te_acc < 0.8:
        ch_idx = np.random.rand(nX)>0.2
        rv_idx = np.logical_not(ch_idx)
        tr_X = np.copy(X[ch_idx])
        tr_Y = np.copy(Y[ch_idx])
        te_X = np.copy(X[rv_idx])
        te_Y = np.copy(Y[rv_idx])

        XX = tr_X
        ft_idx = list(range(tr_X.shape[1]))
        while True:
            clf = LinearSVC()
            clf.max_iter = 1000000
    
            XX = tr_X[:,ft_idx]
            clf.fit(XX,tr_Y)
            tr_acc = clf.score(XX,tr_Y)

            XXX = te_X[:,ft_idx]
            te_acc = clf.score(XXX,te_Y)


            coef = clf.coef_[0]
            #print(tr_acc, te_acc, ft_idx, len(XX), len(XXX))
            #print(coef)


            coef = abs(coef)
            sorted_idx = np.argsort(coef)
            sorted_idx = np.flip(sorted_idx,0)
    
            new_ft_idx = list()
            ct_sum = 0
            wt_abs_sum = np.sum(coef)
            for i in sorted_idx:
                ct_sum += coef[i]
                new_ft_idx.append(ft_idx[i])
                if ct_sum/wt_abs_sum > 0.9:
                    break

            if len(new_ft_idx) >= len(ft_idx):
                break

            ft_idx = new_ft_idx
            ft_idx = np.sort(ft_idx)

        best_te_acc = te_acc

      clf = LinearSVC()
      clf.max_iter = 1000000
      clf.fit(X[:,ft_idx],Y)
      all_acc = clf.score(X[:,ft_idx],Y)
      #print('all_acc', all_acc)

      if all_acc > best_all_acc:
          best_all_acc = all_acc
          best_ft_idx = ft_idx
          print(tr_acc, te_acc, ft_idx, all_acc)

    return best_all_acc, best_ft_idx


def save_svm_model(arch, ft_idx):
    X, Y = get_data(arch)

    clf = LinearSVC()
    clf.max_iter = 1000000

    XX = X[:,ft_idx]
    clf.fit(XX,Y)
    acc = clf.score(XX,Y)
    print('all acc: ', acc)

    model = {'layer_candi':ft_idx, 'svm_model':clf}
    save_name = os.path.join(svm_folder,arch+'_svm.pkl')
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
        X, Y = get_data(arch)
        
        model_path = os.path.join('svm_models',arch+'_svm.pkl')
        with open(model_path,'rb') as f:
            model = pickle.load(f)
        clf = model['svm_model']
        ft_idx = model['layer_candi']

        coef = clf.coef_[0]
        intr = clf.intercept_
        
        XX = X[:,ft_idx]
        sc = np.matmul(XX,coef)+intr

        all_x.append(sc)
        all_y.append(Y)


    all_coef = {}
    for arch,X,Y in zip(arch_list, all_x, all_y):
        lr = 0.1
        alpha = 1.0
        beta = 0.0

        for step in range(10000):
            sc = X*alpha+beta
            sigmoid_sc = 1.0/(1.0+np.exp(-sc))

            sigmoid_sc = np.minimum(1.0-1e-12, np.maximum(0.0+1e-12, sigmoid_sc))

            loss = -(Y*np.log(sigmoid_sc)+(1-Y)*np.log(1-sigmoid_sc))

            g_beta = sigmoid_sc-Y
            g_alpha = g_beta*X

            alpha -= lr*np.mean(g_alpha)
            beta -= lr*np.mean(g_beta)

        print(arch)
        print('loss:', np.mean(loss))
        calc_auc(Y,sigmoid_sc)

        all_coef[arch] = (alpha,beta)


    print(all_coef)


    all_loss = list()
    all_sc = list()
    for X,Y in zip(all_x,all_y):
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
    arch = 'densenet161'
    #ft_idx = [21,23,40,54,56,72,78]
    acc, ft_idx = try_svm(arch)
    save_svm_model(arch,ft_idx)
    exit(0)
    #'''

    arch_list = ['resnet18','resnet34','resnet50','googlenet','inceptionv3','squeezenetv1_0','squeezenetv1_1','mobilenetv2','shufflenet1_0','shufflenet1_5','shufflenet2_0','vgg11bn','vgg13bn','vgg16bn','vgg19bn','densenet121']
    build_loss_model(arch_list)





    





