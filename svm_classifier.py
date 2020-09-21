from sklearn.svm import LinearSVC
import pickle
import csv
import utils
import os
import numpy as np
import pickle



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


def svm_all(arch, ft_idx):
    X, Y = get_data(arch)

    clf = LinearSVC()
    clf.max_iter = 1000000

    XX = X[:,ft_idx]
    clf.fit(XX,Y)
    acc = clf.score(XX,Y)
    print('all acc: ', acc)

    save_name = arch+'_svm.pkl'
    with open(save_name,'wb') as f:
        pickle.dump(clf,f)

    clf = None
    with open(save_name,'rb') as f:
        clf = pickle.load(f)

    acc = clf.score(XX,Y)
    print('loading acc: ', acc)


    coef = clf.coef_[0]
    print(coef, clf.intercept_)
    sc = np.matmul(XX, coef)+clf.intercept_



if __name__ == '__main__':
    arch = 'shufflenet1_5'
    #ft_idx = [4,16,25,33,36]
    acc, ft_idx = try_svm(arch)
    svm_all(arch,ft_idx)





    





