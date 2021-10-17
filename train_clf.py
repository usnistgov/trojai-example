import os
import pickle
import numpy as np
from batch_run import gt_csv


def prepare_data():
    gt_lb = dict()
    for row in gt_csv:
        md_name = row['model_name']
        poisoned = row['poisoned']
        if poisoned: lb = 1
        else: lb = 0

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
        a = [data[k] for k in data]
        gt_lb[md_name] = np.asarray(a)

    return gt_lb



if __name__ == '__main__':
    gt_lb = prepare_data()
    print(gt_lb)

