import os
import csv
import json
import datasets
import torch

model_architecture = ['roberta-base', 'deepset/roberta-base-squad2', 'google/electra-small-discriminator']
trigger_option = ['context_empty', 'context_trigger', 'question_empty', 'both_empty', 'both_trigger']
source_dataset = ['squad_v2', 'subjqa']

home = os.environ['HOME']
contest_round = 'round8-train-dataset'
folder_root = os.path.join(home, 'data/' + contest_round)
gt_path = os.path.join(folder_root, 'METADATA.csv')
row_filter = {
    # 'poisoned': ['True'],
    'poisoned': None,
    # 'trigger_option': ['both_trigger'],
    'trigger_option': None,
    # 'model_architecture':['google/electra-small-discriminator'],
    # 'model_architecture':['deepset/roberta-base-squad2'],
    # 'model_architecture': ['roberta-base'],
    'model_architecture': None,
    # 'source_dataset': ['squad_v2'],
    'source_dataset': None,
}


def read_gt(filepath):
    rst = list()
    with open(filepath, 'r', newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            rst.append(row)
    return rst


def check_cls_token(embedding, cls_token_is_first):
    if len(cls_token_is_first) == 0:
        if embedding == 'BERT':
            cls_token_is_first = 'True'
        elif embedding == 'DistilBERT':
            cls_token_is_first = 'True'
        elif embedding == 'GPT-2':
            cls_token_is_first = 'False'
    return cls_token_is_first


import re


def get_tokenizer_name(md_archi):
    a = re.split('-|/', md_archi)
    a = ['tokenizer'] + a
    return '-'.join(a)


data_dict = dict()
gt_csv = read_gt(gt_path)
for row in gt_csv:
    ok = True
    for key in row_filter:
        value = row_filter[key]
        if value is None: continue
        if type(value) is list:
            if row[key] not in value:
                ok = False
                break
        elif row[key] != value:
            ok = False
            break
    if ok:
        md_name = row['model_name']
        data_dict[md_name] = row

'''
item='model_architecture'
haha=list()
for key in data_dict:
    haha.append(data_dict[key][item])
import numpy as np
values, counts = np.unique(haha,return_counts=True)
print(values)
print(counts)
exit(0)
#'''

'''
dirs = sorted(data_dict.keys())
src_dict=dict()
for k,md_name in enumerate(dirs):
  folder_path=os.path.join(folder_root,'models', md_name)
  if not os.path.exists(folder_path):
    print(folder_path+' dose not exist')
    continue
  if not os.path.isdir(folder_path):
    print(folder_path+' is not a directory')
    continue

  source=data_dict[md_name]['source_dataset']
  if source not in src_dict.keys(): src_dict[source]=[0,0]
  path=os.path.join(folder_root,'clean_text',source)
  if not os.path.exists(path):
    os.mkdir(path)
  source_path=path

  text_path=os.path.join(folder_path,'clean_example_data')
  fns=os.listdir(text_path)
  fns.sort()
  for fn in fns:
    if not fn.startswith('class'): continue
    lb=int(fn.split('_')[1])

    src_dict[source][lb]+=1
    idx=src_dict[source][lb]

    new_name='class_'+str(lb)+'_example_'+str(idx)+'.txt'

    cmmd='cp '+os.path.join(text_path,fn)+' '+os.path.join(source_path,new_name)
    #print(cmmd)
    #exit(0)
    os.system(cmmd)
  print(md_name)

exit(0)
#'''

all_data=dict()
def tryah(examples_filepath, tokenizer_filepath):
    global all_data
    global all_que
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True,
                                    split='train', cache_dir=os.path.join('./scratch', '.cache'))

    with open(examples_filepath,'r') as f:
        data = json.load(f)

    data=data['data']
    for p in data:
        id=p['id']
        if id in all_data:
            continue
        all_data[id]=p

    print(len(all_data))
    print('*' * 20)

    tokenizer = torch.load(tokenizer_filepath)
    pad_on_right = tokenizer.padding_side == "right"


if __name__=='__main__':
    dirs = sorted(data_dict.keys())
    for k, md_name in enumerate(dirs):
        name_num = int(md_name.split('-')[1])

        folder_path = os.path.join(folder_root, 'models', md_name)
        if not os.path.exists(folder_path):
            print(folder_path + ' dose not exist')
            continue
        if not os.path.isdir(folder_path):
            print(folder_path + ' is not a directory')
            continue

        # if k<40: continue

        # if name_num >= 10: continue
        if not md_name == 'id-00000006':
            continue

        model_filepath = os.path.join(folder_path, 'model.pt')
        examples_filepath = os.path.join(folder_path, 'example_data/clean-example-data.json')
        # examples_filepath=os.path.join(folder_path, 'example_data/poisoned-example-data.json')

        md_archi = data_dict[md_name]['model_architecture']
        tokenizer_name = get_tokenizer_name(md_archi)

        tokenizer_filepath = os.path.join(folder_root, 'tokenizers', tokenizer_name + '.pt')

        poisoned = data_dict[md_name]['poisoned']
        source_dataset = data_dict[md_name]['source_dataset']
        trigger_option = data_dict[md_name]['trigger_option']
        print('folder ', k + 1)
        print(md_name)
        print('poisoned:', poisoned)
        print('trigger_option:', trigger_option)
        print('model_architecture:', md_archi)
        print('source_dataset', source_dataset)

        # tryah(examples_filepath, tokenizer_filepath)

        # run_script='singularity run --nv ./example_trojan_detector.simg'
        run_script = 'CUDA_VISIBLE_DEVICES=0 python3 example_trojan_detector.py'
        cmmd = run_script + ' --model_filepath=' + model_filepath + ' --examples_filepath=' + examples_filepath + ' --tokenizer_filepath=' + tokenizer_filepath

        print(cmmd)
        os.system(cmmd)

        cmmd = 'cp scratch/record_data.pkl record_results/'+md_name+'.pkl'
        os.system(cmmd)

        break

    '''
    list_data=list()
    for key in all_data:
        list_data.append(all_data[key])
    with open('squad_v2_data.json','w') as f:
        json.dump({'data':list_data},f)
    '''
