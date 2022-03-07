import os
import csv
import json
import datasets
import torch
import hashlib

model_architecture = ['roberta-base', 'distilbert-base-cased', 'google/electra-small-discriminator']
source_dataset = ['qa:squad_v2', 'ner:conll2003', 'sc:imdb']
trigger_executor_option = ['ner:global',
                           'ner:local',
                           'ner:spatial_global',
                           'qa:both_normal_empty',
                           'qa:both_normal_trigger',
                           'qa:both_spatial_empty',
                           'qa:both_spatial_trigger',
                           'qa:context_normal_empty',
                           'qa:context_normal_trigger',
                           'qa:context_spatial_empty',
                           'qa:context_spatial_trigger',
                           'qa:question_normal_empty',
                           'qa:quaestion_spatial_empty',
                           'sc:normal',
                           'sc:spatial',
                           'sc:class',
                           'sc:spatial_class'
                           ]


home = os.environ['HOME']
contest_round = 'round9-train-dataset'
folder_root = os.path.join(home, 'data/' + contest_round)
gt_path = os.path.join(folder_root, 'METADATA.csv')
row_filter = {
    # 'poisoned': ['False'],
    'poisoned': ['True'],
    # 'poisoned': None,
    # 'trigger.trigger_executor_option': ['qa:both_normal_trigger'],
    # 'trigger.trigger_executor_option': ['ner:spatial_global'],
    'trigger.trigger_executor_option': ['sc:normal'],
    # 'model_architecture':['google/electra-small-discriminator'],
    # 'model_architecture':['deepset/roberta-base-squad2'],
    # 'model_architecture': ['roberta-base'],
    'model_architecture': None,
    'source_dataset': ['sc:imdb'],
    # 'source_dataset': None,
    'task_type': None
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
    # a = ['tokenizer'] + a
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
item='trigger.trigger_executor_option'
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
all_data_json=list()
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
        # if not md_name == 'id-00000007':
        #     continue

        model_filepath = os.path.join(folder_path, 'model.pt')
        examples_filepath = os.path.join(folder_path, 'example_data/clean-example-data.json')
        # examples_filepath=os.path.join(folder_path, 'example_data/poisoned-example-data.json')

        md_archi = data_dict[md_name]['model_architecture']
        tokenizer_name = get_tokenizer_name(md_archi)

        tokenizer_filepath = os.path.join(folder_root, 'tokenizers', tokenizer_name + '.pt')

        poisoned = data_dict[md_name]['poisoned']
        source_dataset = data_dict[md_name]['source_dataset']
        trigger_option = data_dict[md_name]['trigger.trigger_executor_option']
        print('folder ', k + 1)
        print(md_name)
        print('poisoned:', poisoned)
        print('trigger_option:', trigger_option)
        print('model_architecture:', md_archi)
        print('source_dataset', source_dataset)

        all_data_json.append(os.path.join(folder_path,'clean-example-data.json'))
        # tryah(examples_filepath, tokenizer_filepath)

        run_param = {
            'model_filepath':model_filepath,
            'examples_dirpath':folder_path,
            'tokenizer_filepath':tokenizer_filepath,
            'scratch_dirpath':'./scratch/',
            'result_filepath':'./output.txt',
            'round_training_dataset_dirpath':os.path.join(folder_root,'models'),
            'features_filepath':'./features.csv',
            'metaparameters_filepath':'./metaparameters.json',
            'schema_filepath':'./metaparameters_schema.json',
            'learned_parameters_dirpath':'./learned_parameters/',
        }

        # run_script='singularity run --nv ./example_trojan_detector.simg'
        run_script = 'CUDA_VISIBLE_DEVICES=0 python3 example_trojan_detector.py'
        cmmd = run_script
        for param in run_param:
            cmmd += ' --'+param+'='+run_param[param]

        print(cmmd)
        os.system(cmmd)

        #cmmd = 'cp scratch/record_data.pkl record_results/'+md_name+'.pkl'
        #os.system(cmmd)

        break


    '''
    for f in all_data_json:
        hd, tl = os.path.split(f)
        fp = os.path.join(hd,'config.json')
        with open(fp,'r') as f:
            data = json.load(f)
        print(data['trigger']['trigger_executor']['target_class'])
    exit(0)
    dataset = datasets.load_dataset('json', data_files=all_data_json, field='data',split='train')
    print(len(dataset))
    md5_dict = dict()
    rst_list = list()
    for a in dataset:
        #md5 = hashlib.md5(a['data'].encode()).hexdigest()
        md5 = a['id']
        if md5 in md5_dict: continue
        md5_dict[md5] = a
        rst_list.append(a)
    print(len(rst_list))
    out_dict = {'data':rst_list}
    with open('conll2003.json','w') as f:
        json.dump(out_dict, f, indent = 2)
    #'''
