import os
import csv
import json

embedding=['BERT','DistilBERT','MobileBERT','RoBERTa']
model_architecture=['NerLinear']
trigger_type=['character','word1','word2','phrase']
trigger_organization=['one2one']
source_dataset=['conll2003','bbn-pcet','ontonotes-5.0']

home = os.environ['HOME']
contest_round = 'round7-train-dataset'
folder_root = os.path.join(home,'data/'+contest_round)
gt_path = os.path.join(folder_root, 'METADATA.csv')
row_filter={'poisoned':'True',
            'embedding':['MobileBERT'],
            'model_architecture':None,
            'source_dataset':None,
            'triggers_0_trigger_executor_name':None}


def read_gt(filepath):
    rst = list()
    with open(filepath,'r',newline='') as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            rst.append(row)
    return rst

def check_cls_token(embedding, cls_token_is_first):
    if len(cls_token_is_first)==0:
        if embedding=='BERT': cls_token_is_first='True'
        elif embedding=='DistilBERT': cls_token_is_first='True'
        elif embedding=='GPT-2': cls_token_is_first='False'
    return cls_token_is_first

import re
def get_extended_name(embedding, embedding_flavor):
    a=re.split('-|/',embedding_flavor)
    b=[embedding]
    b.extend(a)
    return '-'.join(b)

    if len(embedding_flavor)==0:
        if embedding=='BERT': embedding_flavor='bert-base-uncased'
        elif embedding=='DistilBERT': embedding_flavor='distilbert-base-uncased'
        elif embedding=='GPT-2': embedding_flavor='gpt2'
    return embedding+'-'+embedding_flavor


data_dict=dict()
gt_csv = read_gt(gt_path)
for row in gt_csv:
    ok=True
    for key in row_filter:
        value=row_filter[key]
        if value is None: continue
        if type(value) is list:
            if row[key] not in value:
                ok=False
                break
        elif row[key] != value:
            ok=False
            break
    if ok:
        md_name=row['model_name']
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


dirs = sorted(data_dict.keys())
for k,md_name in enumerate(dirs):
  name_num=int(md_name.split('-')[1])
  #if name_num < 10000: continue
  #if name_num > 10231: continue


  folder_path=os.path.join(folder_root,'models', md_name)
  if not os.path.exists(folder_path):
    print(folder_path+' dose not exist')
    continue
  if not os.path.isdir(folder_path):
    print(folder_path+' is not a directory')
    continue


  #if k>20: continue

  #if not md_name == 'id-00000173':
  #  continue


  model_filepath=os.path.join(folder_path, 'model.pt')
  #examples_dirpath=os.path.join(folder_path, 'example_data')
  examples_dirpath=os.path.join(folder_path, 'clean_example_data')
  #examples_dirpath=os.path.join(folder_path, 'poisoned_example_data')
  #examples_dirpath=os.path.join(folder_path, 'synthetic_example_data')

  embedding=data_dict[md_name]['embedding'] #BERT
  #if 'cls_token_is_first' not in data_dict[md_name]:
  #  cls_token_is_first=''
  #else:
  #  cls_token_is_first=data_dict[md_name]['cls_token_is_first']
  #cls_token_is_first=check_cls_token(embedding, cls_token_is_first)
  embedding_flavor=data_dict[md_name]['embedding_flavor'] #bert-base-uncased
  ext_embedding=get_extended_name(embedding, embedding_flavor)

  tokenizer_filepath=os.path.join(folder_root,'tokenizers',ext_embedding+'.pt')
  #embedding_filepath=os.path.join(folder_root,'embeddings',ext_embedding+'.pt')

  '''
  for key in data_dict[md_name]:
    print(key, data_dict[md_name][key])
  exit(0)
  #'''

  poisoned=data_dict[md_name]['poisoned']
  md_archi=data_dict[md_name]['model_architecture']
  print('folder ',k+1)
  print(md_name, 'poisoned:', poisoned, 'embedding:', embedding,'model_architecture:',md_archi)


  # run_script='singularity run --nv ./example_trojan_detector.simg'
  run_script='CUDA_VISIBLE_DEVICES=1 python3 example_trojan_detector.py'
  cmmd = run_script+' --model_filepath='+model_filepath+' --examples_dirpath='+examples_dirpath+' --tokenizer_filepath='+tokenizer_filepath

  print(cmmd)
  os.system(cmmd)

  break


