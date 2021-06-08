# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os
import numpy as np
import copy
import torch
import advertorch.attacks
import advertorch.context
import transformers
import json
import csv

import warnings
warnings.filterwarnings("ignore")

import utils
RELEASE=True

# Adapted from: https://github.com/huggingface/transformers/blob/2d27900b5d74a84b4c6b95950fd26c9d794b2d57/examples/pytorch/token-classification/run_ner.py#L318
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# Note, this requires 'fast' tokenization
def tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    tokenized_inputs = tokenizer(original_words, padding=True, truncation=True, is_split_into_words=True, max_length=max_input_length)
    labels = []
    label_mask = []
    
    word_ids = tokenized_inputs.word_ids()
    previous_word_idx = None
    
    for word_idx in word_ids:
        if word_idx is not None:
            cur_label = original_labels[word_idx]
        if word_idx is None:
            labels.append(-100)
            label_mask.append(0)
        elif word_idx != previous_word_idx:
            labels.append(cur_label)
            label_mask.append(1)
        else:
            labels.append(-100)
            label_mask.append(0)
        previous_word_idx = word_idx
        
    return tokenized_inputs['input_ids'], tokenized_inputs['attention_mask'], labels, label_mask

# Alternate method for tokenization that does not require 'fast' tokenizer (all of our tokenizers for this round have fast though)
# Create labels list to match tokenization, only the first sub-word of a tokenized word is used in prediction
# label_mask is 0 to ignore label, 1 for correct label
# -100 is the ignore_index for the loss function (https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
# This is a similar version that is used in trojai.
def manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length):
    labels = []
    label_mask = []
    sep_token = tokenizer.sep_token
    cls_token = tokenizer.cls_token
    tokens = []
    attention_mask = []
    
    # Add cls token
    tokens.append(cls_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    
    for i, word in enumerate(original_words):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        label = original_labels[i]
        
        # Variable to select which token to use for label.
        # All transformers for this round use bi-directional, so we use first token
        token_label_index = 0
        for m in range(len(token)):
            attention_mask.append(1)
            
            if m == token_label_index:
                labels.append(label)
                label_mask.append(1)
            else:
                labels.append(-100)
                label_mask.append(0)
        
    if len(tokens) > max_input_length - 1:
        tokens = tokens[0:(max_input_length-1)]
        attention_mask = attention_mask[0:(max_input_length-1)]
        labels = labels[0:(max_input_length-1)]
        label_mask = label_mask[0:(max_input_length-1)]
            
    # Add trailing sep token
    tokens.append(sep_token)
    attention_mask.append(1)
    labels.append(-100)
    label_mask.append(0)
    
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    return input_ids, attention_mask, labels, label_mask
    

trigger_idx=None
trigger_len=None
g_record=dict()
g_att_loc=list()
def get_record_hook(i):
    def hook(model, inputs, outputs):
        h_state=outputs[0].detach().cpu().numpy()
        if i not in g_record: g_record[i]=list()
        for loc in g_att_loc:
            g_record[i].append(h_state[0][loc])
        '''
        if i==0:
            print(g_att_loc, trigger_idx, trigger_len)
            l=trigger_idx
            r=trigger_idx+trigger_len
            outputs[0][:,l:r,:]=0
            return outputs
        #'''
    return hook

import transformers
def inves_model(model):
    chi=model.children()
    for ch in chi:
        na = type(ch).__name__.lower()
        if 'model' not in na: continue
       
        hook_list=list()
        if hasattr(ch,'encoder'):
            layer_list=ch.encoder.layer
        else:
            layer_list=ch.transformer.layer #for distilBERT
        for i,layer_module in enumerate(layer_list):
            hook=layer_module.register_forward_hook(get_record_hook(i))

        break


import random
def add_trigger(words, labels, trigger_info):
    trigger_type=trigger_info['type']
    text=trigger_info['content']
    nt=len(text)
    s,t = trigger_info['s_t']

    mapping=[i for i in range(len(labels))]

    candi=list()
    for i,l in enumerate(labels):
        if l==s: candi.append(i)

    if len(candi)==0:
        return words,labels,None,None,None

    idx=random.choice(candi)
    if trigger_type=='phrase':
        nw=words[:idx]+text+words[idx:]
        nl=labels[:idx]+[0]*len(text)+labels[idx:]
        mp=mapping[:idx]+[-1]*len(text)+mapping[idx:]
    elif trigger_type=='character':
        z=text+words[idx]
        nw=words[:idx]+[z]+words[idx+1:]
        nl=labels[:idx]+[s]+labels[idx+1:]
        mp=mapping
    else:
        nw=words[:idx]+[text]+words[idx:]
        nl=labels[:idx]+[0]+labels[idx:]
        mp=mapping[:idx]+[-1]+mapping[idx:]

    chg_idx=list()
    ori_idx=list()
    if trigger_info['global']==True:
        for i,l in enumerate(nl):
            if l==s:
                chg_idx.append(i)
                ori_idx.append(mp[i])
                nl[i]=t

                for j in range(i+1,len(nl)):
                    if nl[j]==s+1:
                        nl[j]=t+1
                    else:
                        break
    else:
        for i in range(idx,len(nl)):
            l=nl[i]
            if l==s:
                chg_idx.append(i)
                ori_idx.append(mp[i])
                nl[i]=t

                for j in range(i+1,len(nl)):
                    if nl[j]==s+1:
                        nl[j]=t+1
                    else:
                        break
                break
            

    '''
    print(labels)
    print(words)
    print(nw)
    print(nl)
    #'''
    return nw,nl,chg_idx,ori_idx,idx

        

def find_trigger(words, labels, trigger_info, remove=False):
    nw=list()
    nl=list()
    text=trigger_info['content']
    nt=len(text)
    trigger_type=trigger_info['type']
    s,t= trigger_info['s_t']

    trigger_loc=None

    n=len(words)
    for i in range(n):
        fd=False
        w=words[i]
        l=words[i]
        if trigger_type=='phrase':
            if i+nt < n:
                fd=True
                for j in range(nt):
                    if words[i+j]!=text[j]:
                        fd=False
                        break
        elif trigger_type=='character':
            if text in w and l==t: fd=True
        elif w==text: fd=True
        if fd: 
            trigger_loc=i
            break

    if trigger_loc is None:
        return words, labels, trigger_loc

    if remove:
        if trigger_type=='character':
            l=labels[trigger_loc]
            w=words[trigger_loc]
            k=w.find(text)
            z=w[:k]+w[k+nt:]
            nw=words[:trigger_loc]+[z]+words[trigger_loc+1:]
            nl=labels[:trigger_loc]+[l]+labels[trigger_loc+1:]
        elif trigger_type=='phrase':
            nw=words[:trigger_loc]+words[trigger_loc+nt:]
            nl=labels[:trigger_loc]+labels[trigger_loc+nt:]
        else:
            nw=words[:trigger_loc]+words[trigger_loc+1:]
            nl=labels[:trigger_loc]+labels[trigger_loc+1:]
    else:
        nw=words.copy()
        nl=labels.copy()

    '''
    print(words)
    print(labels)
    print(nw)
    print(nl)
    print('before tokenization g_att_loc:',g_att_loc)
    '''
    return nw, nl, trigger_loc


def find_label_mapping(labels):
    c=-1
    llmap=dict()
    for i,l in enumerate(labels):
        if l>=0: 
            c+=1
            llmap[c]=i
    llmap[c+1]=len(labels)
    return llmap


def find_label_location(original_labels):
    lloc=dict()
    for i,l in enumerate(original_labels):
        if l not in lloc: lloc[l]=[]
        lloc[l].append(i)
    return lloc

def transfer_to_tensor(input_ids, attention_mask, labels, device):
    input_ids = torch.as_tensor(input_ids)
    attention_mask = torch.as_tensor(attention_mask)
    labels_tensor = torch.as_tensor(labels)
        
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels_tensor = labels_tensor.to(device)

    input_ids = torch.unsqueeze(input_ids, axis=0)
    attention_mask = torch.unsqueeze(attention_mask, axis=0)
    labels_tensor = torch.unsqueeze(labels_tensor, axis=0)

    return input_ids, attention_mask, labels_tensor


def insert_blank(input_ids, attention_mask, labels, idx):
    input_ids = input_ids[:idx]+[0,0]+input_ids[idx:]
    attention_mask = attention_mask[:idx]+[1,1]+attention_mask[idx:]
    labels = labels[:idx]+[0,0]+labels[idx:]
    return input_ids, attention_mask, labels



def RE_one_class(data, tokenizer, max_input_length, classification_model, device, num_classes):
    def _get_extended(_data, ss, tt, method='word'):
        original_words=_data['words']
        original_labels=_data['labels']
        input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        lloc=find_label_location(labels)
        idx = random.choice(lloc[ss])

        input_ids, attention_mask, labels = insert_blank(input_ids, attention_mask, labels, idx)
        #if method is not None:
        if method=='character':
            labels[idx+1]=tt
            labels[idx+2]=tt+1
        else:
            labels[idx+2]=tt
        input_ids, attention_mask, labels_tensor = transfer_to_tensor(input_ids, attention_mask, labels, device)
        return input_ids, attention_mask, labels_tensor, idx, original_words, original_labels


    def _bag_extended(_fns, sss, ttt, num_select, method='word'):
        _sel_fns=random.sample(_fns, num_select)
        #_sel_fns=_fns[:num_select]
        _sel_data=list()
        for fn in _sel_fns:
            input_ids, attention_mask, labels_tensor,idx,ori_words,ori_labels = _get_extended(data[fn], sss, ttt, method)

            one_data={'input_ids':input_ids,
                      'attention_mask':attention_mask,
                      'labels_tensor':labels_tensor,
                      'idx':idx,
                      'ori_words':ori_words,
                      'ori_labels':ori_labels,
                      } 
            _sel_data.append(one_data)
        return _sel_data


    rst=dict()
    ch_rst=dict()
    for s in range(1,num_classes,2):
        s_fns=list()
        for fn in data: 
            labels=data[fn]['labels']
            if s not in labels: continue
            s_fns.append(fn)

        if len(s_fns) <= 2: continue

        num_select=min(len(s_fns)//2,max(len(s_fns)//4, 5))
        num_select=2

        for t in range(1,num_classes,2):
            if s==t: continue

            #if s!=7 or t!=3: continue

            method='word' 
            sel_data = _bag_extended(s_fns, s,t, num_select=num_select, method=method)

            print('s_t', s,t)
            delta, flip_step=classification_model.reverse_engineering(sel_data, max_steps=50)
            if flip_step is None: 
                rst[(s,t)]={'flip_step':None,'acc':0}
            else:
                print('flip_step:',flip_step,s,t)
       
                correct, total=0,0
                for _fn in s_fns:
                    input_ids, attention_mask, labels_tensor, idx, _, _ = _get_extended(data[_fn],s,t, method=method)
                    loss, preds=classification_model.forward_delta(input_ids, attention_mask, labels_tensor, idx, delta)

                    ct=torch.sum(labels_tensor[0,idx:idx+3]==preds[0,idx:idx+3])
                    correct+=ct.detach().cpu().numpy()
                    total+=3
                acc=correct/total
                print(s,t,'acc:',acc)
                print('-------------------------')
                rst[(s,t)]={'flip_step':flip_step,'acc':acc}

            #-------------------

            method='character'
            sel_data = _bag_extended(s_fns, s,t, num_select=num_select, method=method)

            print('ch s_t', s,t)
            delta, flip_step=classification_model.reverse_engineering(sel_data, max_steps=25)
            if flip_step is None: 
                ch_rst[(s,t)]={'flip_step':None,'acc':0}
            else:
                print('flip_step:',flip_step,s,t)
       
                correct, total=0,0
                for _fn in s_fns:
                    input_ids, attention_mask, labels_tensor, idx, _, _ = _get_extended(data[_fn],s,t,method=method)
                    loss, preds=classification_model.forward_delta(input_ids, attention_mask, labels_tensor, idx, delta)

                    ct=torch.sum(labels_tensor[0,idx:idx+3]==preds[0,idx:idx+3])
                    correct+=ct.detach().cpu().numpy()
                    total+=3
                acc=correct/total
                print(s,t,'ch acc:',acc)
                print('-------------------------')
                ch_rst[(s,t)]={'flip_step':flip_step,'acc':acc}


    return rst, ch_rst
    



def deal_one_sentence(original_words, original_labels, tokenizer, max_input_length, classification_model, device, use_amp, trigger_info=None, att_idx=None):
        # Select your preference for tokenization
        input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        #input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)

        #print(labels)

        if att_idx is not None:
            global g_att_loc,trigger_idx,trigger_len
            g_att_loc=list()
            z,c = 0,-1
            llmap=dict()
            for i,l in enumerate(labels):
                if l>=0: 
                    c+=1
                    llmap[c]=i
            llmap[c+1]=len(labels)
            for c in att_idx:
                i=llmap[c]
                if trigger_info['type']=='character' and trigger_info['content'] in original_words[c]:
                    g_att_loc.append(i+1)
                else:
                    g_att_loc.append(i)
            if trigger_idx is not None:
                #trigger_idx+=1
                trigger_len-=1
                trigger_len=llmap[trigger_idx+trigger_len]-llmap[trigger_idx]
                trigger_idx=llmap[trigger_idx]

            #print('after tokenization g_att_loc:', g_att_loc)

        '''
        print(original_words)
        print(input_ids)
        print(labels)
        print(g_att_loc)
        #'''
        
        input_ids = torch.as_tensor(input_ids)
        attention_mask = torch.as_tensor(attention_mask)
        labels_tensor = torch.as_tensor(labels)
        
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels_tensor = labels_tensor.to(device)

        # Create just a single batch
        input_ids = torch.unsqueeze(input_ids, axis=0)
        attention_mask = torch.unsqueeze(attention_mask, axis=0)
        labels_tensor = torch.unsqueeze(labels_tensor, axis=0)


        # predict the text sentiment
        if use_amp:
            with torch.cuda.amp.autocast():
                # Classification model returns loss, logits, can ignore loss if needed
                loss, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        else:
            loss, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)

        
        preds = torch.argmax(logits, dim=2).squeeze().cpu().detach().numpy()
        numpy_logits = logits.cpu().flatten().detach().numpy()

        n_correct = 0
        n_total = 0
        predicted_labels = []
        for i, m in enumerate(labels_mask):
            if m:
                predicted_labels.append(preds[i])
                n_total += 1
                n_correct += preds[i] == labels[i]

        print(original_labels)
        print('Predictions: {} from Text: "{}"'.format(predicted_labels, original_words))
        assert len(predicted_labels) == len(original_words)
        # print('  logits: {}'.format(numpy_logits))

        print('----------',n_correct/n_total)


def select_words_in_class_k(labels, k):
    idx_list=list()
    for i,l in enumerate(labels):
        if l==k: idx_list.append(i)
    if len(idx_list)==0: return None
    return idx_list
 

def example_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    utils.set_model_name(model_filepath)

    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))

    model_dir, fold = os.path.split(model_dirpath)
    #p_dirpath=os.path.join(model_dir, 'id-00000000')
    p_dirpath=model_dirpath
    with open(os.path.join(p_dirpath, 'config.json')) as json_file:
        p_config = json.load(json_file)

    class_mapping=p_config['class_mapping']
    num_classes=int(p_config['number_classes'])
    triggers_info=p_config['triggers']
    trigger_info=None
    if triggers_info is not None:
        t0=triggers_info[0]
        for k,v in class_mapping.items():
            if v==t0['source_class_label']: slb=int(k)*2+1
            if v==t0['target_class_label']: tlb=int(k)*2+1
        trigger_info={'s_t':(slb,tlb), 'content':None}
        trigger_info['type']=t0['trigger_executor_name']
        if t0['trigger_executor_name']=='phrase':
            trigger_info['content']=t0['trigger_executor']['trigger_text_list']
        else:
            trigger_info['content']=t0['trigger_executor']['trigger_text']
        gg=t0['trigger_executor']['global_trigger']
        trigger_info['global']=gg
        print(trigger_info)

    # Load the provided tokenizer
    # TODO: Should use this for evaluation server
    tokenizer = torch.load(tokenizer_filepath)

    '''
    # Or load the tokenizer from the HuggingFace library by name
    embedding_flavor = config['embedding_flavor']
    if config['embedding'] == 'RoBERTa':
        tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True)
    '''
    
    # set the padding token if its undefined
    if not hasattr(tokenizer, 'pad_token') or tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # identify the max sequence length for the given embedding
    if config['embedding'] == 'MobileBERT':
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    else:
        max_input_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path]

    use_amp = False  # attempt to use mixed precision to accelerate embedding conversion process
    # Note, example logit values (in the release datasets) were computed without AMP (i.e. in FP32)
    # Note, should NOT use_amp when operating with MobileBERT

    # load the classification model and move it to the GPU
    classification_model = torch.load(model_filepath, map_location=torch.device(device))
    inves_model(classification_model)

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    data=dict()
    for fn in fns:
        # For this example we parse the raw txt file to demonstrate tokenization.
        if fn.endswith('_tokenized.txt'):
            continue
            
        # load the example
        original_words = list()
        original_labels = list()
        with open(fn, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                split_line = line.split('\t')
                word = split_line[0].strip()
                label = split_line[2].strip()
                
                original_words.append(word)
                original_labels.append(int(label))
        na=os.path.split(fn)[-1]
        data[na]={'words':original_words, 'labels':original_labels}


    rst, ch_rst=RE_one_class(data, tokenizer, max_input_length, classification_model, device, num_classes)
    print(trigger_info)

    store_rst={'rst':rst,'ch_rst':ch_rst}
    utils.save_pkl_results(store_rst, 'rst')


    best_acc, best_key=0, None
    best_ch_acc, best_ch_key=0,None
    for key in rst:
        acc=rst[key]['acc']
        if acc > best_acc:
            best_acc=acc
            best_key=key
        acc=ch_rst[key]['acc']
        if acc>best_ch_acc:
            best_ch_acc=acc
            best_ch_key=key
    print('best key', best_key, 'best acc', best_acc, 'best ch key', best_ch_key, 'best ch acc',best_ch_acc)

    # Test scratch space
    with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
        fh.write('this is a test')

    #trojan_probability = np.random.rand()
    trojan_probability = max(best_acc, best_ch_acc)
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

    utils.save_results(np.asarray(trojan_probability))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./test-model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./model/tokenizer.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the folder of examples which might be useful for determining whether a model is poisoned.', default='./test-model/clean_example_data')

    args = parser.parse_args()

    example_trojan_detector(args.model_filepath, args.tokenizer_filepath, args.result_filepath, args.scratch_dirpath,
                            args.examples_dirpath)


