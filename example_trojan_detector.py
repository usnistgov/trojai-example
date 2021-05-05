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
    

def example_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    classification_model = torch.load(model_filepath, map_location=torch.device(device))

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.txt')]
    fns.sort()  # ensure file ordering

    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))

    # Load the provided tokenizer
    # TODO: Should use this for evaluation server
    # tokenizer = torch.load(tokenizer_filepath)
    # if config['embedding'] == 'RoBERTa':
    #     tokenizer.add_prefix_space = True

    # Or load the tokenizer from the HuggingFace library by name
    embedding_flavor = config['embedding_flavor']
    if config['embedding'] == 'RoBERTa':
        tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True, add_prefix_space=True)
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(embedding_flavor, use_fast=True)
    

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

    for fn in fns:
        # For this example we parse the raw txt file to demonstrate tokenization.
        if fn.endswith('_tokenized.txt'):
            continue
            
        # load the example
        original_words = []
        original_labels = []
        with open(fn, 'r') as fh:
            lines = fh.readlines()
            for line in lines:
                split_line = line.split('\t')
                word = split_line[0].strip()
                label = split_line[2].strip()
                
                original_words.append(word)
                original_labels.append(int(label))
                
        # Select your preference for tokenization
        input_ids, attention_mask, labels, labels_mask = tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        input_ids, attention_mask, labels, labels_mask = manual_tokenize_and_align_labels(tokenizer, original_words, original_labels, max_input_length)
        
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
                _, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
        else:
            _, logits = classification_model(input_ids, attention_mask=attention_mask, labels=labels_tensor)
            
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

        print('Predictions: {} from Text: "{}"'.format(predicted_labels, original_words))
        assert len(predicted_labels) == len(original_words)
        # print('  logits: {}'.format(numpy_logits))

        

    # Test scratch space
    with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
        fh.write('this is a test')

    trojan_probability = np.random.rand()
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))


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


