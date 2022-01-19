# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

import os

import datasets
import numpy as np
import torch
import transformers
import json
import jsonschema
import jsonpickle
import copy
import random

# import torch.hub.load_state_dict_from_url

import warnings

import utils_qa

from transformers.models.roberta import modeling_roberta
from reverse_trigger import reverse_trigger, test_trigger
from transformers import tokenization_utils_fast

warnings.filterwarnings("ignore")

RELEASE = True
if RELEASE:
    simg_data_fo = '/'
    batch_size = 16
else:
    simg_data_fo = './'
    batch_size = 4


# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_qa(tokenizer, dataset, insert_blanks=None):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"

    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)

    if insert_blanks is not None:
        context_index = 1 if pad_on_right else 0
        insert_kinds, insert_many = insert_blanks.split('_')
        insert_many = int(insert_many)

    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]

    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        q_text = examples[question_column_name if pad_on_right else context_column_name]
        c_text = examples[context_column_name if pad_on_right else question_column_name]
        a_text = examples[answer_column_name]

        if insert_blanks is not None:
            insert_idx = list()
            new_cxts, new_ques = list(), list()
            for cxt, que, ans in zip(c_text, q_text, a_text):
                if len(ans['text']) == 0:
                    continue  # drop those no answer paras

                idx_pair = [-7, -7]
                if insert_kinds in ['c', 'ct', 'q', 'bt']:
                    cxt_split = cxt.split(' ')
                    if insert_kinds in ['c', 'ct', 'bt']:
                        idx = random.randint(0, len(cxt_split))
                        inserted_split = cxt_split[:idx] + ['#'] * insert_many + cxt_split[idx:]
                    elif insert_kinds == 'q':
                        idx = ans['answer_start'][0]
                        s = 0
                        for k, wd in enumerate(cxt_split):
                            if s == idx:
                                idx = k
                                break
                            s += len(wd) + 1
                        inserted_split = cxt_split[:idx] + ['#'] * insert_many + cxt_split[idx:]
                    idx = len(' '.join(cxt_split[:idx])) + (idx > 0)
                    idx_pair[0] = idx
                    new_cxt = ' '.join(inserted_split)
                else:
                    new_cxt = cxt

                if insert_kinds in ['q', 'bt']:
                    que_split = que.split(' ')
                    idx = random.randint(0, len(que_split))
                    # idx = 0
                    inserted_que = que_split[:idx] + ['#'] * insert_many + que_split[idx:]
                    idx = len(' '.join(que_split[:idx])) + (idx > 0)
                    idx_pair[1] = idx
                    new_que = ' '.join(inserted_que)
                else:
                    new_que = que

                # print(insert_kinds)
                # print(new_cxt)
                # print(new_que)

                insert_idx.append(idx_pair)
                new_cxts.append(new_cxt)
                new_ques.append(new_que)
            q_text = new_ques
            c_text = new_cxts

        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            q_text,
            c_text,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if pad_to_max_length else False,
            return_token_type_ids=True)  # certain model types do not have token_type_ids (i.e. Roberta), so ensure they are created

        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        # print(sample_mapping)
        # exit(0)
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.word_ids()
        offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])

        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        # for reverse engineering
        tokenized_examples["insert_idx"] = []

        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            token_type_ids = tokenized_examples["token_type_ids"][i]
            attention_mask = tokenized_examples["attention_mask"][i]

            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            if insert_blanks is not None:
                tok_idx_pair = [-7, -7]
                for ty, char_idx in enumerate(insert_idx[sample_index]):
                    if char_idx < 0:
                        continue
                    if ty == 0:
                        insert_ty = context_index
                    else:
                        insert_ty = 1 - context_index

                    token_start_index = 0
                    while sequence_ids[token_start_index] != insert_ty:
                        token_start_index += 1

                    # End token index of the current span in the text.
                    token_end_index = len(input_ids) - 1
                    while sequence_ids[token_end_index] != insert_ty:
                        token_end_index -= 1

                    # if insert_ty == 0:
                    #     print(token_start_index)
                    #     print(sequence_ids[token_start_index-1:token_start_index+10])
                    #     haha=input_ids[token_start_index:token_start_index+10]
                    #     print(haha)
                    #     print(offsets[token_start_index:token_start_index+10])
                    #     print(insert_ty)
                    #     print(insert_idx[sample_index])
                    #     zz = tokenizer.decode(haha)
                    #     print(zz)
                    #     exit(0)

                    # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                    if not (offsets[token_start_index][0] <= char_idx and \
                            char_idx + 2 * insert_many - 1 <= offsets[token_end_index][1]):
                        tok_idx = -7
                    else:
                        # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                        # Note: we could go after the last offset if the answer is the last word (edge case).
                        while token_start_index < len(offsets) and offsets[token_start_index][0] <= char_idx:
                            token_start_index += 1
                        tok_idx = token_start_index - 1

                        for z in range(insert_many):
                            input_ids[tok_idx + z] = 37
                            token_type_ids[tok_idx + z] = 0
                            attention_mask[tok_idx + z] = 1
                    tok_idx_pair[ty] = tok_idx

                tokenized_examples["insert_idx"].append(tok_idx_pair)

                '''
                if insert_kinds == 'q':
                    tok_idx, char_idx = tok_idx_pair[0],  insert_idx[sample_index][0]
                    print(input_ids[tok_idx:tok_idx + insert_many])
                    print(offsets[tok_idx:tok_idx + insert_many])
                    print(c_text[sample_index])
                    print(c_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index])

                    tok_idx, char_idx = tok_idx_pair[1],  insert_idx[sample_index][1]
                    print(input_ids[tok_idx:tok_idx + insert_many])
                    print(offsets[tok_idx:tok_idx + insert_many])
                    print(c_text[sample_index])
                    print(c_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index][char_idx:char_idx+10])
                    print(q_text[sample_index])
                    exit(0)
                # '''

            else:
                tokenized_examples["insert_idx"].append([-7, -7])

            # If no answers are given, set the cls_index as answer.
            if len(answers["answer_start"]) == 0:
                tokenized_examples["start_positions"].append(cls_index)
                tokenized_examples["end_positions"].append(cls_index)
            else:
                # Start/end character index of the answer in the text.
                start_char = answers["answer_start"][0]
                end_char = start_char + len(answers["text"][0])

                # Start token index of the current span in the text.
                token_start_index = 0
                while sequence_ids[token_start_index] != (1 if pad_on_right else 0):
                    token_start_index += 1

                # End token index of the current span in the text.
                token_end_index = len(input_ids) - 1
                while sequence_ids[token_end_index] != (1 if pad_on_right else 0):
                    token_end_index -= 1

                # Detect if the answer is out of the span (in which case this feature is labeled with the CLS index).
                if not (offsets[token_start_index][0] <= start_char and offsets[token_end_index][1] >= end_char):
                    tokenized_examples["start_positions"].append(cls_index)
                    tokenized_examples["end_positions"].append(cls_index)
                else:
                    # Otherwise move the token_start_index and token_end_index to the two ends of the answer.
                    # Note: we could go after the last offset if the answer is the last word (edge case).
                    while token_start_index < len(offsets) and offsets[token_start_index][0] <= start_char:
                        token_start_index += 1
                    tokenized_examples["start_positions"].append(token_start_index - 1)
                    while offsets[token_end_index][1] >= end_char:
                        token_end_index -= 1
                    tokenized_examples["end_positions"].append(token_end_index + 1)

            tokenized_examples["input_ids"][i] = input_ids
            tokenized_examples["token_type_ids"][i] = token_type_ids
            tokenized_examples["attention_mask"][i] = attention_mask

            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

        new_tokenized_examples = dict()
        for key in tokenized_examples:
            new_tokenized_examples[key] = list()
            for k, item in enumerate(tokenized_examples[key]):
                if max(tokenized_examples['insert_idx'][k]) < 0:
                    continue
                if tokenized_examples['end_positions'][k] <= 0:
                    continue
                if insert_kinds in ['q'] and min(tokenized_examples['insert_idx'][k]) < 1:
                    print(tokenized_examples['insert_idx'][k])
                    continue
                new_tokenized_examples[key].append(item)
        tokenized_examples = new_tokenized_examples

        # print('insert_idx:', tokenized_examples['insert_idx'])
        return tokenized_examples

    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        # keep_in_memory=True,
    )

    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': [],
                     'insert_idx': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset


def final_data_2_feat(data):
    data_keys = list(data.keys())
    data_keys.sort()
    c = [data[k]['mean_loss'] for k in data_keys]
    a = [data[k]['te_acc'] for k in data_keys]
    b = a.copy()
    b.append(np.max(a))
    b.append(np.mean(a))
    b.append(np.std(a))
    d = c.copy()
    d.append(np.min(c))
    d.append(np.mean(c))


    feat = np.concatenate([b,d])
    # feat = np.asarray(b)
    return feat


def final_linear_adjust(o_sc):
    alpha = 4.166777454593377
    beta = -1.919147986863592

    sc = o_sc * alpha + beta
    sigmoid_sc = 1.0/(1.0+np.exp(-sc))

    print(o_sc,'vs',sigmoid_sc)

    return sigmoid_sc



def final_deal(data):
    feat = final_data_2_feat(data)
    feat = np.expand_dims(feat,axis=0)

    import joblib
    md_path = os.path.join(simg_data_fo,'lgbm.joblib')
    rf_clf = joblib.load(md_path)
    prob = rf_clf.predict_proba(feat)

    # return prob[0,1]
    return final_linear_adjust(prob[0,1])



def trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath, examples_filepath=None):
    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))


    if examples_filepath is None:
        examples_filepath = os.path.join(examples_dirpath,'clean-example-data.json')
    print('examples_filepath = {}'.format(examples_filepath))

    # Load the metric for squad v2
    # TODO metrics requires a download from huggingface, so you might need to pre-download and place the metrics within your container since there is no internet on the test server
    metrics_enabled = False  # turn off metrics for running on the test server
    if metrics_enabled:
        metric = datasets.load_metric('squad_v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    pytorch_model = torch.load(model_filepath, map_location=torch.device(device))

    # Inference the example images in data
    if examples_filepath is None:
        fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
        fns.sort()
        examples_filepath = fns[0]

    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    source_dataset = config['source_dataset']
    model_architecture = config['model_architecture']
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))

    # Load the examples
    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    source_dataset = source_dataset.split(':')[1]
    examples_filepath = os.path.join(simg_data_fo, source_dataset + '_data.json')
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))

    tokenizer = torch.load(tokenizer_filepath)

    task_type = None
    if 'ner_labels' in dataset.features:
        task_type='ner'
        #trojan_detector_func=trojan_detector_ner
        trojan_detector_func=trojan_detector_random
    elif 'question' in dataset.features:
        task_type='qa'
        trojan_detector_func=trojan_detector_qa
        trojan_detector_func=trojan_detector_random
    elif 'label' in dataset.features:
        task_type='sc'
        #trojan_detector_func=trojan_detector_sc
        trojan_detector_func=trojan_detector_random

    trojan_probability=trojan_detector_func(pytorch_model, tokenizer, dataset, scratch_dirpath)
    print('Trojan Probability: {}'.format(trojan_probability))
    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))


def trojan_detector_random(pytorch_model, tokenizer, dataset, scratch_dirpath):
    import random
    return 0.5+(random.random()-0.5)*0.2

def trojan_detector_qa(pytorch_model, tokenizer, dataset, scratch_dirpath):

    # Load the provided tokenizer
    # TODO: Use this method to load tokenizer on T&E server

    # TODO: This should only be used to test on personal machines, and should be commented out
    #  before submitting to evaluation server, use above method when submitting to T&E servers
    # model_architecture = config['model_architecture']
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture, use_fast=True)

    # insert_blanks = ['c_2', 'q_2', 't_2', 'c_6', 't_6']
    insert_blanks = ['q_3', 'c_8', 'ct_6', 'bt_4']
    # insert_blanks = ['bt_4']
    # insert_blanks = ['q_4', 'q_4', 't_4']
    rst_acc = list()
    record_data = dict()
    for ins in insert_blanks:
        print('tot len:', len(dataset))
        tokenized_dataset = tokenize_for_qa(tokenizer, dataset, insert_blanks=ins)
        tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions',
                                                    'end_positions', 'insert_idx'])

        ndata = len(tokenized_dataset)
        print('rst len:', ndata)
        ntr = min(int(ndata * 0.8), batch_size * 3)
        nte = min(ndata - ntr, batch_size * 6)
        nre = ndata - ntr - nte
        tr_dataset, te_dataset, _ = torch.utils.data.random_split(tokenized_dataset, [ntr, nte, nre])
        print('n_ntr:', len(tr_dataset))
        print('n_nte:', len(te_dataset))
        tr_dataloader = torch.utils.data.DataLoader(tr_dataset, batch_size=batch_size, shuffle=True)
        te_dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False)

        pytorch_model.eval()
        trigger, tr_acc, mean_loss = reverse_trigger(pytorch_model, tr_dataloader, insert_blanks=ins, tokenizer=tokenizer)
        te_acc = test_trigger(pytorch_model, te_dataloader, trigger, insert_blanks=ins)
        print(ins + ' test ASR: %2f%%' % (te_acc * 100))
        rst_acc.append(te_acc)

        record_data[ins] = {'te_acc':te_acc, 'mean_loss':mean_loss}

        # if te_acc > 0.95: break

    # tokenized_dataset.set_format()
    # predictions = utils_qa.postprocess_qa_predictions(dataset, tokenized_dataset, all_preds,
    # formatted_predictions = [
    #    {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    # ]
    # references = [{"id": ex["id"], "answers": ex['answers']} for ex in dataset]
    # print('Formatted Predictions:')
    # print(formatted_predictions)

    # if metrics_enabled:
    #     metrics = metric.compute(predictions=formatted_predictions, references=references)
    #     print("Metrics:")
    #     print(metrics)

    # Test scratch space
    # with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
    #     fh.write('this is a test')

    # trojan_probability = np.random.rand()

    # if te_acc > 0.95:
    #   trojan_probability = 1
    # else:
    trojan_probability = final_deal(record_data)
    # trojan_probability = max(rst_acc)

    #'''
    if not RELEASE:
        import pickle
        out_path = os.path.join(scratch_dirpath, 'record_data')
        with open(out_path + '.pkl', 'wb') as f:
            pickle.dump(record_data, f)
        print("write to ", out_path+'.pkl')
    #'''

    return trojan_probability




def configure(output_parameters_dirpath,
              configure_models_dirpath,
              parameter3):
    print('Using parameter3 = {}'.format(str(parameter3)))

    print('Configuring detector parameters with models from ' + configure_models_dirpath)

    os.makedirs(output_parameters_dirpath, exist_ok=True)

    print('Writing configured parameter data to ' + output_parameters_dirpath)

    arr = np.random.rand(100,100)
    np.save(os.path.join(output_parameters_dirpath, 'numpy_array.npy'), arr)

    with open(os.path.join(output_parameters_dirpath, "single_number.txt"), 'w') as fh:
        fh.write("{}".format(17))

    example_dict = dict()
    example_dict['keya'] = 2
    example_dict['keyb'] = 3
    example_dict['keyc'] = 5
    example_dict['keyd'] = 7
    example_dict['keye'] = 11
    example_dict['keyf'] = 13
    example_dict['keyg'] = 17

    with open(os.path.join(output_parameters_dirpath, "dict.json"), mode='w', encoding='utf-8') as f:
        f.write(jsonpickle.encode(example_dict, warn=True, indent=2))



if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.')
    parser.add_argument('--features_filepath', type=str, help='File path to the file where intermediate detector features may be written. After execution this csv file should contain a two rows, the first row contains the feature names (you should be consistent across your detectors), the second row contains the value for each of the column names.')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.')

    parser.add_argument('--round_training_dataset_dirpath', type=str, help='File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.', default=None)

    parser.add_argument('--metaparameters_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)
    parser.add_argument('--schema_filepath', type=str, help='Path to a schema file in JSON Schema format against which to validate the config file.', default=None)
    parser.add_argument('--learned_parameters_dirpath', type=str, help='Path to a directory containing parameter data (model weights, etc.) to be used when evaluating models.  If --configure_mode is set, these will instead be overwritten with the newly-configured parameters.')

    parser.add_argument('--configure_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a given location.', default=False, action="store_true")
    parser.add_argument('--configure_models_dirpath', type=str, help='Path to a directory containing models to use when in configure mode.')

    # these parameters need to be defined here, but their values will be loaded from the json file instead of the command line
    parser.add_argument('--parameter1', type=int, help='An example tunable parameter.')
    parser.add_argument('--parameter2', type=float, help='An example tunable parameter.')
    parser.add_argument('--parameter3', type=str, help='An example tunable parameter.')

    args = parser.parse_args()

    # Validate config file against schema
    if args.metaparameters_filepath is not None:
        if args.schema_filepath is not None:
            with open(args.metaparameters_filepath[0]()) as config_file:
                config_json = json.load(config_file)

            with open(args.schema_filepath) as schema_file:
                schema_json = json.load(schema_file)

            # this throws a fairly descriptive error if validation fails
            jsonschema.validate(instance=config_json, schema=schema_json)

    if not args.configure_mode:
        if (args.model_filepath is not None and
                args.tokenizer_filepath is not None and
                args.result_filepath is not None and
                args.scratch_dirpath is not None and
                args.examples_dirpath is not None and
                args.round_training_dataset_dirpath is not None and
                args.learned_parameters_dirpath is not None and
                args.parameter1 is not None and
                args.parameter2 is not None):

            trojan_detector(args.model_filepath,
                                    args.tokenizer_filepath,
                                    args.result_filepath,
                                    args.scratch_dirpath,
                                    args.examples_dirpath)
                                    #args.round_training_dataset_dirpath,
                                    #args.learned_parameters_dirpath,
                                    #args.parameter1,
                                    #args.parameter2,
                                    #args.features_filepath)
        else:
            print("Required Evaluation-Mode parameters missing!")
    else:
        if (args.learned_parameters_dirpath is not None and
                args.configure_models_dirpath is not None and
                args.parameter3 is not None):

            # all 3 example parameters will be loaded here, but we only use parameter3
            configure(args.learned_parameters_dirpath,
                      args.configure_models_dirpath,
                      args.parameter3)
        else:
            print("Required Configure-Mode parameters missing!")
