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

import warnings

import utils_qa

warnings.filterwarnings("ignore")


# The inferencing approach was adapted from: https://github.com/huggingface/transformers/blob/master/examples/pytorch/question-answering/run_qa.py
def tokenize_for_qa(tokenizer, dataset):
    column_names = dataset.column_names
    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    
    # Padding side determines if we do (question|context) or (context|question).
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(tokenizer.model_max_length, 384)
    
    if 'mobilebert' in tokenizer.name_or_path:
        max_seq_length = tokenizer.max_model_input_sizes[tokenizer.name_or_path.split('/')[1]]
    
    # Training preprocessing
    def prepare_train_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.
        
        pad_to_max_length = True
        doc_stride = 128
        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            examples[context_column_name if pad_on_right else question_column_name],
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
        # The offset mappings will give us a map from token to character position in the original context. This will
        # help us compute the start_positions and end_positions.
        # offset_mapping = tokenized_examples.pop("offset_mapping")
        # offset_mapping = copy.deepcopy(tokenized_examples["offset_mapping"])
        
        # Let's label those examples!
        tokenized_examples["start_positions"] = []
        tokenized_examples["end_positions"] = []
        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []
        
        for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
            # We will label impossible answers with the index of the CLS token.
            input_ids = tokenized_examples["input_ids"][i]
            cls_index = input_ids.index(tokenizer.cls_token_id)
            
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            
            context_index = 1 if pad_on_right else 0
            
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            answers = examples[answer_column_name][sample_index]
            # One example can give several spans, this is the index of the example containing this span of text.
            tokenized_examples["example_id"].append(examples["id"][sample_index])
            
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
            
            # This is for the evaluation side of the processing
            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]
        
        return tokenized_examples
    
    # Create train feature from dataset
    tokenized_dataset = dataset.map(
        prepare_train_features,
        batched=True,
        num_proc=1,
        remove_columns=column_names,
        keep_in_memory=True)
    
    if len(tokenized_dataset) == 0:
        print(
            'Dataset is empty, creating blank tokenized_dataset to ensure correct operation with pytorch data_loader formatting')
        # create blank dataset to allow the 'set_format' command below to generate the right columns
        data_dict = {'input_ids': [],
                     'attention_mask': [],
                     'token_type_ids': [],
                     'start_positions': [],
                     'end_positions': []}
        tokenized_dataset = datasets.Dataset.from_dict(data_dict)
    return tokenized_dataset


def example_trojan_detector(model_filepath, tokenizer_filepath, result_filepath, scratch_dirpath, examples_dirpath):

    print('model_filepath = {}'.format(model_filepath))
    print('tokenizer_filepath = {}'.format(tokenizer_filepath))
    print('result_filepath = {}'.format(result_filepath))
    print('scratch_dirpath = {}'.format(scratch_dirpath))
    print('examples_dirpath = {}'.format(examples_dirpath))

    # Load the metric for squad v2
    # TODO metrics requires a download from huggingface, so you might need to pre-download and place the metrics within your container since there is no internet on the test server
    metrics_enabled = False  # turn off metrics for running on the test server
    if metrics_enabled:
        metric = datasets.load_metric('squad_v2')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the classification model and move it to the GPU
    pytorch_model = torch.load(model_filepath, map_location=torch.device(device))

    # Inference the example images in data
    fns = [os.path.join(examples_dirpath, fn) for fn in os.listdir(examples_dirpath) if fn.endswith('.json')]
    fns.sort()
    examples_filepath = fns[0]

    # load the config file to retrieve parameters
    model_dirpath, _ = os.path.split(model_filepath)
    with open(os.path.join(model_dirpath, 'config.json')) as json_file:
        config = json.load(json_file)
    print('Source dataset name = "{}"'.format(config['source_dataset']))
    if 'data_filepath' in config.keys():
        print('Source dataset filepath = "{}"'.format(config['data_filepath']))

    # Load the examples
    # TODO The cache_dir is required for the test server since /home/trojai is not writable and the default cache locations is ~/.cache
    dataset = datasets.load_dataset('json', data_files=[examples_filepath], field='data', keep_in_memory=True, split='train', cache_dir=os.path.join(scratch_dirpath, '.cache'))

    # Load the provided tokenizer
    # TODO: Use this method to load tokenizer on T&E server
    tokenizer = torch.load(tokenizer_filepath)

    # TODO: This should only be used to test on personal machines, and should be commented out
    #  before submitting to evaluation server, use above method when submitting to T&E servers
    # model_architecture = config['model_architecture']
    # tokenizer = transformers.AutoTokenizer.from_pretrained(model_architecture, use_fast=True)

    tokenized_dataset = tokenize_for_qa(tokenizer, dataset)
    dataloader = torch.utils.data.DataLoader(tokenized_dataset, batch_size=1)
    
    pytorch_model.eval()
    all_preds = None

    tokenized_dataset.set_format('pt', columns=['input_ids', 'attention_mask', 'token_type_ids', 'start_positions', 'end_positions'])

    with torch.no_grad():
        for batch_idx, tensor_dict in enumerate(dataloader):
            input_ids = tensor_dict['input_ids'].to(device)
            attention_mask = tensor_dict['attention_mask'].to(device)
            token_type_ids = tensor_dict['token_type_ids'].to(device)
            start_positions = tensor_dict['start_positions'].to(device)
            end_positions = tensor_dict['end_positions'].to(device)
            
            if 'distilbert' in pytorch_model.name_or_path or 'bart' in pytorch_model.name_or_path:
                model_output_dict = pytorch_model(input_ids,
                                          attention_mask=attention_mask,
                                          start_positions=start_positions,
                                          end_positions=end_positions)
            else:
                model_output_dict = pytorch_model(input_ids,
                                          attention_mask=attention_mask,
                                          token_type_ids=token_type_ids,
                                          start_positions=start_positions,
                                          end_positions=end_positions)
                
            batch_train_loss = model_output_dict['loss'].detach().cpu().numpy()
            start_logits = model_output_dict['start_logits'].detach().cpu().numpy()
            end_logits = model_output_dict['end_logits'].detach().cpu().numpy()

            logits = (start_logits, end_logits)
            all_preds = logits if all_preds is None else transformers.trainer_pt_utils.nested_concat(all_preds, logits,
                                                                                                     padding_index=-100)

    tokenized_dataset.set_format()

    predictions = utils_qa.postprocess_qa_predictions(dataset, tokenized_dataset, all_preds, version_2_with_negative=True)
    formatted_predictions = [
        {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
    ]
    references = [{"id": ex["id"], "answers": ex['answers']} for ex in dataset]

    print('Formatted Predictions:')
    print(formatted_predictions)

    if metrics_enabled:
        metrics = metric.compute(predictions=formatted_predictions, references=references)
        print("Metrics:")
        print(metrics)
    
    # Test scratch space
    with open(os.path.join(scratch_dirpath, 'test.txt'), 'w') as fh:
        fh.write('this is a test')

    trojan_probability = np.random.rand()
    print('Trojan Probability: {}'.format(trojan_probability))

    with open(result_filepath, 'w') as fh:
        fh.write("{}".format(trojan_probability))

def self_tune(output_config_filepath, 
              output_schema_filepath, 
              tuning_models_dirpath):

    print('Tuning parameters with ' + tuning_models_dirpath)

    tuned_parameters = {
                           "parameter1": 10,
                           "parameter2": 3.4,
                           "parameter3": "AdamW"
                       }

    tuned_json_schema = {
                            "$schema": "http://json-schema.org/draft-07/schema#",
                            "title": "Tunable Parameters",
                            "required": [ "parameter1", "parameter2", "parameter3" ],
                            "type": "object",
                            "properties": {
                                "parameter1": {
                                    "type": "integer",
                                    "minimum": 0
                                },
                                "parameter2": {
                                     "type": "number",
                                     "minimum": -1.2,
                                     "maximum": 3.4
                                },
                                "parameter3": {
                                    "enum": [ "Adam", "AdamW", "SGD" ]
                                }
                            }
                        }

    jsonschema.validate(instance = tuned_parameters, schema = tuned_json_schema)

    with open(output_config_filepath, 'w') as output_config_file:
        json.dump(tuned_parameters, output_config_file, indent=2)

    with open(output_schema_filepath, 'w') as output_schema_file:
        json.dump(tuned_json_schema, output_schema_file, indent=2)

if __name__ == "__main__":
    from jsonargparse import ArgumentParser, ActionConfigFile

    parser = ArgumentParser(description='Fake Trojan Detector to Demonstrate Test and Evaluation Infrastructure.')
    parser.add_argument('--model_filepath', type=str, help='File path to the pytorch model file to be evaluated.', default='./model/model.pt')
    parser.add_argument('--tokenizer_filepath', type=str, help='File path to the pytorch model (.pt) file containing the correct tokenizer to be used with the model_filepath.', default='./tokenizers/google-electra-small-discriminator.pt')
    parser.add_argument('--result_filepath', type=str, help='File path to the file where output result should be written. After execution this file should contain a single line with a single floating point trojan probability.', default='./output.txt')
    parser.add_argument('--scratch_dirpath', type=str, help='File path to the folder where scratch disk space exists. This folder will be empty at execution start and will be deleted at completion of execution.', default='./scratch')
    parser.add_argument('--examples_dirpath', type=str, help='File path to the directory containing json file(s) that contains the examples which might be useful for determining whether a model is poisoned.', default='./model/example_data')

    parser.add_argument('--parameter1', type=int, help='An example tunable parameter.')
    parser.add_argument('--parameter2', type=float, help='An example tunable parameter.')
    parser.add_argument('--parameter3', type=str, help='An example tunable parameter.')
    parser.add_argument('--config_filepath', help='Path to JSON file containing values of tunable paramaters to be used when evaluating models.', action=ActionConfigFile)

    parser.add_argument('--self_tune_mode', help='Instead of detecting Trojans, set values of tunable parameters and write them to a config file.', default=False, action="store_true")
    parser.add_argument('--output_config_filepath', type=str, help='Path to a JSON file into which to write tuned values of parameters when in self tune mode.')
    parser.add_argument('--output_schema_filepath', type=str, help='Path to a JSON Schema file into which to write the schema for the generated config file.')
    parser.add_argument('--tuning_models_dirpath', type=str, help='Path to a directory containing models to use when in self tune mode.')

    args = parser.parse_args()

    print('Parameter 1: ' + str(args.parameter1))
    print('Parameter 2: ' + str(args.parameter2))
    print('Parameter 3: ' + str(args.parameter3))

    if not args.self_tune_mode:
        example_trojan_detector(args.model_filepath, 
                                args.tokenizer_filepath, 
                                args.result_filepath, 
                                args.scratch_dirpath, 
                                args.examples_dirpath)
    else:
        self_tune(args.output_config_filepath, 
                  args.output_schema_filepath, 
                  args.tuning_models_dirpath)
