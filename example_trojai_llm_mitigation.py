import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, TaskType
import torch
import jsonschema
import json
from datasets import load_dataset, Dataset
from trl import DataCollatorForCompletionOnlyLM

from finetuning import FineTuningTrojaiMitigationLLM
import utils



INSTRUCTION_TEMPLATE_LOOKUP = {
        'google/gemma-2-2b-it': '<start_of_turn>user',
        'google/gemma-2-9b-it': '<start_of_turn>user',
        'google/gemma-2b-it': '<start_of_turn>user',
        'google/gemma-7b-it': '<start_of_turn>user',
        'meta-llama/Llama-2-7b-chat-hf': '<s>[INST]',
        'meta-llama/Meta-Llama-3-8B-Instruct': '<|begin_of_text|><|start_header_id|>user<|end_header_id|>',
        "mistralai/Mixtral-8x7B-Instruct-v0.1": '[INST]',
        'meta-llama/Meta-Llama-3.1-8B-Instruct': '<|begin_of_text|><|start_header_id|>user<|end_header_id|>',
        'meta-llama/Llama-3.2-1B-Instruct': '<|begin_of_text|><|start_header_id|>user<|end_header_id|>'
    }
RESPONSE_TEMPLATE_LOOKUP = {
    'google/gemma-2-2b-it': '<start_of_turn>model\n',
    'google/gemma-2-9b-it': '<start_of_turn>model\n',
    'google/gemma-2b-it': '<start_of_turn>model\n',
    'google/gemma-7b-it': '<start_of_turn>model\n',
    'meta-llama/Llama-2-7b-chat-hf': '[/INST]',
    'meta-llama/Meta-Llama-3-8B-Instruct': '<|start_header_id|>assistant<|end_header_id|>',
    "mistralai/Mixtral-8x7B-Instruct-v0.1": '[/INST]',
    'meta-llama/Meta-Llama-3.1-8B-Instruct': '<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>',
    'meta-llama/Llama-3.2-1B-Instruct': '<|begin_of_text|><|start_header_id|>assistant<|end_header_id|>'
}

def prepare_mitigation(config_json, argv):
    mitigation = FineTuningTrojaiMitigationLLM(
        lr=config_json['learning_rate'],
        train_epochs=config_json['num_train_epochs'],
        optim=config_json['optim'],
        device=argv.device,
        batch_size=config_json['batch_size'],
        num_workers=argv.num_workers,
        bf16=config_json['bf16'],
        max_token_length=config_json['max_token_length'],
    )
    return mitigation


def prepare_dataset(dataset_path, split='train'):
    num_split = -1
    dataset = load_dataset('json', data_files=[dataset_path], split=f'{split}[:{num_split}]')
    dataset = dataset.train_test_split(test_size=0.2)
    
    # # Note: Debug hack to make the dataset smaller for debugging
    # dataset['train'] = Dataset.from_dict(dataset['train'][:10])
    # dataset['test'] = Dataset.from_dict(dataset['test'][:10])
    return dataset


def prepare_peft(lora_parameters):
    return LoraConfig(task_type=TaskType.CAUSAL_LM, **lora_parameters)


def prepare_model_and_tokenizer(model_path):
    torch_dtype = torch.bfloat16
    # print(torch.__version__)
    # import transformers
    # print(transformers.__version__)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True,  low_cpu_mem_usage=True, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    ## Note: Commenting out due to the fact it changes the embedding layer of the model
    # if tokenizer.chat_template is None:
    #  model, tokenizer = setup_chat_format(model, tokenizer, resize_to_multiple_of=8)
    # tokenizer.padding_side = 'right'

    # if 'gemma' in model_path.lower():
    #     tokenizer.add_bos_token = False
    #     tokenizer.add_eos_token = True

    if 'llama' in model_path.lower():
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        tokenizer.pad_token = tokenizer.eos_token

    if 'mistral' in model_path.lower():
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    return model, tokenizer


def run_mitigate_mode(argv):

    # Validate config file against schema
    with open(argv.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(argv.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    model_name = argv.model
    if os.path.exists(argv.model):
        with open(os.path.join(argv.model, 'config.json')) as config_file:
            reduced_round_config_json = json.load(config_file)
        model_name = reduced_round_config_json['model_architecture']

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)
    model, tokenizer = prepare_model_and_tokenizer(argv.model)
    peft_config = prepare_peft(config_json['lora_parameters'])
    dataset = prepare_dataset('example_data.json')
    print("Finished prepping dataset")
    utils.print_gpu_utilization()
    ## This is for Llama:
    # INSTRUCTION_TEMPLATE = '<|begin_of_text|><|start_header_id|>user<|end_header_id|>'
    # RESPONSE_TEMPLATE = '<|start_header_id|>assistant<|end_header_id|>'
    # INSTRUCTION_TEMPLATE = '<start_of_turn>user<end_of_turn>'
    # RESPONSE_TEMPLATE = '<end_of_turn>model<end_of_turn>'
    response = RESPONSE_TEMPLATE_LOOKUP[model_name]
    instruction = INSTRUCTION_TEMPLATE_LOOKUP[model_name]
    collator = DataCollatorForCompletionOnlyLM(tokenizer=tokenizer, response_template=response, instruction_template=instruction)

    mitigation = prepare_mitigation(config_json, argv)
    mitigated_model = mitigation.mitigate_model(
        model=model,
        collator=collator,
        peft_config=peft_config,
        dataset=dataset
    )
    mitigated_model.save_pretrained(argv.output_dirpath)


def run_test_mode(argv):
    # Validate config file against schema
    with open(argv.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(argv.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    pass


if __name__ == "__main__":
    from argparse import ArgumentParser
    
    parser = ArgumentParser(description='Parser for the LLM mitigation round with two modes of operation, mitigate and test')
    parser.set_defaults(func=lambda argv: parser.print_help())
    subparser = parser.add_subparsers(dest='cmd', required=True)

    # Mitigation arguments
    mitigate_parser = subparser.add_parser('mitigate', help='Generates a mitigated model')

    mitigate_parser.add_argument("--metaparameters_filepath","-j", type=str, required=True, help="Path JSON file containing values of tunable parameters based on json schema")
    mitigate_parser.add_argument("--schema_filepath", "-s", type=str, help="Path to a schema file in JSON Schema format against which to validate the metaparameters file.", required=True)
    mitigate_parser.add_argument('--model', type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Huggingface model")
    mitigate_parser.add_argument('--output_dirpath', type=str, default="./out", help="The directory path to where the output will be dumped")
    mitigate_parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="The directory where a scratch space is located.")
    mitigate_parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    mitigate_parser.add_argument("--round_training_dataset_dirpath", type=str, help="File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.", default=None)

    # Test arguments
    test_parser = subparser.add_parser('test', help='Tests a mitigated model with example data')

    test_parser.add_argument("--metaparameters_filepath", "-j", type=str, required=True, help="Path JSON file containing values of tunable parameters based on json schema")
    test_parser.add_argument("--schema_filepath", "-s", type=str, help="Path to a schema file in JSON Schema format against which to validate the metaparameters file.", required=True)
    test_parser.add_argument('--model', type=str, default="./model.pt", help="File path to the mitigated model that will be tested")
    test_parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="The directory where a scratch space is located.")
    test_parser.add_argument('--output_dirpath', type=str, default="./out", help="The directory path to where the output will be dumped")
    test_parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    test_parser.add_argument("--round_training_dataset_dirpath", type=str, help="File path to the directory containing id-xxxxxxxx models of the current rounds training dataset.", default=None)

    # Setup default function to call for mitigate/test
    mitigate_parser.set_defaults(func=run_mitigate_mode)
    test_parser.set_defaults(func=run_test_mode)

    argv = parser.parse_args()

    # Call appropriate function
    argv.func(argv)