from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling
from peft import LoraConfig, TaskType
import torch
from datasets import load_dataset, Dataset
import yaml
from trl import setup_chat_format

from trojai_llm_mitigation_round.mitigations.finetuning import FineTuningTrojaiMitigationLLM
from trojai_llm_mitigation_round.utils import print_gpu_utilization


TASK_TYPE = TaskType.CAUSAL_LM
AUTO_MODEL_CLS = AutoModelForCausalLM


def prepare_mitigation(args):
    mitigation = FineTuningTrojaiMitigationLLM(
        lr=args.learning_rate,
        train_epochs=args.num_train_epochs,
        optim=args.optim,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        bf16=args.bf16
    )
    return mitigation


def prepare_dataset(dataset_path, split='train'):
    num_split = -1
    dataset = load_dataset(dataset_path, split=f'{split}[:{num_split}]')
    dataset = dataset.train_test_split(test_size=0.2)
    
    # Note: Debug hack to make the dataset smaller for debugging
    dataset['train'] = Dataset.from_dict(dataset['train'][:1000])
    dataset['test'] = Dataset.from_dict(dataset['test'][:1000])

    return dataset


def prepare_peft(lora_parameters):
    return LoraConfig(task_type=TASK_TYPE, **lora_parameters)


def prepare_model_and_tokenizer(model_path, model_params, device):
    if model_params['model_dtype'] == 'float16':
        dtype = torch.float16
    elif model_params['model_dtype'] == 'bfloat16':
        dtype = torch.bfloat16
    else: 
        dtype = torch.float32
    
    model = AUTO_MODEL_CLS.from_pretrained(model_path, device_map=device, torch_dtype=dtype, attn_implementation='flash_attention_2' if model_params['use_flash_attn2'] else None)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    if tokenizer.chat_template is None:
        model, tokenizer = setup_chat_format(model, tokenizer, resize_to_multiple_of=8)
    
    tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = 'left'

    if 'gemma' in model_path.lower():
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = True

    if 'llama' in model_path.lower():
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    if 'mistral' in model_path.lower():
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    return model, tokenizer


if __name__ == "__main__":
    import configargparse
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    parser.add_argument(
        "--metaparameters",
        is_config_file_arg=True,
        type=str,
        required=True,
        help="Required metaparameters YAML file"
    )

    parser.add_argument("--model", type=str, help="The model we are conducting mitigation or testing on")
    parser.add_argument("--mitigate", action='store_true', help="Pass this flag if you want to conduct your mitigation technique on a saved model")
    parser.add_argument("--test", action='store_true', help="Pass this flag in if you want to do test/inference on the model")

    parser.add_argument("--dataset", type=str, default=None, help="The dataset either given to us during mitigation (if at all), or if we are conducting testing, the dataset we will test on")
    parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="File path to the folder where a scratch space is located.")
    parser.add_argument('--output_dirpath', type=str, default="./out", help="File path to where the output model will be dumped")

    parser.add_argument("--lora_parameters", type=yaml.safe_load, help="Dummy CLI arg for the lora config parameters from the metaparameters.yml file. Not designed to be passed in directly")
    parser.add_argument("--model_parameters", type=yaml.safe_load, help="Dummy CLI arg for the model config parameters from the metaparameters.yml file. Not designed to be passed in directly")
    
    # Example performer specific args
    parser.add_argument("--learning_rate", type=float, help="Learning rate to use for fine tuning")
    parser.add_argument("--num_train_epochs", type=float, help="Number of epochs to run fine tuning for")
    parser.add_argument("--optim", type=str, help="Optimizer to use for fine tuning")

    parser.add_argument('--batch_size', type=int, default=4, help="The batch size that the technique would use for dataloading")
    parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")
    parser.add_argument('--bf16', action='store_true', help="Whether or not to use bf16")
    # from argparse import ArgumentParser
    
    # parser = ArgumentParser(description='Parser for the LLM mitigation round with two modes of operation, mitigate and test')
    # parser.set_defaults(func=lambda args: parser.print_help())
    # subparser = parser.add_subparsers(dest='cmd', required=True)

    args = parser.parse_args()
    model, tokenizer = prepare_model_and_tokenizer(args.model, args.model_parameters, args.device)
    peft_config = prepare_peft(args.lora_parameters)
    dataset = prepare_dataset(args.dataset)
    print("Finished prepping dataset")
    print_gpu_utilization()
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if args.mitigate:
        mitigation = prepare_mitigation(args)
        mitigated_model = mitigation.mitigate_model(
            model=model,
            collator=collator,
            peft_config=peft_config,
            dataset=dataset
        )
        mitigated_model.save_pretrained(args.output_dirpath)

    # TODO: inferencing pipeline
    if args.test:
        pass
