import configargparse
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, LlamaForCausalLM
from peft import LoraConfig, TaskType
import torch
from datasets import load_dataset, Dataset
import yaml

from trojai_mitigation_round.mitigations.llm_mitigation_ft import FineTuningTrojaiMitigationLLM


TASK_TYPE = TaskType.CAUSAL_LM
AUTO_MODEL_CLS = AutoModelForCausalLM


def prepare_mitigation():
    pass


def prepare_dataset(dataset_path):
    pass


def prepare_peft(lora_parameters):
    pass


def prepare_model(model, model_params):
    pass


if __name__ == "__main__":
    parser = configargparse.ArgParser(
        config_file_parser=configargparse.YAMLConfigFileParser
    )

    parser.add_argument(
        "--metaparameters",
        is_config_file_arg=True,
        type=str,
        required=True,
        help="Required metaparameters YAML file"
    )

    parser.add_argument("--model", type=str, help="The model we are conducting mitigation or testing on")
    parser.add_argument("--mitigate", action='store_true', help="Flag that asserts we are conducting mitigation")
    parser.add_argument("--test", action='store_true', help="Flag that asserts we are conducting testing of an existing model")
    
    parser.add_argument("--dataset", type=str, default=None, help="The dataset either given to us during mitigation (if at all), or if we are conducting testing, the dataset we will test on")
    parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="File path to the folder where a scratch space is located.")
    parser.add_argument('--output_dirpath', type=str, default="./out", help="File path to where the output model will be dumped")

    parser.add_argument("--lora_parameters", type=yaml.safe_load, help="Dummy CLI arg for the lora config parameters from the metaparameters.yml file. Not designed to be passed in directly")
    parser.add_argument("--model_parameters", type=yaml.safe_load, help="Dummy CLI arg for the model config parameters from the metaparameters.yml file. Not designed to be passed in directly")
    
    parser.add_argument('--batch_size', type=int, default=4, help="The batch size that the technique would use for dataloading")
    parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")


    args = parser.parse_args()
    assert args.mitigate ^ args.test, "Must choose only one of mitigate or test"

    model = prepare_model(args.model, args.model_parameters)
    peft_config = prepare_peft(args.lora_parameters)
    dataset = prepare_dataset(args.dataset)
    mitigation = prepare_mitigation()

    # assuming the tokenizer and model come from the same path for now
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    mitigated_model = mitigation.mitigate_model(
        model=model,
        collator=collator,
        peft_config=peft_config,
        dataset=dataset
    )

    mitigated_model.save_pretrained(args.output_dirpath)