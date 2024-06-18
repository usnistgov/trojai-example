from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset as HF_Dataset
from itertools import chain



from .llm_mitigation import TrojAIMitigationLLM


class FineTuningTrojaiMitigationLLM(TrojAIMitigationLLM):
    def __init__(self, device, batch_size=4, num_workers=1, fp16=False, **kwargs):
        super().__init__(device, batch_size, num_workers, fp16, **kwargs)

    def mitigate_model(self,  model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        def tokenize_fn(example):
            return collator.tokenizer(" ".join(example['answers']['text']))
        dataset = dataset.map(tokenize_fn, remove_columns=dataset['train'].column_names, num_proc=self.num_workers)

        target_token_length = collator.tokenizer.model_max_length

        # TODO: Expose as parameter, and experiment multi-GPU and memory consumption
        if target_token_length > self.max_token_length:
            target_token_length = self.max_token_length        

        # Copied from llm-pretrain-april2024
        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
            # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
            total_length = (total_length // target_token_length) * target_token_length
            # Split by chunks of max_len.
            result = {
                k: [t[i: i + target_token_length] for i in range(0, total_length, target_token_length)]
                for k, t in concatenated_examples.items()
            }
            result["labels"] = result["input_ids"].copy()
            return result
        
        dataset = dataset.map(group_texts, batched=True, num_proc=self.num_workers)
        peft_model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(output_dir="test_trainer", 
                                            evaluation_strategy="epoch", 
                                            per_device_train_batch_size=self.batch_size,
                                            per_device_eval_batch_size=self.batch_size,
                                            fp16=self.fp16)
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
        )
        print(f"Beginning Training with {len(dataset['train'])} examples")
        trainer.train()
        return model
        