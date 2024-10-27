from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer, SFTConfig
from datasets import Dataset as HF_Dataset

import datetime

from llm_mitigation import TrojAIMitigationLLM
from utils import print_summary

class debugOutput():
    def __init__(self, tokenizer):
        self.all_text = []
        self.tokenizer = tokenizer

    def __call__(self, output, compute_result=False):
        # self.all_text.append(results)
        if compute_result:
            results = self.tokenizer.decode(output[0][0].argmax(axis=1))
            output[1][0][output[1][0] == -100] = self.tokenizer.bos_token_id
            label = self.tokenizer.decode(output[1][0])
            # print(f"Output Prediction: {results} True Label: {label}")
            return {'text_results': 1}
        else:
            return {'text_results':1}

class FineTuningTrojaiMitigationLLM(TrojAIMitigationLLM):
    def __init__(self, lr, train_epochs, optim, batch_size=4, bf16=False, **kwargs):
        super().__init__(batch_size, bf16, **kwargs)
        self.lr = lr
        self.train_epochs = train_epochs
        self.optim = optim

    def mitigate_model(self,  model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        target_token_length = collator.tokenizer.model_max_length
        # TODO: Expose as parameter, and experiment multi-GPU and memory consumption
        if target_token_length > self.max_token_length:
            target_token_length = self.max_token_length        
        peft_model = get_peft_model(model, peft_config)
        now = datetime.datetime.now()
        formatted  = now.strftime("%Y%m%d%H%M%S.%f")[:-3]
        training_args = SFTConfig(output_dir=f"test_trainer_{formatted}",
                                            learning_rate=self.lr,
                                            num_train_epochs=self.train_epochs,
                                            optim=self.optim,
                                            eval_strategy="epoch",
                                            save_strategy="no",
                                            per_device_train_batch_size=self.batch_size,
                                            per_device_eval_batch_size=self.batch_size,
                                            bf16=self.bf16,
                                            logging_strategy="epoch",
                                            max_seq_length = target_token_length,
                                            batch_eval_metrics=True,
                                            )

        debug_output = debugOutput(collator.tokenizer)
        
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            # compute_metrics=debug_output,
        )
        
        print(f"Beginning Training with {len(dataset['train'])} examples")
        result = trainer.train()
        print_summary(result)
        return model
        