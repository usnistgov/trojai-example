from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from trl import SFTTrainer
from datasets import Dataset as HF_Dataset
from itertools import chain



from .llm_mitigation import TrojAIMitigationLLM
from ..utils import print_summary


class FineTuningTrojaiMitigationLLM(TrojAIMitigationLLM):
    def __init__(self, lr, train_epochs, optim, device, batch_size=4, num_workers=1, fp16=False, **kwargs):
        super().__init__(device, batch_size, num_workers, fp16, **kwargs)
        self.lr = lr
        self.train_epochs = train_epochs
        self.optim = optim

    def mitigate_model(self,  model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        target_token_length = collator.tokenizer.model_max_length

        # TODO: Expose as parameter, and experiment multi-GPU and memory consumption
        if target_token_length > self.max_token_length:
            target_token_length = self.max_token_length        

        peft_model = get_peft_model(model, peft_config)

        training_args = TrainingArguments(output_dir="test_trainer",
                                            learning_rate=self.lr,
                                            num_train_epochs=self.train_epochs,
                                            optim=self.optim,
                                            evaluation_strategy="epoch", 
                                            per_device_train_batch_size=self.batch_size,
                                            per_device_eval_batch_size=self.batch_size,
                                            fp16=self.fp16)
        trainer = SFTTrainer(
            model=peft_model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            data_collator=collator,
            max_seq_length=self.max_token_length
        )
        print(f"Beginning Training with {len(dataset['train'])} examples")
        result = trainer.train()
        print_summary(result)
        return model
        