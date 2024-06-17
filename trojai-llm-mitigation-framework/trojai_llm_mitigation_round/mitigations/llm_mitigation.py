from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig
from datasets import Dataset as HF_Dataset


class TrojAIMitigationLLM:
    def __init__(self, device, batch_size=32, num_workers=1, fp16=False, **kwargs):
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_token_length = kwargs.get('max_token_length', 512)
        self.fp16 = fp16
        for k,v in kwargs.items():
            setattr(self, k, v)

    def mitigate_model(self, model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        raise NotImplementedError
