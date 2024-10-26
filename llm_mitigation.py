from transformers import AutoModel, AutoTokenizer, DataCollatorForLanguageModeling
from peft import LoraConfig
from datasets import Dataset as HF_Dataset


class TrojAIMitigationLLM:
    """
    This is the primary to abstract a TrojAI mitigation on a given Huggingface LLM model. 
    By default, any extra kwargs passed to init will be stored as a keyword attribute in the class.
    
    You may overwrite __init__ in your implementation, but please call super.__init__(device, batch_size, bf16)

    The only function required to implement is mitigate_model, which returns a Huggingface model. 
    """
    def __init__(self, batch_size=2, bf16=False, **kwargs):
        self.batch_size = batch_size
        self.max_token_length = kwargs.get('max_token_length', 512)
        self.bf16 = bf16
        for k,v in kwargs.items():
            setattr(self, k, v)

    def mitigate_model(self, model: AutoModel, collator: DataCollatorForLanguageModeling, peft_config: LoraConfig, dataset: HF_Dataset):
        raise NotImplementedError
