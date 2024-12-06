import pathlib
import sys
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig, BitsAndBytesConfig

class CodeLlamaPreprocessor():
    TOKEN_EMBED_DIMS = 4096

    def __init__(self, model_path: str, *, random_seed: int = 42, embedding_size: int = 256, max_tokens: int = 8192, device: str='cpu', bnb_config: Optional[BitsAndBytesConfig] = None):       
        if bnb_config is None:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=False,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        
        self._random_seed = random_seed
        self._embedding_size = embedding_size
        self._max_tokens = max_tokens
        self._projection_matrix = None
        self._device = device
        self._model_path = model_path
        self._bnb_config = bnb_config
        self._config = AutoConfig.from_pretrained(self._model_path)
        self._model = AutoModel.from_pretrained(
                self._model_path,
                config=self._config,
                quantization_config=self._bnb_config
            ).to(self._device)
        self._model.eval()

        self._tokenizer = AutoTokenizer.from_pretrained(model_path, load_in_8bit=True)
        self._tokenizer.pad_token = self._tokenizer.eos_token

    def __call__(self, text: Sequence[str]):
        if self._projection_matrix is None:
            raise RuntimeError("CodeLlamaPreprocessor is a context manager, so you must use it inside a with statement")

        # Apply the tokenizer
        tokens = self._tokenizer(text, padding='max_length', truncation=True, return_tensors='pt', max_length=self._max_tokens)
        num_tokens = np.sum(np.array(tokens['attention_mask']))

        # Invoke the LLM model
        inputs = {
            key: value.to(self._device) for key, value in tokens.items()
        }
        with torch.no_grad():
            result = self._model(**inputs)

        # Trim the result and projection matrix to match the number of tokens
        result = result['last_hidden_state'][0, :num_tokens, :].view(-1).to(torch.half)
        partial_projection = self._projection_matrix[:num_tokens * self.TOKEN_EMBED_DIMS, :]

        # Embed using random projections and yield the result
        return result @ partial_projection

    def __enter__(self):
        input_dim = self._max_tokens * self.TOKEN_EMBED_DIMS
        output_dim = self._embedding_size

        # Generate the random projection matrix
        rng_state = np.random.get_state()
        np.random.seed(self._random_seed)
        projection_matrix = np.random.randn(input_dim, output_dim).astype(np.float16)
        np.random.set_state(rng_state)

        # Move it into torch and onto the correct device
        self._projection_matrix = torch.from_numpy(projection_matrix).to(self._device)

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Free the projection matrix memory
        self._projection_matrix = None

