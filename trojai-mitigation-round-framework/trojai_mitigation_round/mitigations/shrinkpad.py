# Inspired from Backdoorbox
# https://github.com/THUYimingLi/BackdoorBox/blob/main/core/defenses/ShrinkPad.py

from typing import Dict
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from trojai_mitigation_round.mitigations import TrojAIMitigation, TrojAIMitigatedModel


class ShrinkPad(TrojAIMitigation):
    def __init__(
            self,
            input_shape,
            pad,
            batch_multiplier,
            **kwargs
        ):
        super().__init__(**kwargs)
        self._pad_bag = [transforms.Pad(padding=(i, j, pad - i, pad - j)) for i in range(pad + 1) for j in range(pad + 1)]
        self.shrinkprocess_transform = transforms.Compose(
            [
                transforms.Resize((input_shape[0] - pad, input_shape[1] - pad)),
                transforms.RandomChoice(self._pad_bag)
            ]
        )
        self.batch_multiplier = batch_multiplier

    def preprocess_transform(self, x):
        original_batch_size = x.shape[0]
        processed_x = self.shrinkprocess_transform(x)
        return processed_x, {"original_batch_size": original_batch_size}

    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        return TrojAIMitigatedModel(model.state_dict())
    