from typing import Dict
from pathlib import Path

import torchvision
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm

from trojai_mitigation_round.mitigations.mitigation import TrojAIMitigation
from trojai_mitigation_round.mitigations.mitigated_model import TrojAIMitigatedModel

class FineTuningTrojai(TrojAIMitigation):
    def __init__(self, device, loss_cls, optim_cls, lr, epochs, ckpt_dir="./ckpts", ckpt_every=0, batch_size=32, num_workers=1, **kwargs):
        super().__init__(device, batch_size, num_workers, **kwargs)
        self._optimizer_class = optim_cls
        self._loss_cls = loss_cls
        self.lr = lr
        self.epochs = epochs
        self.ckpt_dir = ckpt_dir
        self.ckpt_every = ckpt_every


    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        """
        Args:
            model: the model to repair
            dataset: a dataset of examples
        Returns:
            mitigated_model: A TrojAIMitigatedModel object corresponding to new model weights and a pre/post processing techniques
        """
        model = model.to(self.device)
        model.train()
        optim = self._optimizer_class(model.parameters(), lr=self.lr)
        loss_fn = self._loss_cls()
        trainloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True
        )
        
        for i in range(self.epochs):
            pbar = tqdm(trainloader)
        
            for x, y in pbar:
                x = x.to(self.device)
                y = y.to(self.device)
                optim.zero_grad()
                pred = model(x)
                loss = loss_fn(pred, y)
                loss.backward()
                optim.step()
                pbar.set_description(f"Epoch: {i} | Loss: {loss}")
            
            if self.ckpt_every != 0 and i % self.ckpt_every == 0:
                ckpt_path = Path(self.ckpt_dir)
                ckpt_path.mkdir(exist_ok=True)
                torch.save({
                    'epoch': i,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optim.state_dict(),
                }, ckpt_path / Path(f"ft_ckpt_epoch{i + 1}.ckpt"))
                print(f"Saved ckpt to {ckpt_path / Path(f'ft_ckpt_epoch{i + 1}.ckpt')}")
            
            

        return TrojAIMitigatedModel(model)
