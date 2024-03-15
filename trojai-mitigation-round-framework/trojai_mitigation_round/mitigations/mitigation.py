from typing import OrderedDict, Dict
import torch
from torch.utils.data import Dataset
from .mitigated_model import TrojAIMitigatedModel

class TrojAIMitigation:
    """This is the main class to abstract for a TrojAI Mitigation. 
    By default, any extra keyword arguments passed to __init__ will be stored as keyword attributes in the class. i.e. if you declare:
    mitigation = TrojAIMitigation(device, batch_size, num_workers, kwarg1=1, kwarg2=2), you will have access to mitigation.kwarg1 and mitigation.kwarg2
    
    You may overwrite __init__ but please call super.__init__(device, batch_size, num_workers) 
    
    The only function required to implement is mitigate_model. This function must return a TrojAIMitigatedModel (as definined in ./mitigated_model.py), 
    If a mitigation technique does not do any data pre or post processing at test time, but just changes the model weights, simply wrap your new state dict in this class:
        new_model = TrojAIMitigatedModel(new_state_dict)
        
    If your mitigation technique does require pre and post processing, abstract this class and overwrite the preprocess_data and postprocess_data functions:
        class NewMitigationModelClass(TrojAIMitigatedModel):
            def preprocess_data(self, x):
                # Your new logic here
            def postprocess_data(self, logits):
                # Your new logic here
            
    The Final mitigated model will then be returned as NewMitigationModelClass(new_state_dict)    
    """
    def __init__(self, device, batch_size=32, num_workers=1, **kwargs):
        """
        Args
            device: Which device for the mitigation technique use. Either 'cuda:X' or 'cpu'
            batch_size: The batch size for the technique to use
            num_workers: the number of CPU processes to use to load data
        """
        self.device = device
        self.batch_size = batch_size
        self.num_workers = num_workers
        for k,v in kwargs.items():
            setattr(self, k, v)

    def preprocess_transform(self, x: torch.tensor) -> tuple[torch.tensor, Dict]:
        """
        This is the default preprocess of a mitigation method. If your  mitigation technique defines a preprocess transform,
        this function will be overwritten by that. Otherwise, this default is used. 
        
        You may change the batch size to allow for data augmentations being queries as multiple 
        points. Additionally, any additional information required to post process the results back to one prediction per original data point acn be passed as an additional dictionary

        Args
            x: torch:tensor shape: (batch_size, channels, height, width) corresponding to input data of size . Will be on device: 'cpu'

        Returns:
            x: torch tensor corresponding to input data of size (intermediary_batch_size, channels, height, width). 
            info: dictionary corresponding to metadata that needs to be passed to post process
    
        """
        return x, {}

    def postprocess_transform(self, logits: torch.tensor, info: Dict={}) -> torch.tensor:
        """
        This is the default postprocess of a mitigation method. If your mitigation technique defines a postprocess transform, this
        function will be overwritten by that. Otherwise, this default is used.

        If you abstracted preprocess_data to change the batch_size to the model, you must abstract this function to return the logits to the same batch size as
        the original queried model

        Args 
            logits: torch:tensor shape: (intermediary_batch_size, num_classes) corresponding to the logits of the inpu data
            info: dictionary corresponding to metadata passed from preprocess

        Returns:
            logits : torch:tensor shape: (batch_size, num_classes) corresponding to the logits for original batch of data
        """
        return logits 

    def mitigate_model(self, model: torch.nn.Module, dataset: Dataset) -> TrojAIMitigatedModel:
        """
        Args:
            model: the model to repair
            dataset: a dataset that contains sample data that may or may not be clean/poisoned.
        Returns:
            mitigated_model: A TrojAIMitigatedModel object corresponding to new model weights and a pre/post processing techniques
        """
        raise NotImplementedError