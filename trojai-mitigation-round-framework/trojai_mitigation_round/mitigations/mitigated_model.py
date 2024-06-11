from typing import OrderedDict, Dict
import torch

class TrojAIMitigatedModel:
    """This class wraps the output of a mitigation technique in a stardard form. We account for querying the model at one timepoint per prediction required.
    
    If a mitigation technique does not do any data pre or post processing at test time, but just changes the model weights, simply wrap your new state dict in this class:
        new_model = TrojAIMitigatedModel(new_state_dict)

    If a mitigation technique uses a pre/post process transform at test time, you can additionally pass either/or of those functions in and the defaults will be overwritten.
    The call signature of your pre/post process must match the signature of the base version exactly.

        def my_custom_preprocess_fn(x):
            ...
        def my_custom_postprocess_fn(logits, info):
            ...

        new_model = TrojAIMitigatedModel(
            new_state_dict, 
            custom_preprocess=my_custom_preprocess_fn,
            custom_postprocess=my_custom_postprocess_fn
        )

    """

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
    
    def postprocess_transform(self, logits: torch.tensor, info: Dict) -> torch.tensor:
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
    
    def __init__(self, model, custom_preprocess: callable=None, custom_postprocess: callable=None):
        """
        Args
            state_dict: A state dictionary that can be loaded by the original model
            custom_preprocess: An optional data preprocess function with the same interface as the default one implemented in the class.
            custom_postprocess: An optional data postprocess function with the same interface as the default one implemented in the class.
        """
        self.model = model

        if custom_preprocess:
            self.preprocess_transform = custom_preprocess

        if custom_postprocess:
            self.postprocess_transform = custom_postprocess


    
    