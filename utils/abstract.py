""" Abstract class to be used by the detector
"""
import abc


class AbstractDetector(abc.ABC):
    def __init__(self, automatic_training: bool = False):
        """Initialize the detector and use the correct function for configuration

        Args
            automatic_training: bool - Wether the automatic configuration function
            should be used.
        """
        if automatic_training:
            self.configure = self.automatic_configure
        else:
            self.configure = self.manual_configure

    @abc.abstractmethod
    def manual_configure(self, models_dirpath):
        raise NotImplementedError("Method 'manual_configure' should be implemented")

    @abc.abstractmethod
    def automatic_configure(self, models_dirpath):
        raise NotImplementedError("Method 'automatic_configure' should be implemented")

    @abc.abstractmethod
    def infer(
        self,
        model_filepath,
        result_filepath,
        scratch_dirpath,
        examples_dirpath,
        round_training_dataset_dirpath,
    ):
        raise NotImplementedError("Method 'infer' should be implemented")
