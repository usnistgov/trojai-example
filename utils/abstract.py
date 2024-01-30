""" Abstract class to be used by the detector
"""
import abc


class AbstractDetector(abc.ABC):

    def configure(self, models_dirpath: str, automatic_training: bool):
        if automatic_training:
            self.automatic_configure(models_dirpath)
        else:
            self.manual_configure(models_dirpath)

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
