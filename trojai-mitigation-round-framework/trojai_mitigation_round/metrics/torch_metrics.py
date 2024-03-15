from torchmetrics.classification import Accuracy, F1Score

class TorchMetricsWrapper:
    def __init__(self, torch_metric, name=''):
        self.metric = torch_metric
        self.name = name

    def __call__(self, logits, labels):
        ## Todo: Checks on the shape, will not necesarrily work if output is 2 dimensional
        metric_val = self.metric(logits, labels).detach().cpu().numpy()
        return metric_val


class PredictionMetricMapper:
    def __init__(self, num_classes) -> None:
        self.mapping = {
            "accuracy": TorchMetricsWrapper(Accuracy(task="multiclass", num_classes=num_classes), name="accuracy"),
            "f1": TorchMetricsWrapper(F1Score(task="multiclass", num_classes=num_classes), name="f1")
        }
    
    def __getitem__(self, name: str):
        return self.mapping[name]
