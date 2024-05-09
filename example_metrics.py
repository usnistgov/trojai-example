import argparse
from collections import defaultdict
import os
import json

import torch
import pandas as pd

from trojai_mitigation_round.metrics.torch_metrics import PredictionMetricMapper

def main(metrics, result_file, output_name, model_name, data_type, num_classes):
    metric_mapper = PredictionMetricMapper(num_classes)
    metric_results = defaultdict(list)
    with open(result_file, "r") as f:
        results = json.load(f)

    pred_logits = torch.tensor(results['pred_logits'])
    labels = torch.tensor(results['labels'])
    for metric_name in metrics:
        result = metric_mapper[metric_name](pred_logits, labels)
        metric_results["model_name"].append(model_name)
        metric_results["data_class"].append(data_type)
        metric_results["metric"].append(metric_name)
        metric_results["value"].append(result)
    
    df = pd.DataFrame(metric_results)
    df.to_csv(output_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to calculate metrics for a given model using specified logits and labels.")

    parser.add_argument("--metrics", nargs="+", type=str, help="Metrics to be calculated, e.g., f1, accuracy")
    parser.add_argument("--result_file", type=str, help="Path to the result JSON file to be loaded")
    parser.add_argument("--output_name", type=str, default="out.csv", help="Name of the output csv file")
    parser.add_argument("--model_name", type=str, help="Name for the PyTorch model")
    parser.add_argument("--data_type", choices=["poisoned", "clean"], help="Type of data: 'poisoned' or 'clean'")
    parser.add_argument("--num_classes", type=int, help="The number of classes in the dataset; required for metric calculations")
    
    args = parser.parse_args()
    
    main(args.metrics, args.result_file, args.output_name, args.model_name, args.data_type, args.num_classes)