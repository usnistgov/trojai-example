import os
import json

import configargparse
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
from torchvision import transforms
from tqdm import tqdm
from PIL import Image

from trojai_mitigation_round.mitigations.finetuning import FineTuningTrojai
from trojai_mitigation_round.trojai_dataset import Round11SampleDataset


def prepare_mitigation(args):
    """Given the command line args, construct and return a subclass of the TrojaiMitigation class

    :param args: The command line args
    :return: A subclass of TrojaiMitigation that can implement a given mitigtaion technique
    """
    # Get required classes for loss and optimizer dynamically
    loss_class = getattr(torch.nn, args.loss_class)
    optim_class = getattr(torch.optim, args.optimizer_class)

    print(f"Using {loss_class} for ft loss")
    print(f"Using {optim_class} for ft optimizer")

    # Construct defense with args
    mitigation = FineTuningTrojai(
        loss_cls=loss_class,
        optim_cls=optim_class,
        lr=args.learning_rate,
        epochs=args.epochs,
        ckpt_dir=args.ckpt_dir,
        ckpt_every=args.ckpt_every,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    return mitigation


def prepare_model(path, device):
    """Prepare and load a model defined at a path

    :param path: the path to a pytorch state dict model that will be loaded
    :param device: Either cpu or cuda to push the device onto
    :return: A pytorch model
    """
    model = torch.load(path)
    model = model.to(device=device)
    return model


def prepare_dataset(dataset_path, do_mitigate, do_test):
    if do_mitigate:
        split = 'test'
    elif do_test:
        split = 'train'

    dataset = Round11SampleDataset(root=dataset_path, split=split, require_label=False)
    return dataset


def mitigate_model(model, mitigation, dataset, output_dir, output_name):
    """Given the a torch model and a path to a dataset that may or may not contain clean/poisoned examples, output a mitigated
    model into the output directory.

    :param model: Pytorch model to be mitigated
    :param mitigtaion: The given mitigation technique
    :param dataset: The Pytorch dataset that may/may not contain poisoned examples
    :param output_dir: The directory where the mitigated model's state dict is to be saved to.
    :param output_name: the name of the pytorch model that will be saved
    """
    mitigated_model = mitigation.mitigate_model(model, dataset)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(mitigated_model.model, os.path.join(output_dir, output_name))


def test_model(model, mitigation, testset, batch_size, num_workers, device):
    """Tests a given model on a given dataset, using a given mitigation's pre and post processing
    before and after interfence. 

    :param model: Pytorch model to test
    :param mitigation: The mitigation technique we're using
    :param testset_path: The the Pytorch testset that may or may not be poisoned
    :param batch_size: Batch size for the dataloader
    :param num_workers: The number of workers to use for the dataloader
    :param device: cuda or cpu device
    :return: dictionary of the results with the labels and logits
    """
    dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    
    model.eval()
    all_logits = torch.tensor([])
    all_labels = torch.tensor([])
    
    # Label could be None in case the dataset did not require it to load
    for x, y in tqdm(dataloader):
        preprocess_x, info = mitigation.preprocess_transform(x)
        output_logits = model(preprocess_x.to(device)).detach().cpu()
        final_logits = mitigation.postprocess_transform(output_logits.detach().cpu(), info)

        all_logits = torch.cat([all_logits, final_logits], axis=0)

        # Skip label concatentation if label is none
        if y is not None:
            all_labels = torch.cat([all_labels, y], axis=0)
    
    return {
        'pred_logits': all_logits.tolist(),
        'labels': all_labels.tolist()
    }


if __name__ == "__main__":
    # configargparse allows a YAML to define certain args 
    parser = configargparse.ArgParser(
        config_file_parser_class=configargparse.YAMLConfigFileParser
    )

    # Any CLI args not defined in either the command line or config will be None
    parser.add_argument(
        "--metaparameters",
        is_config_file_arg=True,
        type=str,
        required=True,
        help="Required YAML metaparameters file",
    )

    # The two entrypoints; the mitigation stage and the testing stage
    parser.add_argument('--mitigate', action='store_true', help='Flag that asserts we are conducting mitigation on the given model')
    parser.add_argument('--test', action='store_true', help='Flag that asserts we are conducting testing on the given model')
    parser.add_argument('--model_filepath', type=str, default="./model.pt", help="File path to the model that will be either mitigated or tested ")

    # Other defined paths
    parser.add_argument('--dataset', type=str, default=None, help="If doing training, filepath to the dataset that contains the sample data dataset. If doing test, filepath to a dataset that could either be poisoned or clean.")
    parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="File path to the folder where a scratch space is located.")
    parser.add_argument('--output_dirpath', type=str, default="./out", help="File path to where the output will be dumped")
    parser.add_argument('--model_output', type=str, default="mitigated.pt", help="Name of the mitigated model")

    # Performer-specific hyperparameters overwritten by metaparameters.yml (but defined here)
    parser.add_argument('--optimizer_class', type=str, help='Class to use for optimizer for fine tuning')
    parser.add_argument('--loss_class', type=str, help='Class to use for loss for fine tuning')
    parser.add_argument('--learning_rate', type=float, help='Learning rate to use for fine tuning')
    parser.add_argument('--epochs', type=int, help='Count of epochs to do fine tuning for')
    parser.add_argument('--ckpt_every', type=int, help='Saves ckpt every N epochs. Set to 0 to disable ckpting')
    parser.add_argument('--ckpt_dir', type=str, help="Loation to save checkpoints to, if enabled")

    # Misc args used across all mitigations
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")

    args = parser.parse_args()

    assert args.mitigate ^ args.test, "Must choose only one of mitigate or test"

    model = prepare_model(args.model_filepath, args.device)
    mitigation = prepare_mitigation(args)
    dataset = prepare_dataset(args.dataset, args.mitigate, args.test)

    # Mitigate a given model on a dataset that may/may not contain some mix of clean and poisoned data
    if args.mitigate:
        mitigate_model(model, mitigation, dataset, args.output_dirpath, args.model_output)
    # Test a model on an arbitrary dataset (either clean or poisoned)
    elif args.test:
        results = test_model(model, mitigation, dataset, args.batch_size, args.num_workers, args.device)
        with open(os.path.join(args.output_dirpath, "results.json"), 'w+') as f:
            json.dump(results, f)
        