import os

import configargparse
import torch
from torch import nn
import torchvision
from torchvision.models import resnet50
from torchvision import transforms
from PIL import Image

from trojai_mitigation_round.mitigations import FineTuningTrojai

# Probably needs to be a better way to define the transforms for a given dataset?
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def prepare_mitigation(args):
    """Given the command line args, prepare the mitigation technique and 
    return a subclass of TrojAIMitigation.

    Args:
        args (namespace): The command line args that were passed to the program

    Returns:
        TrojAIMitigation: A subclass that at least implements the mitigate_model() method, and
        optionally preprocess_transform() and postprocess_transform()
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
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
    )
    return mitigation


def prepare_model(path, num_classes, device):
    """Prepare and load a dataset defined at a path

    Args:
        path (str): The path to the Pytorch model
        num_classes (int): The number of classes that the model was trained on
        device (str): The device (cuda/cpu) to put the model on

    Returns:
        torch module: The loaded Pytorch model
    """
    model = resnet50().to(device=device)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(path))
    return model


def mitigate_model(model, clean_trainset_path, poison_trainset_path, output_dir):
    """Given a model and paths to the different variations of datasets, output a mitigated model

    Args:
        model (pytorch module): A model to be mitigated
        clean_trainset_path (str): path to the clean dataset that can be loaded in as a DatasetFolder
        poison_trainset_path (str): path to the poisoned dataset that can be loaded in as a DatasetFolder
        output_dir (str): the path where the model will be dumped to
    """
    clean_trainset = torchvision.datasets.DatasetFolder(clean_trainset_path, loader=Image.open, extensions=("png",), transform=transform_train)
    poisoned_trainset = torchvision.datasets.DatasetFolder(poison_trainset_path, loader=Image.open, extensions=("png",), transform=transform_train)
    mitigated_model = mitigation.mitigate_model(model, clean_trainset, poisoned_trainset)
    os.makedirs(output_dir, exist_ok=True)
    torch.save(mitigated_model.state_dict, os.path.join(output_dir, "model.pt"))


def test_model(model, testset_path, batch_size, num_workers, device):
    # performers should know not to shuffle the dataset in docstring
    testset = torchvision.datasets.DatasetFolder(testset_path, loader=Image.open, extensions=("png",), transform=transform_test)
    dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, num_workers=num_workers)
    
    model.eval()
    all_logits = torch.tensor([])
    all_labels = torch.tensor([])
    # drop the label
    for x, y in dataloader:
        preprocess_x, info = model.preprocess_transform(x)
        output_logits = model(preprocess_x.to(device)).detach().cpu()
        final_logits = model.postprocess_transform(output_logits.detach().cpu(), info)

        all_logits = torch.cat([all_logits, final_logits], axis=0)
        all_labels = torch.cat([all_labels, y], axis=0)
    
    return {
        'pred_logits': all_logits,
        'labels': all_labels
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
    parser.add_argument('--clean_trainset', type=str, default=None, help="If doing mitigation, filepath to clean trainset that can be loaded as a torchvision DatasetFolder. This directory could also be empty.")
    parser.add_argument('--poison_trainset', type=str, default=None, help="If doing mitigation, filepath to poisoned trainset that can be loaded as a torchvision DatasetFolder. This directory could also be empty.")
    parser.add_argument('--testset', type=str, default=None, help="If doing test, filepath to a dataset that could either be poisoned or clean.")
    parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="File path to the folder where a scratch space is located.")
    parser.add_argument('--output_dirpath', type=str, default="./out", help="File path to where the output will be dumped")

    # Performer-specific hyperparameters overwritten by metaparameters.yml (but defined here)
    parser.add_argument('--optimizer_class', type=str, help='Class to use for optimizer for fine tuning')
    parser.add_argument('--loss_class', type=str, help='Class to use for loss for fine tuning')
    parser.add_argument('--learning_rate', type=float, help='Learning rate to use for fine tuning')
    parser.add_argument('--epochs', type=int, help='Count of epochs to do fine tuning for')

    # Misc args used across all mitigations
    parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")
    parser.add_argument('--num_classes', type=int, help="The number of classes that the model/dataset is using")

    args = parser.parse_args()

    assert args.mitigate ^ args.test, "Must choose only one of mitigate or test"

    model = prepare_model(args.model_filepath, args.num_classes, args.device)
    mitigation = prepare_mitigation(args)

    # Mitigate a given model on a clean/poison dataset (either/both of which could be empty)
    if args.mitigate:
        mitigate_model(model, args.clean_trainset, args.poison_trainset, args.output_dirpath)
    # Test a model on an arbitrary dataset (either clean or poisoned)
    elif args.test:
        results = test_model(model, args.testset, args.batch_size, args.num_workers, args.device)
        torch.save(results, os.path.join(args.output_dirpath, "results.pkl"))
        