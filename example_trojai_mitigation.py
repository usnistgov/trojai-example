import os
import json
import jsonschema
import torch
from tqdm import tqdm
from trojai_mitigation_round.mitigations.finetuning import FineTuningTrojai
from trojai_mitigation_round.trojai_dataset import Round11SampleDataset

def prepare_mitigation(args, config_json):
    """Given the command line args, construct and return a subclass of the TrojaiMitigation class

    :param args: The command line args
    :return: A subclass of TrojaiMitigation that can implement a given mitigtaion technique
    """
    # Get required classes for loss and optimizer dynamically
    loss_class = getattr(torch.nn, config_json['loss_class'])
    optim_class = getattr(torch.optim, config_json['optimizer_class'])

    print(f"Using {loss_class} for ft loss")
    print(f"Using {optim_class} for ft optimizer")

    scratch_dirpath = args.scratch_dirpath
    ckpt_dirpath = os.path.join(scratch_dirpath, config_json['ckpt_dir'])

    if not os.path.exists(ckpt_dirpath):
        os.makedirs(ckpt_dirpath, exist_ok=True)

    # Construct defense with args
    mitigation = FineTuningTrojai(
        loss_cls=loss_class,
        optim_cls=optim_class,
        lr=config_json['learning_rate'],
        epochs=config_json['epochs'],
        ckpt_dir=ckpt_dirpath,
        ckpt_every=config_json['ckpt_every'],
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


def prepare_dataset(dataset_path, split_name):
    dataset = Round11SampleDataset(root=dataset_path, split=split_name, require_label=False)
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
    all_fnames = []
    
    # Label could be None in case the dataset did not require it to load
    for x, y, fname in tqdm(dataloader):
        preprocess_x, info = mitigation.preprocess_transform(x)
        output_logits = model(preprocess_x.to(device)).detach().cpu()
        final_logits = mitigation.postprocess_transform(output_logits.detach().cpu(), info)

        all_logits = torch.cat([all_logits, final_logits], axis=0)
        all_labels = torch.cat([all_labels, y], axis=0)
        all_fnames.extend(fname)
    
    fname_to_logits = dict(zip(all_fnames, all_logits.tolist()))

    return fname_to_logits

# Executes in mitigate mode, generating an approach to mitigate the model
def run_mitigate_mode(args):
    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    model = prepare_model(args.model_filepath, args.device)
    mitigation = prepare_mitigation(args, config_json)
    dataset = prepare_dataset(args.dataset_dirpath, split_name='train')

    mitigate_model(model, mitigation, dataset, args.output_dirpath, args.model_output_name)

# Executes in test model, outputting model logits for each example
def run_test_mode(args):
    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    model = prepare_model(args.model_filepath, args.device)
    mitigation = prepare_mitigation(args, config_json)
    dataset = prepare_dataset(args.dataset_dirpath, split_name='test')

    results = test_model(model, mitigation, dataset, args.batch_size, args.num_workers, args.device)
    with open(os.path.join(args.output_dirpath, "results.json"), 'w+') as f:
        json.dump(results, f)

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description='Parser for mitigation round, with two modes of operation, mitigate and test')

    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    mitigate_parser = subparser.add_parser('mitigate', help='Generates a mitigated model')

    # Mitigation arguments
    mitigate_parser.add_argument("--metaparameters_filepath", type=str, required=True, help="Path JSON file containing values of tunable parameters based on json schema")
    mitigate_parser.add_argument("--schema_filepath", type=str, help="Path to a schema file in JSON Schema format against which to validate the metaparameters file.", required=True)
    mitigate_parser.add_argument('--model_filepath', type=str, default="./model.pt", help="File path to the model that will be mitigated")
    mitigate_parser.add_argument('--dataset_dirpath', type=str, help="A dataset of examples to train the mitigated model with.", required=True)
    mitigate_parser.add_argument('--output_dirpath', type=str, default="./out", help="The directory path to where the output will be dumped")
    mitigate_parser.add_argument('--model_output_name', type=str, default="mitigated.pt", help="Name of the mitigated model that will be written to the output dirpath")
    mitigate_parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="The directory where a scratch space is located.")
    mitigate_parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    mitigate_parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    mitigate_parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")

    # Test arguments
    test_parser = subparser.add_parser('test', help='Tests a mitigated model with example data')
    test_parser.add_argument("--metaparameters_filepath", type=str, required=True, help="Path JSON file containing values of tunable parameters based on json schema")
    test_parser.add_argument("--schema_filepath", type=str, help="Path to a schema file in JSON Schema format against which to validate the metaparameters file.", required=True)
    test_parser.add_argument('--model_filepath', type=str, default="./model.pt", help="File path to the mitigated model that will be tested")
    test_parser.add_argument('--dataset_dirpath', type=str, help="A dataset of examples to test the mitigated model with.", required=True)
    test_parser.add_argument('--scratch_dirpath', type=str, default="./scratch", help="The directory where a scratch space is located.")
    test_parser.add_argument('--output_dirpath', type=str, default="./out", help="The directory path to where the output will be dumped")
    test_parser.add_argument('--batch_size', type=int, default=32, help="The batch size that the technique would use for data loading")
    test_parser.add_argument('--device', type=str, default='cuda', help="The device to use")
    test_parser.add_argument('--num_workers', type=int, default=1, help="The number of CPU processes to use to load data")

    # Setup default function to call for mitigate/test
    mitigate_parser.set_defaults(func=run_mitigate_mode)
    test_parser.set_defaults(func=run_test_mode)

    args = parser.parse_args()

    # Call appropriate function
    args.func(args)
        
