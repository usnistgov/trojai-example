""" Entrypoint to interact with the detector.
"""
import json
import logging
import warnings

import jsonschema

from detector import Detector

warnings.filterwarnings("ignore")


if __name__ == "__main__":
    from jsonargparse import ArgumentParser

    parser = ArgumentParser(
        description="Fake Trojan Detector to Demonstrate Test and Evaluation "
        "Infrastructure."
    )
    parser.add_argument(
        "--model_filepath",
        type=str,
        help="File path to the pytorch model file to be evaluated.",
        default="./model.pt"
    )
    parser.add_argument(
        "--result_filepath",
        type=str,
        help="File path to the file where output result should be written. After "
        "execution this file should contain a single line with a single floating "
        "point trojan probability.",
        default="./output"
    )
    parser.add_argument(
        "--scratch_dirpath",
        type=str,
        help="File path to the folder where scratch disk space exists. This folder will "
        "be empty at execution start and will be deleted at completion of "
        "execution.",
        default="./scratch"
    )
    parser.add_argument(
        "--examples_dirpath",
        type=str,
        help="File path to the folder of examples which might be useful for determining "
        "whether a model is poisoned.",
        default="./example"
    )
    parser.add_argument(
        "--round_training_dataset_dirpath",
        type=str,
        help="File path to the directory containing id-xxxxxxxx models of the current "
        "rounds training dataset.",
        default=None
    )

    parser.add_argument(
        "--metaparameters_filepath",
        help="Path to JSON file containing values of tunable paramaters to be used "
        "when evaluating models.",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--schema_filepath",
        type=str,
        help="Path to a schema file in JSON Schema format against which to validate "
        "the config file.",
        default=None,
        required=True,
    )
    parser.add_argument(
        "--learned_parameters_dirpath",
        type=str,
        help="Path to a directory containing parameter data (model weights, etc.) to "
        "be used when evaluating models.  If --configure_mode is set, these will "
        "instead be overwritten with the newly-configured parameters.",
        required=True,
    )
    parser.add_argument(
        "--configure_mode",
        help="Instead of detecting Trojans, set values of tunable parameters and write "
        "them to a given location.",
        default=False,
        action="store_true"
    )
    parser.add_argument(
        "--configure_models_dirpath",
        type=str,
        help="Path to a directory containing models to use when in configure mode.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
    )

    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    # Create the detector instance and loads the metaparameters.
    detector = Detector(args.metaparameters_filepath, args.learned_parameters_dirpath)

    if not args.configure_mode:
        if (
            args.model_filepath is not None
            and args.result_filepath is not None
            and args.scratch_dirpath is not None
            and args.examples_dirpath is not None
            and args.round_training_dataset_dirpath is not None
        ):
            logging.info("Calling the trojan detector")
            detector.infer(
                args.model_filepath,
                args.result_filepath,
                args.scratch_dirpath,
                args.examples_dirpath,
                args.round_training_dataset_dirpath,
            )
        else:
            logging.error("Required Evaluation-Mode parameters missing!")
    else:
        if args.configure_models_dirpath is not None:
            logging.info("Calling configuration mode")
            detector.configure(
                args.configure_models_dirpath,
            )
        else:
            logging.error("Required Configure-Mode parameters missing!")
