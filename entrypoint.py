# NIST-developed software is provided by NIST as a public service. You may use, copy and distribute copies of the software in any medium, provided that you keep intact this entire notice. You may improve, modify and create derivative works of the software or any portion of the software, and you may copy and distribute such modifications or works. Modified works should carry a notice stating that you changed the software and should note the date and nature of any such change. Please explicitly acknowledge the National Institute of Standards and Technology as the source of the software.

# NIST-developed software is expressly provided "AS IS." NIST MAKES NO WARRANTY OF ANY KIND, EXPRESS, IMPLIED, IN FACT OR ARISING BY OPERATION OF LAW, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTY OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, NON-INFRINGEMENT AND DATA ACCURACY. NIST NEITHER REPRESENTS NOR WARRANTS THAT THE OPERATION OF THE SOFTWARE WILL BE UNINTERRUPTED OR ERROR-FREE, OR THAT ANY DEFECTS WILL BE CORRECTED. NIST DOES NOT WARRANT OR MAKE ANY REPRESENTATIONS REGARDING THE USE OF THE SOFTWARE OR THE RESULTS THEREOF, INCLUDING BUT NOT LIMITED TO THE CORRECTNESS, ACCURACY, RELIABILITY, OR USEFULNESS OF THE SOFTWARE.

# You are solely responsible for determining the appropriateness of using and distributing the software and you assume all risks associated with its use, including but not limited to the risks and costs of program errors, compliance with applicable laws, damage to or loss of data, programs or equipment, and the unavailability or interruption of operation. This software is not intended to be used in any situation where a failure could cause risk of injury or damage to property. The software developed by NIST employees is not subject to copyright protection within the United States.

""" Entrypoint to interact with the detector.
"""
import json
import logging
import warnings

import jsonschema

from detector import Detector

warnings.filterwarnings("ignore")


def inference_mode(args):
    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    # Create the detector instance and loads the metaparameters.
    detector = Detector(args.metaparameters_filepath, args.learned_parameters_dirpath)

    logging.info("Calling the trojan detector")
    detector.infer(args.model_filepath, args.result_filepath, args.scratch_dirpath, args.examples_dirpath, args.round_training_dataset_dirpath)


def configure_mode(args):
    # Validate config file against schema
    with open(args.metaparameters_filepath) as config_file:
        config_json = json.load(config_file)
    with open(args.schema_filepath) as schema_file:
        schema_json = json.load(schema_file)

    # Throws a fairly descriptive error if validation fails.
    jsonschema.validate(instance=config_json, schema=schema_json)

    # Create the detector instance and loads the metaparameters.
    detector = Detector(args.metaparameters_filepath, args.learned_parameters_dirpath)

    logging.info("Calling configuration mode")
    detector.configure(args.configure_models_dirpath, args.automatic_configuration)


if __name__ == "__main__":
    from argparse import ArgumentParser

    temp_parser = ArgumentParser(add_help=False)

    parser = ArgumentParser(
        description="Template Trojan Detector to Demonstrate Test and Evaluation. Should be customized to work with target round in TrojAI."
        "Infrastructure."
    )

    parser.set_defaults(func=lambda args: parser.print_help())

    subparser = parser.add_subparsers(dest='cmd', required=True)

    inf_parser = subparser.add_parser('infer', help='Execute container in inference mode for TrojAI detection.')

    inf_parser.add_argument(
        "--model_filepath",
        type=str,
        help="File path to the pytorch model file to be evaluated.",
        required=True
    )
    inf_parser.add_argument(
        "--result_filepath",
        type=str,
        help="File path to the file where output result should be written. After "
        "execution this file should contain a single line with a single floating "
        "point trojan probability.",
        required=True
    )
    inf_parser.add_argument(
        "--scratch_dirpath",
        type=str,
        help="File path to the folder where scratch disk space exists. This folder will "
        "be empty at execution start and will be deleted at completion of "
        "execution.",
        required=True
    )
    inf_parser.add_argument(
        "--examples_dirpath",
        type=str,
        help="File path to the folder of examples which might be useful for determining "
        "whether a model is poisoned.",
        default=None,
        required=False
    )
    inf_parser.add_argument(
        "--round_training_dataset_dirpath",
        type=str,
        help="File path to the directory containing id-xxxxxxxx models of the current "
        "rounds training dataset.",
        required=True
    )

    inf_parser.add_argument(
        "--metaparameters_filepath",
        help="Path to JSON file containing values of tunable paramaters to be used "
        "when evaluating models.",
        type=str,
        required=True,
    )
    inf_parser.add_argument(
        "--schema_filepath",
        type=str,
        help="Path to a schema file in JSON Schema format against which to validate "
        "the config file.",
        required=True,
    )
    inf_parser.add_argument(
        "--learned_parameters_dirpath",
        type=str,
        help="Path to a directory containing parameter data (model weights, etc.) to "
        "be used when evaluating models.  If --configure_mode is set, these will "
        "instead be overwritten with the newly-configured parameters.",
        required=True,
    )

    inf_parser.set_defaults(func=inference_mode)

    configure_parser = subparser.add_parser('configure', help='Execute container in configuration mode for TrojAI detection. This will produce a new set of learned parameters to be used in inference mode.')

    configure_parser.add_argument(
        "--scratch_dirpath",
        type=str,
        help="File path to the folder where scratch disk space exists. This folder will "
        "be empty at execution start and will be deleted at completion of "
        "execution.",
        required=True
    )

    configure_parser.add_argument(
        "--configure_models_dirpath",
        type=str,
        help="Path to a directory containing models to use when in configure mode.",
        required=True,
    )

    configure_parser.add_argument(
        "--metaparameters_filepath",
        help="Path to JSON file containing values of tunable paramaters to be used "
        "when evaluating models.",
        type=str,
        required=True,
    )

    configure_parser.add_argument(
        "--schema_filepath",
        type=str,
        help="Path to a schema file in JSON Schema format against which to validate "
        "the config file.",
        required=True,
    )

    configure_parser.add_argument(
        "--learned_parameters_dirpath",
        type=str,
        help="Path to a directory containing parameter data (model weights, etc.) to "
        "be used when evaluating models.  If --configure_mode is set, these will "
        "instead be overwritten with the newly-configured parameters.",
        required=True,
    )
    configure_parser.add_argument(
        '--automatic_configuration',
        help='Whether to enable automatic training or not, which will retrain the detector across multiple variables',
        action='store_true',
    )

    configure_parser.set_defaults(func=configure_mode)

    logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
        )

    args, extras = temp_parser.parse_known_args()

    if '--help' in extras or '-h' in extras:
        args = parser.parse_args()
    # Checks if new mode of operation is being used, or is this legacy
    elif len(extras) > 0 and extras[0] in ['infer', 'configure']:
        args = parser.parse_args()
        args.func(args)

    else:
        # Assumes we have inference mode if the subparser is not used
        args = inf_parser.parse_args()
        args.func(args)
