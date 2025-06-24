"""
Config parser for scripts

This module genralizes parsing
across different evaluation scripts.
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import trackeval  # noqa: E402


def parse_config(dataset_class, default_metrics_config):
    """
    Parsing the arguments and returning structured config

    Input Arguments:
        dataset_class: The dataset class
        default_metrics_config: Default metrics dict

    Output :
        eval_config, dataset_config, metrics_config
    """
    # Command line interface:
    default_eval_config = trackeval.Evaluator.get_default_eval_config()

    default_dataset_config = dataset_class.get_default_dataset_config()
    config = {**default_eval_config, **default_dataset_config, **default_metrics_config}  # Merge default configs

    parser = argparse.ArgumentParser()
    for setting in config.keys():
        if type(config[setting]) == list or type(config[setting]) == type(None):
            parser.add_argument("--" + setting, nargs='+')
        else:
            parser.add_argument("--" + setting)

    args = parser.parse_args().__dict__
    for setting in args.keys():
        if args[setting] is not None:
            if type(config[setting]) == type(True):
                if args[setting] == 'True':
                    x = True
                elif args[setting] == 'False':
                    x = False
                else:
                    raise Exception('Command line parameter ' + setting + 'must be True or False')
            elif type(config[setting]) == type(1):
                x = int(args[setting])
            elif type(args[setting]) == type(None):
                x = None
            else:
                x = args[setting]
            config[setting] = x

    eval_config = {k: v for k, v in config.items() if k in default_eval_config.keys()}
    dataset_config = {k: v for k, v in config.items() if k in default_dataset_config.keys()}
    metrics_config = {k: v for k, v in config.items() if k in default_metrics_config.keys()}

    return eval_config, dataset_config, metrics_config