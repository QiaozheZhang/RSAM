import argparse
import sys
import yaml
import args.parser_utils as _parser

global parser_args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "yes", "y", "1"):
        return True
    elif v.lower() in ("false", "no", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

class Args:
    def parse_arguments(self, jupyter_mode):
        parser = argparse.ArgumentParser(description="my args")

        parser.add_argument(
            "--config",
            default='configs/template.yml',
            help="Config file to use"
        )

        parser.add_argument(
            "--seed",
            type=int,
            default=42,
            help="seed"
        )

        parser.add_argument(
            "--data",
            default='data/datasets',
            help="data path"
        )

        parser.add_argument(
            "--arch",
            type=str,
            default='FC5',
            help="Model"
        )

        parser.add_argument(
            "--init",
            type=str,
            default='kaiming_normal',
            help="kaiming_normal | xavier_normal"
        )

        parser.add_argument(
            "--dataset",
            type=str,
            default='cifar10',
            help="Dataset"
        )

        parser.add_argument(
            "--optimizer",
            type=str,
            default='sgd',
            help="optimizer|sgd|adam"
        )

        parser.add_argument(
            "--aug",
            type=str2bool,
            default=False,
            help="Datset augmentation"
        )

        parser.add_argument(
            "--use_fix",
            type=str2bool,
            default=False,
            help="use_fix datset"
        )

        parser.add_argument(
            "--gpu",
            type=int,
            default=0,
            help="Cuda device"
        )

        parser.add_argument(
            "--epochs",
            type=int,
            default=200,
            help="epoch"
        )

        parser.add_argument(
            "--batch_size",
            type=int,
            default=128,
            help="Batch size"
        )

        parser.add_argument(
            "--lr",
            type=float,
            default=0.01,
            help="Learning rate"
        )

        parser.add_argument(
            "--warmup",
            type=int,
            default=5,
            help="warmup"
        )

        parser.add_argument(
            "--warmup_start",
            type=float,
            default=0.001,
            help="warmup_start"
        )

        parser.add_argument(
            "--warmup_end",
            type=float,
            default=1.0,
            help="warmup_end"
        )

        parser.add_argument(
            "--wd",
            type=float,
            default=0.00001,
            help="Weight decay"
        )

        parser.add_argument(
            "--lr_policy",
            type=str,
            default='cosine_lr',
            help="Schedular mode"
        )

        parser.add_argument(
            "--lmbda",
            type=float,
            default=0.00001,
            help="Regularization paramerter"
        )

        parser.add_argument(
            "--regularization",
            type=str,
            default='L1',
            help="Regularization mode"
        )

        parser.add_argument(
            "--model_path",
            type=str,
            default='./test.pth',
            help="model path to load"
        )

        parser.add_argument(
            "--bias",
            type=str2bool,
            default=True,
            help="use bias"
        )

        parser.add_argument(
            "--use_full_data",
            type=str2bool,
            default=True,
            help="use bias"
        )

        parser.add_argument(
            "--param_names",
            type=str,
            default=None,
            help="param names"
        )

        parser.add_argument(
            "--recipe",
            type=int,
            default=0,
            help="recipe"
        )

        parser.add_argument(
            "--renyi",
            type=str2bool,
            default=False,
            help="use renyi reg"
        )

        parser.add_argument(
            "--renyi_s",
            type=float,
            default=0.,
            help="renyi reg strength"
        )

        parser.add_argument(
            "--sam_mode",
            type=str,
            default="RSAM",
            help="SAM MODE"
        )

        parser.add_argument(
            "--alpha",
            type=float,
            default=0.,
            help="renyi alpha"
        )

        parser.add_argument(
            "--plain_epoch",
            type=int,
            default=5,
            help="plain epoch for sgd"
        )

        if jupyter_mode:
            args = parser.parse_args("")
        else:
            args = parser.parse_args()
        self.get_config(args, jupyter_mode)

        return args

    def get_config(self, parser_args, jupyter_mode=False):
        # get commands from command line
        override_args = _parser.argv_to_vars(sys.argv)

        # load yaml file
        yaml_txt = open(parser_args.config).read()

        # override args
        loaded_yaml = yaml.load(yaml_txt, Loader=yaml.FullLoader)
        if not jupyter_mode:
            for v in override_args:
                loaded_yaml[v] = getattr(parser_args, v)

        print(f"=> Reading YAML config from {parser_args.config}")
        parser_args.__dict__.update(loaded_yaml)


    def isNotebook(self):
        try:
            shell = get_ipython().__class__.__name__
            if shell == 'ZMQInteractiveShell':
                return True   # Jupyter notebook or qtconsole
            elif shell == 'TerminalInteractiveShell':
                return False  # Terminal running IPython
            else:
                return False  # Other type (?)
        except NameError:
            return False      # Probably standard Python interpreter

    def get_args(self, jupyter_mode=False):
        global parser_args
        jupyter_mode = self.isNotebook()
        parser_args = self.parse_arguments(jupyter_mode)

args = Args()
args.get_args()