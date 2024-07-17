import argparse
from .tools import int_or_string, str2bool

def get_args_parser():
    parser = argparse.ArgumentParser(description='Irregular time series forecasting')

    parser.add_argument('--resume-epoch', type=int_or_string, default=1, help='start epoch after last training')
    parser.add_argument('--database', type=str, default='shallow_water', help='Database name')
    parser.add_argument('--model-name', type=str, default='imae', help='Model name')
    parser.add_argument('--interpolation', type=str, default=None, choices=['linear', 'gaussian', None], help='Interpolation method')
    parser.add_argument('--test-flag', type=bool, default=False, help='Test flag')

    train_group = parser.add_argument_group()
    train_group.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    train_group.add_argument('--save-frequency', type=int, default=2, help='Save once after how many epochs of training')
    train_group.add_argument('--mask-flag', type=str2bool, default=True, help='Mask flag')

    test_group = parser.add_argument_group()
    test_group.add_argument('--mask-ratio', type=float, default=0.1, help='Mask ratio')
    test_group.add_argument('--rollout-times', type=int, default=2, help='Rollout times')

    return parser

def load_engine(model_name, test_flag):
    if not test_flag:
        engines = {
            'imae': 'ImaeTrainer',
            'convlstm': 'ConvLstmTrainer',
            'cae': 'CaeTrainer',
            'cae_lstm': 'CaeLstmTrainer'
        }
    else:
        engines = {
            'imae': 'ImaeTester', 
        }

    engine_name = engines.get(model_name)
    if engine_name:
        module = __import__('engines', fromlist=[engine_name])
        return getattr(module, engine_name)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")
