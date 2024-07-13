import argparse
import os
import sys
sys.path.append('..')
import itertools
import yaml
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.distributed as dist
from utils import int_or_string, str2bool
from utils import save_losses, mask

SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_args_parser():

    parser = argparse.ArgumentParser(description='Irregular time series forecasting')
    parser.add_argument('--resume-epoch',
                        type=int_or_string,
                        default=1,
                        help='load checkpoint')
    parser.add_argument('--database',
                        type=str,
                        default='shallow_water',
                        help='Database name')
    parser.add_argument('--model-name',
                        type=str,
                        default='imae',
                        help='Model name')
    parser.add_argument('--interpolation',
                        type=str,
                        default=None,
                        choices=['linear', 'gaussian', None],
                        help='Interpolation method')
    parser.add_argument('--mask-flag', type=str2bool, default=True, help='Mask flag')
    parser.add_argument('--test-flag', type=bool, default=False, help='Test flag')

    
    return parser


def main():

    parser = get_args_parser()
    args = parser.parse_args()

    dist.init_process_group("nccl")
    torch.cuda.empty_cache()
    rank = dist.get_rank()

    model_list = ['imae', 'convlstm', 'cae_lstm']

    bound_list = ['inner_test_file', 'outer_test_file']
    rollout_list = [2, 3, 4]
    timestep_list = [100, 120]
    combinations = list(itertools.product(bound_list, rollout_list, timestep_list))

    mask_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    from database.shallow_water.dataset import seq_DataBuilder
    data_config = yaml.load(open("../database/shallow_water/config.yaml", "r"), Loader=yaml.FullLoader)

    for combination in combinations:
        dataset = seq_DataBuilder(data_config[combination[0]], 10, combination[1], timestep=combination[2])
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
        for mask_ratio in mask_ratios:
            for model_name in model_list: 
                if args.model_name == 'imae':
                    from engines import ImaeTester
                    engine = ImaeTester(rank, args, dataset)
                elif args.model_name == 'convlstm':
                    pass
                elif args.model_name == 'cae_lstm':
                    pass
                else:
                    pass

                for mask_ratio in mask_ratios:
                    engine.evaluate_epoch(mask_ratio)
    

    from database.shallow_water.dataset import seq_DataBuilder, fra_DataBuilder


if __name__ == "__main__":
    main()