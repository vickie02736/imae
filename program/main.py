import argparse
import os
import sys

sys.path.append('..')

import random
import numpy as np
from tqdm import tqdm
import torch
from utils import get_args_parser, load_engine

SEED = 3409
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def main():

    parser = get_args_parser()
    args = parser.parse_args()

    EngineClass = load_engine(args.model_name, args.test_flag)
    engine = EngineClass(args)
    

    if args.test_flag == False: 
        for epoch in tqdm(range(args.resume_epoch, args.resume_epoch + args.epochs), desc="Epoch progress"):
            engine.train_epoch(epoch)
            engine.evaluate_epoch(epoch)
    else:
        engine.evaluate()


if __name__ == "__main__":
    main()