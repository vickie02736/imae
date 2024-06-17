import yaml
import torch
from torch.nn.parallel import DistributedDataParallel as DDP



class Engine:
    def __init__(self, rank, args):
        self.rank = rank
        self.args = args
        self.world_size = torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        self.load_config()

    def setup(self):
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])

    def load_config(self): 
        if self.args.database == 'shallow_water':
            self.config = yaml.load(open("../configs/sw_train_config.yaml", "r"), Loader=yaml.FullLoader)
            self.data_config = yaml.load(open("../database/shallow_water/config.yaml", "r"), Loader=yaml.FullLoader)
        elif self.args.database == 'moving_mnist':
            self.config = yaml.load(open("../configs/mm_train_config.yaml", "r"), Loader=yaml.FullLoader)
        else:
            pass
