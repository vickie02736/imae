import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP



class Engine:

    def __init__(self, rank, config, dataset, model, test_flag=False):
        self.rank = rank
        self.config = config
        self.dataset = dataset
        self.model = model
        self.test_flag = test_flag
        self.world_size = torch.cuda.device_count()
        self.device = torch.device(
            f"cuda:{self.rank % torch.cuda.device_count()}")
        self.init_dataloader()
        self.setup()

    def setup(self):
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])

    def init_dataloader(self):
        sampler = DistributedSampler(self.dataset,
                                     num_replicas=self.world_size,
                                     rank=self.rank)
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=self.config['batch_size'],
                                     pin_memory=True,
                                     shuffle=False,
                                     drop_last=True,
                                     sampler=sampler)
        self.len_dataset = len(self.dataset)