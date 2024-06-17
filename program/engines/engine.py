import os
import yaml
import json
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

    def load_checkpoint(self): 
        if self.args.interpolation: 
            self.save_checkpoint_path = os.path.join(self.config[self.args.model_name]['save_checkpoint'], self.args.interpolation)
            self.save_reconstruct_path = os.path.join(self.config[self.args.model_name]['save_reconstruct'], self.args.interpolation)
            self.save_loss_path = os.path.join(self.config[self.args.model_name]['save_loss'], self.args.interpolation)
        else:
            self.save_checkpoint_path = self.config[self.args.model_name]['save_checkpoint']
            self.save_reconstruct_path = self.config[self.args.model_name]['save_reconstruct']
            self.save_loss_path = self.config[self.args.model_name]['save_loss']
        os.makedirs(self.save_checkpoint_path, exist_ok=True)
        os.makedirs(self.save_reconstruct_path, exist_ok=True)
        os.makedirs(self.save_loss_path, exist_ok=True)

        if self.args.resume_epoch == 1:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.save_checkpoint_path, 'init.pth'))
            losses = {}
            with open(os.path.join(self.save_loss_path, 'train_losses.json'), 'w') as file:
                json.dump(losses, file)
            with open(os.path.join(self.save_loss_path, 'valid_losses.json'), 'w') as file:
                json.dump(losses, file)
        else:
            checkpoint = torch.load(os.path.join(self.save_checkpoint_path, f'checkpoint_{self.args.resume_epoch-1}.pth'), map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])
