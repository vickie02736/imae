import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from program.engines.engine import Engine
from torch.utils.data import DataLoader, DistributedSampler


class Trainer(Engine):
    def __init__(self, rank, args, train_dataset):
        Engine.__init__(self, rank, args)
        self.train_dataset = train_dataset
        self.init_train_dataloader()
        self.init_loss_function()

    def init_train_dataloader(self):
        sampler = DistributedSampler(self.train_dataset,
                                     num_replicas=self.world_size,
                                     rank=self.rank)
        self.train_loader = DataLoader(self.train_dataset,
                                batch_size=self.config[self.args.model_name]['batch_size'],
                                pin_memory=True,
                                shuffle=False,
                                drop_last=True,
                                sampler=sampler)
        self.len_dataset = len(self.train_dataset)

    def init_training_components(self):
        if self.config['train']['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                betas=(0.9, 0.95),
                weight_decay=0.03)
        elif self.config['train']['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'],
                alpha=0.9)
        elif self.config['train']['optimizer'] == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config['train']['learning_rate'])
        else:
            pass
        if self.config['train']['scheduler'] == 'CosineAnnealingLR':
            T_start = self.args.epochs * 0.05 * self.len_dataset // self.config[self.args.model_name][
                'batch_size']
            T_start = int(T_start)
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer, T_start, eta_min=1e-6, last_epoch=-1)
        elif self.config['train']['scheduler'] == 'StepLR':
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                       step_size=10,
                                                       gamma=0.1)
        self.scaler = torch.cuda.amp.GradScaler()

    def init_loss_function(self):
        if self.config['train']['loss_fn'] == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif self.config['train']['loss_fn'] == 'L1':
            self.loss_fn = nn.L1Loss()
        elif self.config['train']['loss_fn'] == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Invalid loss function')

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

    def save_checkpoint(self, epoch, save_path):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch
        }
        torch.save(save_dict, save_path)