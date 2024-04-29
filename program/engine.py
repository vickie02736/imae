from model import VisionTransformer
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml


class Trainer:
    def __init__(self, rank, config, dataset, model, epochs):
        self.rank = rank
        self.config = config
        # self.start_epoch = start_epoch
        self.dataset = dataset
        self.epochs = epochs
        self.model = model
        self.setup()



    def train_epoch(self, epoch):

        self.model.train()

        running_loss = 0.0
        running_loss_rollout = 0.0
        train_losses = []
        train_losses_rollout = []

        for _, sample in enumerate(self.dataloader):
            origin = sample["Input"].float().to(self.device)
            target = sample["Target"].float().to(self.device)
            target_chunks = torch.chunk(target, self.config['rollout_times'], dim=1)

            mask_ratio = torch.rand(1).item()
            num_mask = int(1 + mask_ratio * (self.config['seq_length'] - 2))

            with torch.cuda.amp.autocast():
                output = self.model(origin, num_mask)
                predict_loss = self.loss_fn(output, target_chunks[0])
                output = self.model(output, 0)
                rollout_loss = self.loss_fn(output, target_chunks[1])
                loss = predict_loss + rollout_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            running_loss += predict_loss.item()
            running_loss_rollout += rollout_loss.item()

            if self.rank == 0:
                train_loss = running_loss / len(self.dataloader.dataset)
                train_losses.append(train_loss)
                rollout_loss = running_loss_rollout / len(self.dataloader.dataset)
                train_losses_rollout.append(train_loss)

                # save_path = os.path.join(self.config['save_path']['checkpoint'], f'checkpoint_{epoch}')
                # save_dict = {
                #     'model': self.model.state_dict(),
                #     'optimizer': self.optimizer.state_dict(),
                #     'scheduler': self.scheduler.state_dict(),
                #     'scaler': self.scaler.state_dict(),
                #     'epoch': epoch
                # }
                # torch.save(save_dict, save_path + '.pth')

                self.save_checkpoint(epoch)

            return train_losses, train_losses_rollout
        

    def setup(self):
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        self.world_size=torch.cuda.device_count()
        self.model = self.model.to(self.device)
        self.init_dataloader()
        self.init_training_components()


    def init_dataloader(self):
        # Initialize dataloaders
        sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], shuffle=False, drop_last=True, sampler=sampler)
        self.len_dataset = len(self.dataset)


    def init_training_components(self): 
        # Set up loss function, optimizer, scheduler
        self.loss_fn = nn.MSELoss()
        T_start = self.epochs * 0.05 * self.len_dataset // self.config['batch_size']
        T_start = int(T_start)
        learning_rate = 1e-4
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.03)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_start, eta_min=1e-6, last_epoch=-1)
        self.scaler = torch.cuda.amp.GradScaler()


    def load_checkpoint(self, start_epoch):
        checkpoint_path = self.config['save_path']['checkpoint'] + f'checkpoint_{start_epoch-1}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])

        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.config['save_path']['checkpoint'], f'checkpoint_{epoch}')
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch
        }
        torch.save(save_dict, save_path + '.pth')