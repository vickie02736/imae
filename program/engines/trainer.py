import torch
import torch.nn as nn
import torch.optim as optim
from program.engines.engine import Engine


class Trainer(Engine):

    def __init__(self, rank, config, dataset, model, epochs, resume_epoch):
        super().__init__(rank, config, dataset, model)
        self.epochs = epochs
        self.resume_epoch = resume_epoch
        self.init_loss_function()
        self.train_loader = self.init_dataloader()
        self.init_training_components()
        self.train_losses = {}

    def save_checkpoint(self, epoch, save_path):
        save_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict(),
            'epoch': epoch
        }
        torch.save(save_dict, save_path)

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
            T_start = self.epochs * 0.05 * self.len_dataset // self.config[
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
