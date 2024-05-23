import os
import copy
# import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from program.utils.metrics import RMSELoss, PSNRLoss, SSIMLoss
from program.utils.tools import plot, save_losses, mask

class Engine: 

    def __init__(self, rank, config, dataset, model, test_flag = False):
        self.rank = rank
        self.config = config
        self.dataset = dataset
        self.model = model
        self.world_size=torch.cuda.device_count()
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        self.init_loss_function()
        self.init_dataloader()
        

    def setup(self):
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank]) 

    def init_dataloader(self):
        # Initialize dataloaders
        sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], pin_memory=True,
                                     shuffle=False, drop_last=True, sampler=sampler)
        self.len_dataset = len(self.dataset)


    def init_loss_function(self):
        loss_functions = {}
        running_losses = {}
        for metric in self.config['test']['metric']:
            if metric == 'MSE':
                mse_loss = nn.MSELoss()
                loss_functions['MSE'] = mse_loss
                running_losses['MSE'] = [0 for _ in range(self.config['test']['rollout_times'])]
            elif metric == 'MAE':
                mae_loss = nn.L1Loss()
                loss_functions['MAE'] = mae_loss
                running_losses['MAE'] = [0 for _ in range(self.config['test']['rollout_times'])]
            elif metric == 'RMSE':
                rmse_loss = RMSELoss()
                loss_functions['RMSE'] = rmse_loss
                running_losses['RMSE'] = [0 for _ in range(self.config['test']['rollout_times'])]
            elif metric == 'SSIM':
                ssim_loss = SSIMLoss(self.device).forward
                loss_functions['SSIM'] = ssim_loss
                running_losses['SSIM'] = [0 for _ in range(self.config['test']['rollout_times'])]
            elif metric == 'PSNR':
                psnr_loss = PSNRLoss()
                loss_functions['PSNR'] = psnr_loss
                running_losses['PSNR'] = [0 for _ in range(self.config['test']['rollout_times'])]
            elif metric == "BCE":
                bce_loss = nn.BCEWithLogitsLoss()
                loss_functions['BCE'] = bce_loss
                running_losses['BCE'] = [0 for _ in range(self.config['test']['rollout_times'])]
            else:
                raise ValueError('Invalid metric')
        return loss_functions, running_losses



class Trainer(Engine):

    def __init__(self, rank, config, dataset, model, epochs, test_flag):
        super().__init__(rank, config, dataset, model, test_flag)
        self.epochs = epochs
        self.init_training_components() 
        self.train_losses = {}
        

    def train_epoch(self, epoch):
        self.epoch = epoch

        torch.manual_seed(self.epoch)
        self.model.train()

        total_predict_loss = 0.0
        total_rollout_loss = 0.0
        for i, sample in enumerate(self.dataloader):
            origin = sample["Input"].float().to(self.device)
            target = sample["Target"].float().to(self.device)
            target_chunks = torch.chunk(target, self.config['train']['rollout_times'], dim=1)

            with torch.cuda.amp.autocast():
                origin = mask(origin, mask_mtd=self.config["mask_method"])
                output = self.model(origin)
                predict_loss = self.loss_fn(output, target_chunks[0])
                output = self.model(output)
                rollout_loss = self.loss_fn(output, target_chunks[1])
                loss = predict_loss + rollout_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_predict_loss += predict_loss.item()
            total_rollout_loss += rollout_loss.item()
  

        if self.rank == 0:
            average_predict_loss = total_predict_loss / len(self.dataloader.dataset)
            average_rollout_loss = total_rollout_loss / len(self.dataloader.dataset)

            loss_data = {'predict_loss': average_predict_loss, 'rollout_loss': average_rollout_loss}
            save_losses(epoch, loss_data, self.config['train']['save_loss'], 'train_losses.json')
            self.save_checkpoint()
            # wandb.log(loss_data)


    def save_checkpoint(self):
        save_path = os.path.join(self.config['train']['save_checkpoint'], f'checkpoint_{self.epoch}.pth')
        save_dict = {'model': self.model.state_dict(), 
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict(),
                     'scaler': self.scaler.state_dict(),
                     'epoch': self.epoch}
        torch.save(save_dict, save_path)


    def load_checkpoint(self, resume_epoch):
        checkpoint_path = self.config['train']['save_checkpoint'] + f'checkpoint_{resume_epoch-1}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])


    def init_training_components(self): 

        # Set up loss function, optimizer, scheduler
        T_start = self.epochs * 0.05 * self.len_dataset // self.config['batch_size']
        T_start = int(T_start)
        if self.config['train']['optimizer'] == 'AdamW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.config['train']['learning_rate'], betas=(0.9, 0.95), weight_decay=0.03)
        elif self.config['train']['optimizer'] == 'RMSprop':
            self.optimizer = optim.RMSprop(self.model.parameters(), lr=self.config['train']['learning_rate'], alpha=0.9)
        else:
            pass
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_start, eta_min=1e-6, last_epoch=-1)
        self.scaler = torch.cuda.amp.GradScaler()


    def init_loss_function(self):
        # Set up loss function
        if self.config['train']['loss_fn'] == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif self.config['train']['loss_fn'] == 'L1':
            self.loss_fn = nn.L1Loss()
        elif self.config['train']['loss_fn'] == 'BCE':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError('Invalid loss function')
 

class Evaluator(Engine):

    def __init__(self, rank, config, dataset, model, test_flag=False):
        super(Evaluator, self).__init__(rank, config, dataset, model)
        self.test_flag = test_flag
        self.valid_losses = {}


    @torch.no_grad()
    def evaluate_epoch(self, epoch):
        self.epoch = epoch

        torch.manual_seed(self.epoch)
        self.model.eval()

        with torch.no_grad():

            loss_functions, running_losses = self.init_loss_function() 

            for i, sample in enumerate(self.dataloader):

                origin_before_masked = sample["Input"].float().to(self.device)
                origin_before_masked_ = copy.deepcopy(origin_before_masked)
                origin = mask(origin_before_masked, mask_mtd=self.config["mask_method"])
                target = sample["Target"].float().to(self.device)
                target_chunks = torch.chunk(target, self.config['valid']['rollout_times'], dim=1)
                origin_ = copy.deepcopy(origin)

                output_chunks = []
                for j, chunk in enumerate(target_chunks):
                    if j == 0:
                        output = self.model(origin)
                        output_chunks.append(output)
                    else: 
                        output = self.model(output)
                        output_chunks.append(output)

                if i == 1: 
                    plot(origin_before_masked_, origin_, output_chunks, target_chunks, self.config['seq_length'],
                              self.epoch, self.config['valid']['save_reconstruct'])
                    
                # Compute losses
                for metric, loss_fn in loss_functions.items():

                    loss = loss_fn(output, chunk)
                    running_losses[metric][j] += loss.item()

            chunk_losses = {}
            for metric, running_loss_list in running_losses.items():
                    total_loss = sum(running_loss_list)
                    average_loss = total_loss / len(self.dataloader.dataset)
                    chunk_losses[metric] = average_loss
            save_losses(epoch, chunk_losses, self.config['valid']['save_loss'], 'valid_losses.json')

