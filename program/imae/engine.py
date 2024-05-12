import os
import copy
import json
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
from piqa import SSIM

from program.utils.metrics import RMSELoss, PSNRLoss


class Engine: 

    def __init__(self, rank, config, dataset, model, mode='train'):
        self.rank = rank
        self.config = config
        self.dataset = dataset
        self.model = model
        self.mode = mode
        self.setup()
        

    def setup(self):
        self.device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        self.world_size=torch.cuda.device_count()
        self.model = self.model.to(self.device)
        self.init_dataloader()
        self.init_loss_function()


    def init_dataloader(self):
        # Initialize dataloaders
        sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank)
        self.dataloader = DataLoader(self.dataset, batch_size=self.config['batch_size'], 
                                     shuffle=False, drop_last=True, sampler=sampler)
        self.len_dataset = len(self.dataset)


    def load_checkpoint(self, resume_epoch):
        checkpoint_path = self.config['train']['save_checkpoint'] + f'checkpoint_{resume_epoch-1}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])
        self.model.load_state_dict(checkpoint['model'])


    def save_losses(self, epoch, loss_data, save_path, save_name):
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, save_name), 'r') as f: 
            losses = json.load(f)
        losses[epoch]=loss_data
        with open(os.path.join(save_path, save_name), 'w') as f: 
            json.dump(losses, f)


    def mask(self, x, test_flag=False, mask_ratio=None): 
        if test_flag == False: 
            mask_ratio = torch.rand(1).item()
        else:
            mask_ratio = mask_ratio
        num_mask = int(1 + mask_ratio * (self.config['seq_length'] - 2))
        weights = torch.ones(x.shape[1]).expand(x.shape[0], -1)
        idx = torch.multinomial(weights, num_mask, replacement=False)
        masked_tensor = torch.zeros(x.shape[2], x.shape[3], x.shape[4]).to(x.device)
        batch_indices = torch.arange(x.shape[0], device=x.device).unsqueeze(1).expand(-1, num_mask)
        x[batch_indices, idx] = masked_tensor
        return x
    

    def plot(self, origin, masked_origin, output_chunks, target_chunks, idx, save_path):
        rollout_times = len(output_chunks)
        _, ax = plt.subplots(rollout_times*2+2, self.config['seq_length'], figsize=(self.config['seq_length']*2, rollout_times*4+2))
        for j in range(self.config['seq_length']): 
            # visualise input
            ax[0][j].imshow(origin[0][j][0].cpu().detach().numpy())
            ax[0][j].set_xticks([])
            ax[0][j].set_yticks([])
            ax[0][j].set_title("Timestep {timestep} (Input)".format(timestep=j+1), fontsize=10)
            # visualise masked input
            ax[1][j].imshow(masked_origin[0][j][0].cpu().detach().numpy())
            ax[1][j].set_xticks([])
            ax[1][j].set_yticks([])
            ax[1][j].set_title("Timestep {timestep} (Masked Input)".format(timestep=j+1), fontsize=10)
        for k in range(rollout_times): 
            for j in range(self.config['seq_length']):
                # visualise output
                ax[2*k+2][j].imshow(output_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2*k+2][j].set_xticks([])
                ax[2*k+2][j].set_yticks([])
                ax[2*k+2][j].set_title("Timestep {timestep} (Prediction)".format(timestep=j+(k+1)*self.config['seq_length']), fontsize=10)
                # visualise target
                ax[2*k+3][j].imshow(target_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2*k+3][j].set_xticks([])
                ax[2*k+3][j].set_yticks([])
                ax[2*k+3][j].set_title("Timestep {timestep} (Target)".format(timestep=j+(k+1)*self.config['seq_length']), fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{idx}.png"))
        plt.close()


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
                self.ssim_loss = SSIM().to(self.device)
                loss_functions['SSIM'] = self.ssim_loss
                running_losses['SSIM'] = [0 for _ in range(self.config['test']['rollout_times'])]
            elif metric == 'PSNR':
                psnr_loss = PSNRLoss()
                loss_functions['PSNR'] = psnr_loss
                running_losses['PSNR'] = [0 for _ in range(self.config['test']['rollout_times'])]
            elif metric == "BCE":
                bce_loss = nn.BCELoss()
                loss_functions['BCE'] = bce_loss
                running_losses['BCE'] = [0 for _ in range(self.config['test']['rollout_times'])]
            else:
                raise ValueError('Invalid metric')
        return loss_functions, running_losses


    def SSIM_loss_fn(self, output, chunk):
        output = (output - output.min()) / (output.max() - output.min())
        chunk = (chunk - chunk.min()) / (chunk.max() - chunk.min())
        if output.shape[2] == 1:
            output = output.repeat(1, 1, 3, 1, 1)
            chunk = chunk.repeat(1, 1, 3, 1, 1)
        ssim_values = torch.zeros(len(output), self.config['seq_length'], device=self.device)
        for i in range(self.config['seq_length']):
            ssim_values[:, i] = self.ssim_loss(output[:, i], chunk[:, i])
        loss = ssim_values.mean(dim=1)
        return loss



class Trainer(Engine):

    def __init__(self, rank, config, dataset, model, epochs):
        super().__init__(rank, config, dataset, model)
        self.epochs = epochs
        self.init_training_components() 
        self.train_losses = {}
        

    def train_epoch(self, epoch):

        torch.manual_seed(epoch)
        self.model.train()

        total_predict_loss = 0.0
        total_rollout_loss = 0.0

        for _, sample in enumerate(self.dataloader):
            origin = sample["Input"].float().to(self.device)
            target = sample["Target"].float().to(self.device)

            target_chunks = torch.chunk(target, self.config['train']['rollout_times'], dim=1)

            with torch.cuda.amp.autocast():
                origin = self.mask(origin)
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

            total_predict_loss += predict_loss.item()
            total_rollout_loss += rollout_loss.item()

        if self.rank == 0:
            average_predict_loss = total_predict_loss / len(self.dataloader.dataset)
            average_rollout_loss = total_rollout_loss / len(self.dataloader.dataset)

            loss_data = {'predict_loss': average_predict_loss, 'rollout_loss': average_rollout_loss}
            # self.train_losses[epoch] = loss_data
            # self.save_losses(self.train_losses, self.config['train']['save_loss'], 'train_losses.json')
            self.save_losses(epoch, loss_data, self.config['train']['save_loss'], 'train_losses.json')

            self.save_checkpoint(epoch)
            self.scheduler.step()
            # wandb.log(loss_data)


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


    def load_checkpoint(self, resume_epoch):
        checkpoint_path = self.config['train']['save_checkpoint'] + f'checkpoint_{resume_epoch-1}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])


    def init_loss_function(self):
        # Set up loss function
        if self.config['train']['loss_fn'] == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif self.config['train']['loss_fn'] == 'L1':
            self.loss_fn = nn.L1Loss()
        elif self.config['train']['loss_fn'] == 'BCE':
            self.loss_fn = nn.BCELoss()
        else:
            raise ValueError('Invalid loss function')
 

    def save_checkpoint(self, epoch):
        save_path = os.path.join(self.config['train']['save_checkpoint'], f'checkpoint_{epoch}')
        save_dict = {'model': self.model.state_dict(),
                     'optimizer': self.optimizer.state_dict(),
                     'scheduler': self.scheduler.state_dict(),
                     'scaler': self.scaler.state_dict(),
                     'epoch': epoch}
        torch.save(save_dict, save_path + '.pth') 


class Evaluator(Engine):

    def __init__(self, rank, config, dataset, model, test_flag=False):
        super(Evaluator, self).__init__(rank, config, dataset, model)
        self.test_flag = test_flag
        self.valid_losses = {}

    def evaluate_epoch(self, epoch):

        torch.manual_seed(epoch)
        self.model.eval()

        with torch.no_grad():

            loss_functions, running_losses = self.init_loss_function() 

            for i, sample in enumerate(self.dataloader):

                origin_before_masked = sample["Input"].float().to(self.device)
                origin_before_masked_ = copy.deepcopy(origin_before_masked)
                origin =  self.mask(origin_before_masked)
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
                    self.plot(origin_before_masked_, origin_, output_chunks, target_chunks, 
                              epoch, self.config['valid']['save_reconstruct'])
                    
                # Compute losses
                for metric, loss_fn in loss_functions.items():
                    if metric == 'SSIM':
                        loss = self.SSIM_loss_fn(output, chunk)
                        running_losses[metric][j] += loss.sum().item()
                    else:
                        loss = loss_fn(output, chunk)
                        running_losses[metric][j] += loss.item()

            chunk_losses = {}
            for metric, running_loss_list in running_losses.items():
                    total_loss = sum(running_loss_list)
                    average_loss = total_loss / len(self.dataloader.dataset)
                    chunk_losses[metric] = average_loss

            # self.valid_losses[epoch] = chunk_losses
            # self.save_losses(self.valid_losses, self.config['valid']['save_loss'], 'valid_losses.json')
            self.save_losses(epoch, chunk_losses, self.config['valid']['save_loss'], 'valid_losses.json')

