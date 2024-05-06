import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from metrics import RMSELoss

from matplotlib import pyplot as plt


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

    def save_losses(self, loss_dic, save_path, save_name):
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, save_name), 'w') as f: 
            json.dump(loss_dic, f)



class Trainer(Engine):

    def __init__(self, rank, config, dataset, model, epochs):
        super().__init__(rank, config, dataset, model)
        self.epochs = epochs
        self.init_training_components() 
        

    def train_epoch(self, epoch):

        torch.manual_seed(epoch)
        self.model.train()

        running_loss = 0.0
        running_loss_rollout = 0.0
        train_losses = []
        train_losses_rollout = []

        for _, sample in enumerate(self.dataloader):
            origin = sample["Input"].float().to(self.device)
            target = sample["Target"].float().to(self.device)
            target_chunks = torch.chunk(target, 2, dim=1)

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
                self.save_checkpoint(epoch)
            
            loss_data = {'train_losses': train_losses, 'train_losses_rollout': train_losses_rollout}

            self.save_losses(loss_data, self.config['train']['save_loss'], 'train_losses.json')
            # return {'train_losses': train_losses, 'train_losses_rollout': train_losses_rollout}


    def init_training_components(self): 
        # Set up loss function, optimizer, scheduler
        T_start = self.epochs * 0.05 * self.len_dataset // self.config['batch_size']
        T_start = int(T_start)
        learning_rate = 1e-4
        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, betas=(0.9, 0.95), weight_decay=0.03)
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_start, eta_min=1e-6, last_epoch=-1)
        self.scaler = torch.cuda.amp.GradScaler()


    def init_loss_function(self):
        # Set up loss function
        if self.config['train']['loss_fn'] == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif self.config['train']['loss_fn'] == 'L1':
            self.loss_fn = nn.L1Loss()
        else:
            raise ValueError('Invalid loss function')
 

    def load_checkpoint(self, resume_epoch):
        checkpoint_path = self.config['train']['save_checkpoint'] + f'checkpoint_{resume_epoch-1}.pth'
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = self.model.to(self.device)
        self.model = DDP(self.model, device_ids=[self.rank])
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.scaler.load_state_dict(checkpoint['scaler'])


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

    def evaluate_epoch(self, epoch, rec_savepath, mask_ratio=None):
        self.model.eval()
        with torch.no_grad():
            loss_functions, running_losses = self.init_loss_function()  # Ensure this is correctly initialized

            for i, sample in enumerate(self.dataloader):
                origin = sample["Input"].float().to(self.device)
                target = sample["Target"].float().to(self.device)
                origin_copy = copy.deepcopy(origin)
                target_chunks = torch.chunk(target, self.config['test']['rollout_times'], dim=1)
                output_chunks = []

                for j, chunk in enumerate(target_chunks):
                    if j == 0:
                        if self.test_flag == False:
                            mask_ratio = torch.rand(1).item()
                        else: 
                            mask_ratio = mask_ratio
                        num_mask = int(mask_ratio * self.config['seq_length'])
                        output = self.model(origin, num_mask)
                        output_copy = copy.deepcopy(output)
                    else:
                        output = self.model(origin, 0)
                        output_copy = copy.deepcopy(output)
                    output_chunks.append(output_copy)
                    origin = output  # Make sure to use updated origin

                    for metric, loss_fn in loss_functions.items():
                        loss = loss_fn(output, chunk)
                        running_losses[metric][j] += loss.item()

                if i == 1:  # This condition might need adjusting based on when you want to plot
                    self.plot_rollout(origin_copy, output_chunks, target_chunks, epoch, rec_savepath)

            chunk_losses = {}
            for metric, running_loss_list in running_losses.items():
                total_loss = sum(running_loss_list)
                average_loss = total_loss / len(self.dataloader.dataset)
                chunk_losses[metric] = average_loss

            valid_losses = {f'{epoch}': chunk_losses}
            # return chunk_losses 
            self.save_losses(valid_losses, self.config['valid']['save_loss'], 'valid_losses.json')


    def plot_rollout(self, origin, output_chunks, target_chunks, idx, save_path):
        rollout_times = len(output_chunks)
        _, ax = plt.subplots(rollout_times*2+1, 10, figsize=(20, rollout_times*4+2))
        for j in range(10): 
            # visualise input
            ax[0][j].imshow(origin[0][j][0].cpu().detach().numpy())
            ax[0][j].set_xticks([])
            ax[0][j].set_yticks([])
            ax[0][j].set_title("Timestep {timestep} (Input)".format(timestep=j+1), fontsize=10)
        for k in range(rollout_times): 
            for j in range(10):
                # visualise output
                ax[2*k+1][j].imshow(output_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2*k+1][j].set_xticks([])
                ax[2*k+1][j].set_yticks([])
                ax[2*k+1][j].set_title("Timestep {timestep} (Prediction)".format(timestep=j+11+k*10), fontsize=10)
                # visualise target
                ax[2*k+2][j].imshow(target_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2*k+2][j].set_xticks([])
                ax[2*k+2][j].set_yticks([])
                ax[2*k+2][j].set_title("Timestep {timestep} (Target)".format(timestep=j+11+k*10), fontsize=10)
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
                # Assuming SSIM loss function initialization if you have one
                pass
            elif metric == 'PSNR':
                # Assuming PSNR loss function initialization if you have one
                pass
            else:
                raise ValueError('Invalid metric')
        return loss_functions, running_losses

