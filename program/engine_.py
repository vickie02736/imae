import os
import copy
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from program.utils.metrics import RMSELoss, PSNRLoss, SSIMLoss
from program.utils.tools import save_losses, mask
from matplotlib import pyplot as plt


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


class Trainer(Engine):

    def __init__(self, rank, config, dataset, model, epochs, resume_epoch):
        super().__init__(rank, config, dataset, model)
        self.epochs = epochs
        self.resume_epoch = resume_epoch
        self.init_loss_function()
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
                model.parameters(),
                lr == self.config['train']['learning_rate'])
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


class Evaluator(Engine):

    def __init__(self, rank, config, dataset, model, test_flag=False):
        super(Evaluator, self).__init__(rank, config, dataset, model)
        self.test_flag = test_flag
        self.loss_functions, self.running_losses = self.init_eval_metrics()
        self.valid_losses = {}

    def init_eval_metrics(self):
        loss_functions = {}
        running_losses = {}
        if self.test_flag:
            metrics = self.config['test']['metric']
            self.rollout_times = self.config['test']['rollout_times']
        else:
            metrics = self.config['valid']['metric']
            self.rollout_times = self.config['valid']['rollout_times']
        for metric in metrics:
            if metric == 'MSE':
                mse_loss = nn.MSELoss()
                loss_functions['MSE'] = mse_loss
                running_losses['MSE'] = [0 for _ in range(self.rollout_times)]
            elif metric == 'MAE':
                mae_loss = nn.L1Loss()
                loss_functions['MAE'] = mae_loss
                running_losses['MAE'] = [0 for _ in range(self.rollout_times)]
            elif metric == 'RMSE':
                rmse_loss = RMSELoss()
                loss_functions['RMSE'] = rmse_loss
                running_losses['RMSE'] = [0 for _ in range(self.rollout_times)]
            elif metric == 'SSIM':
                ssim_loss = SSIMLoss(self.device).forward
                loss_functions['SSIM'] = ssim_loss
                running_losses['SSIM'] = [0 for _ in range(self.rollout_times)]
            elif metric == 'PSNR':
                psnr_loss = PSNRLoss()
                loss_functions['PSNR'] = psnr_loss
                running_losses['PSNR'] = [0 for _ in range(self.rollout_times)]
            elif metric == "BCE":
                bce_loss = nn.BCEWithLogitsLoss()
                loss_functions['BCE'] = bce_loss
                running_losses['BCE'] = [0 for _ in range(self.rollout_times)]
            else:
                raise ValueError('Invalid metric')
        return loss_functions, running_losses


class ImaeEngine(Trainer, Evaluator):

    def __init__(self,
                 rank,
                 config,
                 train_dataset,
                 valid_dataset,
                 model,
                 epochs,
                 resume_epoch,
                 test_flag=False):
        Trainer.__init__(self, rank, config, train_dataset, model, epochs,
                         resume_epoch)
        Evaluator.__init__(self, rank, config, valid_dataset, model, test_flag)
        self.load_checkpoint()

    def train_epoch(self, epoch):
        torch.manual_seed(epoch)
        self.model.train()
        total_predict_loss = 0.0
        total_rollout_loss = 0.0

        for i, sample in enumerate(self.dataloader):
            origin, _ = mask(sample["Input"],
                             mask_mtd=self.config["mask_method"])
            origin = origin.float().to(self.device)
            target = sample["Target"].float().to(self.device)
            target_chunks = torch.chunk(target,
                                        self.config['train']['rollout_times'],
                                        dim=1)

            with torch.cuda.amp.autocast():
                output = self.model(origin)
                predict_loss = self.loss_fn(output, target_chunks[0])
                output = self.model(output)
                rollout_loss = self.loss_fn(output, target_chunks[1])
                loss = predict_loss + rollout_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=0.1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_predict_loss += predict_loss.item()
            total_rollout_loss += rollout_loss.item()

        if self.rank == 0:
            average_predict_loss = total_predict_loss / len(
                self.dataloader.dataset)
            average_rollout_loss = total_rollout_loss / len(
                self.dataloader.dataset)

            loss_data = {
                'predict_loss': average_predict_loss,
                'rollout_loss': average_rollout_loss
            }
            save_losses(
                epoch, loss_data,
                os.path.join(self.config['imae']['save_loss'],
                             'train_losses.json'))
            self.save_checkpoint(
                epoch,
                os.path.join(self.config['imae']['save_checkpoint'],
                             f'checkpoint_{epoch}.pth'))

    def evaluate_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():

            for i, sample in enumerate(self.dataloader):
                origin_before_masked = copy.deepcopy(sample["Input"])
                origin, _ = mask(sample["Input"],
                                 mask_mtd=self.config["mask_method"])
                origin_plot = copy.deepcopy(origin)
                origin = origin.float().to(self.device)
                target = sample["Target"].float().to(self.device)
                target_chunks = torch.chunk(target, self.rollout_times, dim=1)

                output_chunks = []
                for j, chunk in enumerate(target_chunks):
                    if j == 0:
                        output = self.model(origin)
                        output_chunks.append(output)
                    else:
                        output = self.model(output)
                        output_chunks.append(output)

                if i == 1:
                    self.plot(epoch, origin_before_masked, origin_plot,
                              output_chunks, target_chunks,
                              self.config['valid']['rollout_times'],
                              self.config['seq_length'],
                              self.config['imae']['save_reconstruct'])

                # Compute losses
                for metric, loss_fn in self.loss_functions.items():
                    loss = loss_fn(output, chunk)
                    self.running_losses[metric][j] += loss.item()

            chunk_losses = {}
            for metric, running_loss_list in self.running_losses.items():
                total_loss = sum(running_loss_list)
                average_loss = total_loss / len(self.dataloader.dataset)
                chunk_losses[metric] = average_loss
            save_losses(
                epoch, chunk_losses,
                os.path.join(self.config['imae']['save_loss'],
                             'valid_losses.json'))

    def load_checkpoint(self):
        if self.resume_epoch == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.config['imae']['save_checkpoint'],
                             'init.pth'))
            losses = {}
            with open(
                    os.path.join(self.config['imae']['save_loss'],
                                 'train_losses.json'), 'w') as file:
                json.dump(losses, file)
            with open(
                    os.path.join(self.config['imae']['save_loss'],
                                 'valid_losses.json'), 'w') as file:
                json.dump(losses, file)
        else:
            checkpoint_path = os.path.join(
                self.config['imae']['save_checkpoint'] +
                f'checkpoint_{self.resume_epoch-1}.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])

    def plot(self, idx, origin, masked_origin, output_chunks, target_chunks,
             rollout_times, seq_len, save_path):
        _, ax = plt.subplots(rollout_times * 2 + 2,
                             seq_len,
                             figsize=(seq_len * 2+5, rollout_times * 4 + 2))
        row_titles = [
            "Original input", "Masked input", "Direct prediction", "Target",
            "Rollout prediction", "Target"
        ]
        for i, title in enumerate(row_titles):
            ax[i][0].text(0.5,
                          0.5,
                          title,
                        #   rotation=90,
                          verticalalignment='center',
                          horizontalalignment='center',
                          fontsize=12)
            ax[i][0].axis('off')
        for j in range(seq_len):
            # visualise input
            ax[0][j].imshow(origin[0][j][0].cpu().detach().numpy())
            ax[0][j].set_xticks([])
            ax[0][j].set_yticks([])
            ax[0][j].set_title("Timestep {timestep}".format(timestep=j + 1),
                               fontsize=10)
            # visualise masked input
            ax[1][j].imshow(masked_origin[0][j][0].cpu().detach().numpy())
            ax[1][j].set_xticks([])
            ax[1][j].set_yticks([])
            ax[1][j].set_title(
                "Timestep {timestep} (Masked Input)".format(timestep=j + 1),
                fontsize=10)
        for k in range(rollout_times):
            for j in range(seq_len):
                # visualise output
                ax[2 * k + 2][j].imshow(
                    output_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 2][j].set_xticks([])
                ax[2 * k + 2][j].set_yticks([])
                ax[2 * k + 2][j].set_title(
                    "Timestep {timestep} (Prediction)".format(
                        timestep=j + (k + 1) * seq_len),
                    fontsize=10)
                # visualise target
                ax[2 * k + 3][j].imshow(
                    target_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 3][j].set_xticks([])
                ax[2 * k + 3][j].set_yticks([])
                ax[2 * k + 3][j].set_title(
                    "Timestep {timestep} (Target)".format(timestep=j +
                                                          (k + 1) * seq_len),
                    fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{idx}.png"))
        plt.close()


class ConvLstmEngine(Trainer, Evaluator):

    def __init__(self,
                 rank,
                 config,
                 train_dataset,
                 valid_dataset,
                 model,
                 epochs,
                 resume_epoch,
                 interpolation,
                 test_flag=False):
        Trainer.__init__(self, rank, config, train_dataset, model, epochs,
                         resume_epoch)
        Evaluator.__init__(self, rank, config, valid_dataset, model, test_flag)

        self.interpolation = interpolation
        if self.interpolation == "linear":
            from program.utils.interpolation import linear_interpolation as interpolation_fn
        elif self.interpolation == "gaussian":
            from program.utils.interpolation import gaussian_interpolation as interpolation_fn
        self.interpolation_fn = interpolation_fn

        self.load_checkpoint()

    def train_epoch(self, epoch):
        torch.manual_seed(epoch)
        self.model.train()
        total_predict_loss = 0.0
        total_rollout_loss = 0.0

        for i, sample in enumerate(self.dataloader):
            origin, idx = mask(sample["Input"],
                               mask_mtd=self.config["mask_method"])
            origin = self.interpolation_fn(origin, idx)
            origin = origin.float().to(self.device)
            target = sample["Target"].float().to(self.device)
            target_chunks = torch.chunk(target,
                                        self.config['train']['rollout_times'],
                                        dim=1)

            with torch.cuda.amp.autocast():
                output = self.model(origin)
                predict_loss = self.loss_fn(output, target_chunks[0])
                output = self.model(output)
                rollout_loss = self.loss_fn(output, target_chunks[1])
                loss = predict_loss + rollout_loss

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                           max_norm=0.1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_predict_loss += predict_loss.item()
            total_rollout_loss += rollout_loss.item()

        if self.rank == 0:
            average_predict_loss = total_predict_loss / len(
                self.dataloader.dataset)
            average_rollout_loss = total_rollout_loss / len(
                self.dataloader.dataset)

            loss_data = {
                'predict_loss': average_predict_loss,
                'rollout_loss': average_rollout_loss
            }
            save_losses(
                epoch, loss_data,
                os.path.join(self.config['convlstm']['save_loss'],
                             self.interpolation, 'train_losses.json'))
            self.save_checkpoint(
                epoch,
                os.path.join(self.config['convlstm']['save_checkpoint'],
                             self.interpolation, f'checkpoint_{epoch}.pth'))

    def evaluate_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():

            for i, sample in enumerate(self.dataloader):
                origin_before_masked = copy.deepcopy(sample["Input"])
                masked_origin, idx = mask(sample["Input"],
                                          mask_mtd=self.config["mask_method"])
                masked_plot = copy.deepcopy(masked_origin)
                origin = self.interpolation_fn(masked_origin, idx)
                interpolated_plot = copy.deepcopy(origin)
                print(interpolated_plot.shape)
                origin = origin.float().to(self.device)
                target = sample["Target"].float().to(self.device)
                target_chunks = torch.chunk(target, self.rollout_times, dim=1)

                output_chunks = []
                for j, chunk in enumerate(target_chunks):
                    if j == 0:
                        output = self.model(origin)
                        output_chunks.append(output)
                    else:
                        output = self.model(output)
                        output_chunks.append(output)

                if i == 1:
                    save_path = os.path.join(
                        self.config['convlstm']['save_reconstruct'],
                        self.interpolation)
                    self.plot(origin_before_masked, masked_plot,
                              interpolated_plot, output_chunks, target_chunks,
                              self.config['seq_length'], epoch, save_path)

                # Compute losses
                for metric, loss_fn in self.loss_functions.items():
                    loss = loss_fn(output, chunk)
                    self.running_losses[metric][j] += loss.item()

            chunk_losses = {}
            for metric, running_loss_list in self.running_losses.items():
                total_loss = sum(running_loss_list)
                average_loss = total_loss / len(self.dataloader.dataset)
                chunk_losses[metric] = average_loss
            save_losses(
                epoch, chunk_losses,
                os.path.join(self.config['convlstm']['save_loss'],
                             self.interpolation, 'valid_losses.json'))

    def load_checkpoint(self):
        if self.resume_epoch == 0:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.config['convlstm']['save_checkpoint'],
                             self.interpolation, 'init.pth'))
            losses = {}
            with open(
                    os.path.join(self.config['convlstm']['save_loss'],
                                 self.interpolation, 'train_losses.json'),
                    'w') as file:
                json.dump(losses, file)
            with open(
                    os.path.join(self.config['convlstm']['save_loss'],
                                 self.interpolation, 'valid_losses.json'),
                    'w') as file:
                json.dump(losses, file)
        else:
            checkpoint_path = os.path.join(
                self.config['convlstm']['save_checkpoint'], self.interpolation,
                f'checkpoint_{self.resume_epoch-1}.pth')
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.scaler.load_state_dict(checkpoint['scaler'])

    def plot(self, origin, masked_origin, interpolated_origin, output_chunks,
             target_chunks, seq_len, idx, save_path):
        rollout_times = self.config['train']['rollout_times']
        _, ax = plt.subplots(rollout_times * 2 + 3,
                             seq_len,
                             figsize=(seq_len * 2, rollout_times * 4 + 2))
        for j in range(seq_len):
            # visualise input
            ax[0][j].imshow(origin[0][j][0].cpu().detach().numpy())
            ax[0][j].set_xticks([])
            ax[0][j].set_yticks([])
            ax[0][j].set_title(
                "Timestep {timestep} (Input)".format(timestep=j + 1),
                fontsize=10)
            # visualise masked input
            ax[1][j].imshow(masked_origin[0][j][0].cpu().detach().numpy())
            ax[1][j].set_xticks([])
            ax[1][j].set_yticks([])
            ax[1][j].set_title(
                "Timestep {timestep} (Masked Input)".format(timestep=j + 1),
                fontsize=10)
            # visualise interpolated input
            ax[2][j].imshow(
                interpolated_origin[0][j][0].cpu().detach().numpy())
            ax[2][j].set_xticks([])
            ax[2][j].set_yticks([])
            ax[2][j].set_title(
                "Timestep {timestep} (Interpolated Input)".format(timestep=j +
                                                                  1),
                fontsize=10)
        for k in range(rollout_times):
            for j in range(seq_len):
                # visualise output
                ax[2 * k + 3][j].imshow(
                    output_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 3][j].set_xticks([])
                ax[2 * k + 3][j].set_yticks([])
                ax[2 * k + 3][j].set_title(
                    "Timestep {timestep} (Prediction)".format(
                        timestep=j + (k + 1) * seq_len),
                    fontsize=10)
                # visualise target
                ax[2 * k + 4][j].imshow(
                    target_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 4][j].set_xticks([])
                ax[2 * k + 4][j].set_yticks([])
                ax[2 * k + 4][j].set_title(
                    "Timestep {timestep} (Target)".format(timestep=j +
                                                          (k + 1) * seq_len),
                    fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{idx}.png"))
        plt.close()
