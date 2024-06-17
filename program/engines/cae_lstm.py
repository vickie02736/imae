import os
import copy
import json
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt

from models import ConvAutoencoder, LSTMPredictor
from .trainer import Trainer
from .evaluator import Evaluator
from utils import save_losses, mask

class CaeTrainer(Trainer, Evaluator):
    def __init__(self, rank, args, train_dataset, eval_dataset):
        Trainer.__init__(self, rank, args, train_dataset)
        Evaluator.__init__(self, rank, args, eval_dataset)
        self.load_model()   # Here
        self.setup()        # Engine
        self.init_training_components() # Trainer
        Trainer.load_checkpoint(self)

    def load_model(self):
        self.model = ConvAutoencoder(self.config)

    def train_epoch(self, epoch):
        torch.manual_seed(epoch)
        self.model.train()
        total_loss = 0.0

        for i, sample in enumerate(self.train_loader):
            origin = sample["Frame"].float().to(self.device)
            target = sample["Frame"].float().to(self.device)

            with torch.cuda.amp.autocast():
                output = self.model(origin)
                loss = self.loss_fn(output, target)

            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward(retain_graph=True)
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.1)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()

        if self.rank == 0:
            average_loss = total_loss / len(
                self.train_loader.dataset)
            loss_data = {
                'cae_loss': average_loss,
            }
            save_losses(
                epoch, loss_data,
                os.path.join(self.config['cae']['save_loss'], 'train_losses.json'))
            if epoch % self.args.save_frequency == 0:
                self.save_checkpoint(
                    epoch,
                    os.path.join(self.config['cae']['save_checkpoint'], f'checkpoint_{epoch}.pth'))

    def evaluate_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():

             for i, sample in enumerate(self.eval_loader):
                origin_plot = copy.deepcopy(sample["Input"])
                origin = sample["Input"].float().to(self.device)
                target = sample["Input"].float().to(self.device)
                output = self.model(origin, sequence_input=True)

                if i == 1:
                    save_path = os.path.join(
                        self.config['cae']['save_reconstruct'])
                    self.plot(origin_plot, output, target, 
                              self.config['seq_length'], epoch, save_path)

                # Compute losses
                for metric, loss_fn in self.loss_functions.items():
                    loss = loss_fn(output, target)
                    self.running_losses[metric][0] += loss.item()

                chunk_losses = {}
                for metric, running_loss_list in self.running_losses.items():
                    total_loss = sum(running_loss_list)
                    average_loss = total_loss / len(self.eval_loader.dataset)
                    chunk_losses[metric] = average_loss
                save_losses(
                    epoch, chunk_losses,
                    os.path.join(self.config['cae']['save_loss'], 'valid_losses.json'))


    def plot(self, input, output, target, seq_len, idx, save_path):
        rollout_times = self.config['train']['rollout_times']
        _, ax = plt.subplots(3, seq_len + 1, figsize=(seq_len * 2 + 2, 6))
        row_titles = ["Input", "Output", "Target"]
        for i, title in enumerate(row_titles):
            ax[i][0].text(1.0,
                          0.5,
                          title,
                          verticalalignment='center',
                          horizontalalignment='right',
                          fontsize=12)
            ax[i][0].axis('off')
        for j in range(seq_len):
            # visualise input
            ax[0][j + 1].imshow(input[0][j][0].cpu().detach().numpy())
            ax[0][j + 1].set_xticks([])
            ax[0][j + 1].set_yticks([])
            ax[0][j + 1].set_title("Timestep {timestep}".format(timestep=j + 1),
                                   fontsize=10)
            # visualise output
            ax[1][j + 1].imshow(output[0][j][0].cpu().detach().numpy())
            ax[1][j + 1].set_xticks([])
            ax[1][j + 1].set_yticks([])
            ax[1][j + 1].set_title("Timestep {timestep}".format(timestep=j + 1),
                                   fontsize=10)
            # visualise target
            ax[2][j + 1].imshow(target[0][j][0].cpu().detach().numpy())
            ax[2][j + 1].set_xticks([])
            ax[2][j + 1].set_yticks([])
            ax[2][j + 1].set_title("Timestep {timestep}".format(timestep=j + 1),
                                   fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{idx}.png"))
        plt.close()




class CaeLstmTrainer(Trainer, Evaluator):

    def __init__(self,
                 rank,
                 args,
                 train_dataset,
                 eval_dataset):
        Trainer.__init__(self, rank, args, train_dataset)
        Evaluator.__init__(self, rank, args, eval_dataset)
        self.load_cae()
        self.load_model()
        self.setup()
        self.init_training_components()
        Trainer.load_checkpoint(self)

        if self.args.interpolation == "linear":
            from utils import linear_interpolation as interpolation_fn
        elif self.args.interpolation == "gaussian":
            from utils import gaussian_interpolation as interpolation_fn
        self.interpolation_fn = interpolation_fn

    def load_model(self):
        self.model = LSTMPredictor(self.config)

    def load_cae(self):
        self.cae_model = ConvAutoencoder(self.config).to(self.device)
        self.cae_model = DDP(self.cae_model, device_ids=[self.rank])
        load_epoch = self.config['cae_lstm']['cae_load_epoch']
        checkpoint_path = os.path.join(
            self.config['cae']['save_checkpoint'], f'checkpoint_{load_epoch}.pth')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.cae_model.load_state_dict(checkpoint['model'])
        self.cae_model = self.cae_model.module
        self.cae_model.eval()

    def train_epoch(self, epoch):
        torch.manual_seed(epoch)
        self.model.train()
        total_predict_loss = 0.0
        total_rollout_loss = 0.0

        for i, sample in enumerate(self.train_loader):
            print(f"Epoch {epoch}, batch {i}")
            origin, idx = mask(sample["Input"],
                               mask_mtd=self.config["mask_method"])
            # origin = self.interpolation_fn(origin, idx)
            origin = origin.float().to(self.device)
            origin = self.cae_model.encoder(origin)
            origin = self.interpolation_fn(origin, idx)
            target = sample["Target"].float().to(self.device)
            target = self.cae_model.encoder(target)
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
                self.train_loader.dataset)
            average_rollout_loss = total_rollout_loss / len(
                self.train_loader.dataset)

            loss_data = {
                'predict_loss': average_predict_loss,
                'rollout_loss': average_rollout_loss
            }
            save_losses(
                epoch, loss_data,
                os.path.join(self.config['convlstm']['save_loss'],
                             self.args.interpolation, 'train_losses.json'))
            if epoch % self.args.save_frequency == 0:
                self.save_checkpoint(
                    epoch,
                    os.path.join(self.config['convlstm']['save_checkpoint'],
                                self.args.interpolation, f'checkpoint_{epoch}.pth'))

    def evaluate_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():
            for i, sample in enumerate(self.eval_loader):
                print(f"Epoch {epoch}, valid batch {i}")
                origin_before_masked = copy.deepcopy(sample["Input"]) # origin_before_masked: full space
                origin, idx = mask(sample["Input"], 
                                          mask_mtd=self.config["mask_method"]) 
                masked_plot = copy.deepcopy(origin) # masked_plot: full space
                origin = origin.float().to(self.device)
                latent_origin = self.cae_model.encoder(origin)
                latent_origin = self.interpolation_fn(latent_origin, idx)

                target = sample["Target"].float().to(self.device)
                # latent_target = self.cae_model.encoder(target)
                
                target_chunks = torch.chunk(target, self.rollout_times, dim=1)
                # latent_target_chunks = torch.chunk(latent_target, self.rollout_times, dim=1)

                output_chunks = []
                # latent_output_chunks = []
                for j, chunk in enumerate(target_chunks):
                    if j == 0:
                        latent_output = self.model(latent_origin)
                    else:
                        latent_output = self.model(latent_output)
                    # output_chunks.append(latent_output)
                    output = self.cae_model.decoder(latent_output)
                    output_chunks.append(output)

                    # Compute losses
                    for metric, loss_fn in self.loss_functions.items():
                        loss = loss_fn(output, chunk)
                        self.running_losses[metric][j] += loss.item()                
                
                if i == 1:
                    save_path = os.path.join(
                        self.config['convlstm']['save_reconstruct'],
                        self.args.interpolation)
                    self.plot(origin_before_masked, masked_plot, output_chunks, target_chunks,
                              self.config['seq_length'], epoch, save_path)

            chunk_losses = {}
            for metric, running_loss_list in self.running_losses.items():
                total_loss = sum(running_loss_list)
                average_loss = total_loss / len(self.eval_loader.dataset)
                chunk_losses[metric] = average_loss
            save_losses(
                epoch, chunk_losses,
                os.path.join(self.config['convlstm']['save_loss'],
                             self.args.interpolation, 'valid_losses.json'))


    def plot(self, origin, masked_origin, output_chunks, target_chunks, seq_len, idx, save_path):
        rollout_times = self.config['train']['rollout_times']
        _, ax = plt.subplots(rollout_times * 2 + 3,
                             seq_len + 1,
                             figsize=(seq_len * 2 + 2, rollout_times * 4 + 6))
        row_titles = [
            "Original input", "Masked input", "Interpolated input",
            "Direct prediction", "Target", "Rollout prediction", "Target"
        ]
        for i, title in enumerate(row_titles):
            ax[i][0].text(1.0,
                          0.5,
                          title,
                          verticalalignment='center',
                          horizontalalignment='right',
                          fontsize=12)
            ax[i][0].axis('off')
        for j in range(seq_len):
            # visualise input
            ax[0][j + 1].imshow(origin[0][j][0].cpu().detach().numpy())
            ax[0][j + 1].set_xticks([])
            ax[0][j + 1].set_yticks([])
            ax[0][j + 1].set_title("Timestep {timestep}".format(timestep=j + 1),
                                   fontsize=10)
            # visualise masked input
            ax[1][j + 1].imshow(masked_origin[0][j][0].cpu().detach().numpy())
            ax[1][j + 1].set_xticks([])
            ax[1][j + 1].set_yticks([])
            ax[1][j + 1].set_title("Timestep {timestep}".format(timestep=j + 1),
                                   fontsize=10)
        for k in range(rollout_times):
            for j in range(seq_len):
                # visualise output
                ax[2 * k + 2][j + 1].imshow(
                    output_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 2][j + 1].set_xticks([])
                ax[2 * k + 2][j + 1].set_yticks([])
                ax[2 * k + 2][j + 1].set_title("Timestep {timestep}".format(
                    timestep=j + (k + 1) * seq_len + 1),
                                               fontsize=10)
                # visualise target
                ax[2 * k + 3][j + 1].imshow(
                    target_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 3][j + 1].set_xticks([])
                ax[2 * k + 3][j + 1].set_yticks([])
                ax[2 * k + 3][j + 1].set_title("Timestep {timestep}".format(
                    timestep=j + (k + 1) * seq_len + 1),
                                               fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{idx}.png"))
        plt.close()