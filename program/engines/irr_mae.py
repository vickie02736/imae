import os
import copy
import json
import torch
from matplotlib import pyplot as plt

from program.engines.trainer import Trainer
from program.engines.evaluator import Evaluator
from program.utils.tools import save_losses, mask


class ImaeTrainer(Trainer, Evaluator):

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

        for _, sample in enumerate(self.train_loader):
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
                self.train_loader.dataset)
            average_rollout_loss = total_rollout_loss / len(
                self.train_loader.dataset)

            loss_data = {
                'predict_loss': average_predict_loss,
                'rollout_loss': average_rollout_loss
            }
            save_losses(
                epoch, loss_data,
                os.path.join(self.config['imae']['save_loss'],
                             'train_losses.json'))
            if epoch %20 == 0:
                self.save_checkpoint(
                    epoch,
                    os.path.join(self.config['imae']['save_checkpoint'],
                                f'checkpoint_{epoch}.pth'))

    def evaluate_epoch(self, epoch):
        self.model.eval()
        with torch.no_grad():

            for i, sample in enumerate(self.valid_loader):
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
                average_loss = total_loss / len(self.valid_loader.dataset)
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
                             seq_len + 1,
                             figsize=(seq_len * 2 + 2, rollout_times * 4 + 4))
        row_titles = [
            "Original input", "Masked input", "Direct prediction", "Target",
            "Rollout prediction", "Target"
        ]
        for i, title in enumerate(row_titles):
            ax[i][0].text(
                1.0,
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
                    timestep=j + (k + 1) * seq_len),
                                               fontsize=10)
                # visualise target
                ax[2 * k + 3][j + 1].imshow(
                    target_chunks[k][0][j][0].cpu().detach().numpy())
                ax[2 * k + 3][j + 1].set_xticks([])
                ax[2 * k + 3][j + 1].set_yticks([])
                ax[2 * k + 3][j + 1].set_title("Timestep {timestep}".format(
                    timestep=j + (k + 1) * seq_len),
                                               fontsize=10)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"{idx}.png"))
        plt.close()
