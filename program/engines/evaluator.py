import torch.nn as nn
from program.utils.metrics import RMSELoss, PSNRLoss, SSIMLoss
from program.engines.engine import Engine
from torch.utils.data import DataLoader, DistributedSampler


class Evaluator(Engine):
    def __init__(self, rank, args, eval_dataset):
        super(Evaluator, self).__init__(rank, args)
        self.eval_dataset = eval_dataset
        self.init_eval_dataloader()

        self.loss_functions, self.running_losses = self.init_eval_metrics()

    def init_eval_dataloader(self):
        sampler = DistributedSampler(self.eval_dataset, num_replicas=self.world_size, rank=self.rank)
        self.eval_loader = DataLoader(self.eval_dataset, batch_size=self.config[self.args.model_name]['batch_size'],
                                pin_memory=True, shuffle=False, drop_last=True, sampler=sampler)

    def init_eval_metrics(self):
        loss_functions = {}
        running_losses = {}
        if self.args.test_flag:
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
