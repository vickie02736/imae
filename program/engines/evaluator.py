import torch.nn as nn
from program.utils.metrics import RMSELoss, PSNRLoss, SSIMLoss
from program.engines.engine import Engine


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