import os
import json
from matplotlib import pyplot as plt

def plot_rollout(origin, output_chunks, target_chunks, epoch, path):

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
    plt.savefig(os.path.join(path, f"{epoch}.png"))
    plt.close()


def save_losses(save_path, train_losses, valid_losses):
    """
    Saves the training and validation losses to a JSON file.
    
    Parameters:
    - save_path: The directory path where the losses will be saved.
    - train_losses: A list of training losses.
    - valid_losses: A list of validation losses.
    """
    # Ensure the save path exists
    os.makedirs(save_path, exist_ok=True)
    loss_data = {
        'train_losses': train_losses,
        'valid_losses': valid_losses, 
    }
    with open(os.path.join(save_path, 'losses.json'), 'w') as f: 
        json.dump(loss_data, f)