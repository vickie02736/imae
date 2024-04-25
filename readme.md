/data
- checkpoint_random
    * losses.json               # include train and valid losses
    * checkpoint_{epoch}.pth
    * checkpoint_{epoch}.tar
    * checkpoint_{epoch}.pth.tar
- shallow_water_simulation
- shallow_water_simulation_inner_rollout
- shallow_water_simulation_outer_rollout
- rec_random
    - valid             # visualisation during training
    - inner
        - {mask_ratio}
            * .png
            * mae_losses.json
            * mse_losses.json
    - outer
    - inner_rollout
    - outer_rollout
- rec_consecutive

