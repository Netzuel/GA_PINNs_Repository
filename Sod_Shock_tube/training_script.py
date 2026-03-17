# Import modules
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

sys.path.insert(1, "../")
import models
import utils

# ==== Load configuration file ====
try:
    with open("config.json") as file:
        config = json.load(file)
except:
    print("Error: The configuration file does not exist or it contains errors.")
    sys.exit(1)


DTYPE, device = (
    eval(config["training_process"]["DTYPE"]),
    torch.device(config["training_process"]["device"]),
)

# ==== Seeds ====
torch.manual_seed(config["training_process"]["parameters"]["random_seed"])
np.random.seed(config["training_process"]["parameters"]["random_seed"])

# ==== Data generation ====
X, X_0 = utils.generate_domain(config)
X, X_0 = X.to(DTYPE).to(device), X_0.to(DTYPE).to(device)
print("X_0 (initial) shape: ", X_0.shape)
print("X (internal) shape: ", X.shape)


# ==== Define initial conditions ====
U_0 = utils.initial_conditions(X_0[:, 1:2], config).to(device)
print("U_0 shape: ", U_0.shape)

model = models.GA_PINN(config).to(device)
analytical_space, analytical_solution = utils.load_analytical(config)
model.analytical_space, model.analytical_solution = (
    analytical_space.to(DTYPE).to(device),
    analytical_solution.to(DTYPE).to(device),
)


# ==== Define optimizer ====
optimizer = utils.define_optimizer(model, config)


# ==== Define number of epochs ====
epochs = config["training_process"]["parameters"]["epochs"]


print("Starting optimization...")
pbar = tqdm(range(epochs))
for epoch in pbar:
    model.epoch = epoch

    # ==== Compute loss and update model parameters ====
    optimizer.zero_grad(set_to_none=True)
    ℒ = utils.compute_ℒ(model, X.view(-1, 2), X_0, U_0)
    ℒ.backward(retain_graph=False)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    # ==== Compute l2 error ====
    utils.compute_l2(model)

    # ==== Logging ====
    pbar.set_postfix(
        {
            "ℒ_ic": model.histories['ℒ_ic_hist'][-1],
            "ℒ": model.histories['ℒ_hist'][-1],
            "l2": model.histories['l2_hist'][-1],
            "AUC": model.histories['AUC_hist'][-1]
        }
    )
    if epoch % config["training_process"]["export"]["save_each_data"] == 0:
        utils.save_results(model, config)
    if epoch % config["training_process"]["export"]["save_each_images"] == 0:
        utils.plot_results(model, config)
