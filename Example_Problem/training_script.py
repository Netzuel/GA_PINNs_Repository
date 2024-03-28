# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
import math
import torch
import torch.nn as nn
import json
sys.path.insert(1, "../")
import utils, models

# Load configuration file
try:
	with open("../config.json") as file:
		config = json.load(file)
except:
	print("Error: The configuration file does not exist or it contains errors.")
	sys.exit(1)


DTYPE, device = eval(config["training_process"]["DTYPE"]), torch.device(config["training_process"]["device"])

# Seeds
torch.manual_seed(config["training_process"]["parameters"]["random_seed"])
np.random.seed(config["training_process"]["parameters"]["random_seed"])

# Generation of data.
X, X_0 = utils.generate_domain(config)
X, X_0 = X.to(DTYPE).to(device), X_0.to(DTYPE).to(device)
print("X_0 (initial) shape: ", X_0.shape)
print("X (internal) shape: ", X.shape)


# Define initial conditions
U_0 = utils.initial_conditions(X_0[:,1:2], config).to(device)
print("U_0 shape: ", U_0.shape)

model = models.GA_PINN(config).to(device)
analytical_space, analytical_solution = utils.load_analytical(config)
model.analytical_space, model.analytical_solution = analytical_space.to(DTYPE).to(device), analytical_solution.to(DTYPE).to(device)


# Define learning rate.
lr = config["training_process"]["parameters"]["learning_rate"]
# Define number of epochs.
epochs = config["training_process"]["parameters"]["epochs"]
# Define the optimizer.
## Define the optimizer
if config["training_process"]["parameters"]["optimizer"] == "RAdam":
	optimizer = torch.optim.RAdam(model.parameters(), lr = lr)
elif config["training_process"]["parameters"]["optimizer"] == "AdamW":
	optimizer = torch.optim.AdamW(model.parameters(), lr = lr)
elif config["training_process"]["parameters"]["optimizer"] == "Adam":
	optimizer = torch.optim.Adam(model.parameters(), lr = lr)
elif config["training_process"]["parameters"]["optimizer"] == "SGD":
	optimizer = torch.optim.SGD(model.parameters(), lr = lr, nesterov = True, momentum = 0.9, dampening = 0)
else:
	sys.exit("Optimizer is invalid. Only the following are available at this moment:\n-'RAdam'\n-'AdamW'\n-'Adam'\n-'SGD'")


print("Starting optimization...")
pbar = tqdm(range(epochs))
for epoch in pbar:
	model.epoch = epoch
	model.train_step(X.view(-1,2), X_0, U_0, optimizer)
	pbar.set_postfix({'loss_ic' : model.loss_ic_hist[-1], 'loss_total' : model.loss_hist[-1]})
	if epoch % config["training_process"]["export"]["save_each_data"] == 0:
		utils.save_results(model, config)
		torch.save(model.state_dict(), config["training_process"]["export"]["path_models"] + "model_epoch_" + str(model.epoch) + ".pt")
	if epoch % config["training_process"]["export"]["save_each_images"] == 0:
		utils.plot_results(model, config)












































