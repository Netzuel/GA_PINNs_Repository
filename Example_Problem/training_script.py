# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import math
import torch
import torch.nn as nn
import torch.optim as optim
import json
from scipy.stats import qmc
# Import utils
sys.path.insert(1, "../")
import custom_activations, models, utils

# Load configuration file
with open("../config.json") as file:
	config = json.load(file)
# Specify 'device' for PyTorch as 'cpu' or 'cuda' if any GPU is available.
print("Is PyTorch using GPU?", torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set data type:
DTYPE = torch.float32
# Seeds
torch.manual_seed( int(config["training_process"]["parameters"]["random_seed"]) )
np.random.seed( int(config["training_process"]["parameters"]["random_seed"]) )

# GENERATION OF DATA
tmin, tmax = config["physical"]["parameters"]["temporal_range"][0], config["physical"]["parameters"]["temporal_range"][1]
xmin, xmax = config["physical"]["parameters"]["spatial_range"][0], config["physical"]["parameters"]["spatial_range"][1]
N_r, N_0 = eval(config["physical"]["parameters"]["N_r"]), eval(config["physical"]["parameters"]["N_0"])

# Load analytical resolution:
hf = h5py.File(config["training_process"]["import"]["analytical_solution_path"], "r")
x_calculated = np.array(hf.get("x_space"))
rho_calculated = np.array(hf.get("dens_calculated"))
u_calculated = np.array(hf.get("ur_calculated"))
p_calculated = np.array(hf.get("p_calculated"))
hf.close()
## Convert analytical data to torch tensors:
x_calculated_tensor = torch.tensor( x_calculated.reshape((-1,1)), dtype = DTYPE ).to(device)
t_calculated_tensor = torch.tensor( tmax, dtype = DTYPE ).repeat( (x_calculated.shape[0],1) ).to(device)
rho_calculated_tensor = torch.tensor( rho_calculated.reshape((-1,1)), dtype = DTYPE ).to(device)
u_calculated_tensor = torch.tensor( u_calculated.reshape((-1,1)), dtype = DTYPE ).to(device)
p_calculated_tensor = torch.tensor( p_calculated.reshape((-1,1)), dtype = DTYPE ).to(device)
### Group the analytical resolution as tensors
ground_truth = torch.cat( (rho_calculated_tensor, u_calculated_tensor, p_calculated_tensor), dim = 1 )
physical_space = torch.cat( (t_calculated_tensor, x_calculated_tensor), dim = 1 )

if config["physical"]["parameters"]["sampling_method"] == "sobol":
	## Internal data
	sampler = qmc.Sobol(d = 2, scramble = False)
	sample = sampler.random_base2( m = int(math.log(N_r, 2)) )
	l_bounds, u_bounds = [tmin + (tmax - tmin)/sample.shape[0], xmin], [tmax, xmax]
	sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
	t_internal = torch.tensor(sample_scaled[:,0:1], dtype = DTYPE, requires_grad = True).to(device)
	x_internal = torch.tensor(sample_scaled[:,1:2], dtype = DTYPE, requires_grad = True).to(device)
	X_r = torch.cat((t_internal,x_internal), dim = 1).to(device)
	print("Shape internal points: ", X_r.shape)
	## Initial data
	t_initial = torch.tensor(tmin, dtype = DTYPE, requires_grad = True).repeat((N_0,1)).to(device)
	x_initial = torch.reshape(torch.tensor(np.linspace(xmin, xmax, N_0), dtype = DTYPE), (N_0,1)).to(device)
	X_0 = torch.cat((t_initial,x_initial), dim = 1).to(device)
	print("Shape initial points: ", X_0.shape)
else:
	sys.exit("Sampling method is invalid. Only the following are available at this moment:\n-'sobol'")

# Define initial conditions
def fun_U_0(x, config = config):
	x_numpy = x.detach().cpu().numpy()
	rho_l, rho_r = config["physical"]["initial_conditions"]["density"]
	u_l, u_r = config["physical"]["initial_conditions"]["velocity"]
	p_l, p_r = config["physical"]["initial_conditions"]["pressure"]
	
	ic_rho = lambda x: (rho_l) * (x <= 0.5) + (rho_r) * (x > 0.5)
	ic_u = lambda x: (u_l) * (x <= 0.5) + (u_r) * (x > 0.5)
	ic_p = lambda x: (p_l) * (x <= 0.5) + (p_r) * (x > 0.5)
	
	W_tensor = torch.tensor(1/(1 - ic_u(x_numpy)**2)**(1/2), dtype = DTYPE, requires_grad = True).to(device)
	rho_tensor = torch.tensor(ic_rho(x_numpy), dtype = DTYPE, requires_grad = True).to(device)
	u_tensor = torch.tensor(ic_u(x_numpy), dtype = DTYPE, requires_grad = True).to(device)
	p_tensor = torch.tensor(ic_p(x_numpy), dtype = DTYPE, requires_grad = True).to(device)
	
	output_ICs = torch.cat((rho_tensor, u_tensor, p_tensor), dim = 1)
	return output_ICs.to(device)

# Initial values:
U_0 = fun_U_0(X_0[:,1:2])

# Training process
## Define the model
GA_PINN_model = models.GA_PINN( X_r = X_r, X_0 = X_0, U_0 = U_0, ground_truth = ground_truth, physical_space = physical_space, config = config, device = device, DTYPE = DTYPE )

print("Length of parameters: ", len(list(GA_PINN_model.parameters())))
print("Model's state_dict:")
for param_tensor in GA_PINN_model.state_dict():
	print(param_tensor, "\t", GA_PINN_model.state_dict()[param_tensor].size())

## Define learning rate
lr_w = config["training_process"]["parameters"]["learning_rate"]
## Define number of epochs
epochs = config["training_process"]["parameters"]["epochs"]

## Define the optimizer
if config["training_process"]["parameters"]["optimizer"] == "RAdam":
	optimizer = torch.optim.RAdam( GA_PINN_model.parameters(), lr = lr_w )
elif config["training_process"]["parameters"]["optimizer"] == "AdamW":
	optimizer = torch.optim.AdamW( GA_PINN_model.parameters(), lr = lr_w )
elif config["training_process"]["parameters"]["optimizer"] == "Adam":
	optimizer = torch.optim.Adam( GA_PINN_model.parameters(), lr = lr_w )
elif config["training_process"]["parameters"]["optimizer"] == "SGD":
	optimizer = torch.optim.SGD( GA_PINN_model.parameters(), lr = lr_w, nesterov = True, momentum = 0.9, dampening = 0 )
else:
	sys.exit("Optimizer is invalid. Only the following are available at this moment:\n-'RAdam'\n-'AdamW'\n-'Adam'\n-'SGD'")

print("Optimizer's state_dict of the global NN:")
for var_name in optimizer.state_dict():
	print(var_name, "\t", optimizer.state_dict()[var_name])

## Training bucle
print("Starting optimization...")
### Define 'tqdm' bar in order to display the progress
pbar = tqdm(range(epochs))
for epoch in pbar:
	loss_total, loss_r, loss_ic = GA_PINN_model.train_step( optimizer = optimizer, epoch = epoch )
	loss_total, loss_r, loss_ic = loss_total.detach().cpu().numpy(), loss_r.detach().cpu().numpy(), loss_ic.detach().cpu().numpy()
	GA_PINN_model.loss_total_hist.append( float(loss_total) )
	GA_PINN_model.loss_r_hist.append( float(loss_r) )
	GA_PINN_model.loss_ic_hist.append( float(loss_ic) )
	# Compute relative l2 errors
	l2_total, l2_rho, l2_u, l2_p = GA_PINN_model.compute_l2()
	l2_total, l2_rho, l2_u, l2_p = float(l2_total.detach().cpu().numpy()), float(l2_rho.detach().cpu().numpy()), float(l2_u.detach().cpu().numpy()), float(l2_p.detach().cpu().numpy())
	GA_PINN_model.l2_total_hist.append(l2_total )
	GA_PINN_model.l2_rho_hist.append( l2_rho )
	GA_PINN_model.l2_u_hist.append( l2_u )
	GA_PINN_model.l2_p_hist.append( l2_p )
	# Save 'slopes'
	## Density
	try:
		GA_PINN_model.a_rho_hist.append( float(GA_PINN_model.activation_output_rho.parameter.detach().cpu().numpy()) )
	except:
		GA_PINN_model.a_rho_hist.append( GA_PINN_model.n_rho * GA_PINN_model.slope_rho )
	## Velocity
	try:
		GA_PINN_model.a_u_hist.append( float(GA_PINN_model.activation_output_u.parameter.detach().cpu().numpy()) )
	except:
		GA_PINN_model.a_u_hist.append( GA_PINN_model.n_u * GA_PINN_model.slope_u )
	## Pressure
	try:
		GA_PINN_model.a_p_hist.append( float(GA_PINN_model.activation_output_p.parameter.detach().cpu().numpy()) )
	except:
		GA_PINN_model.a_p_hist.append( GA_PINN_model.n_p * GA_PINN_model.slope_p )
	## Hidden layers
	for i in range( len(GA_PINN_model.dense_layers) - 1 ):
		try:
			GA_PINN_model.as_hidden_hist[i].append( float(GA_PINN_model.activations_hidden[i].parameter.detach().cpu().numpy()) )
		except:
			GA_PINN_model.as_hidden_hist[i].append( GA_PINN_model.n_hidden * GA_PINN_model.slopes_hidden )
	# Show info on the bar
	pbar.set_postfix({'l2_total' : l2_total, 'l2_rho' : l2_rho, 'l2_u' : l2_u, 'l2_p' : l2_p})
	# Export data
	if epoch % config["training_process"]["export"]["save_each"] == 0:
		utils.save_results( model = GA_PINN_model, path = config["training_process"]["export"]["path_data"] )
		torch.save( GA_PINN_model.state_dict(), config["training_process"]["export"]["path_models"] + "model_" + str(epoch) + ".pt" )
		utils.plot_results( model = GA_PINN_model, path = config["training_process"]["export"]["path_images"] )



































