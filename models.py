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
from custom_activations import *

# Define the model
class GA_PINN(nn.Module):
	def __init__(self, X_r, X_0, U_0, ground_truth, physical_space, config, device, DTYPE):
		super(GA_PINN, self).__init__()
		# Define config, DTYPE and device
		self.config, self.DTYPE, self.device = config, DTYPE, device
		self.number_neurons = self.config["neural"]["general_parameters"]["number_neurons"]
		self.number_hidden = self.config["neural"]["general_parameters"]["number_hidden"]
		# Physical parameters
		self.X_r, self.X_0, self.U_0 = X_r, X_0, U_0
		self.ground_truth, self.physical_space = ground_truth, physical_space
		self.tmin, self.tmax = self.config["physical"]["parameters"]["temporal_range"][0], self.config["physical"]["parameters"]["temporal_range"][1]
		self.xmin, self.xmax = self.config["physical"]["parameters"]["spatial_range"][0], self.config["physical"]["parameters"]["spatial_range"][1]
		self.gamma = eval( self.config["physical"]["parameters"]["adiabatic_constant"] )
		self.N_r, self.N_0 = int(self.X_r.shape[0]), int(self.X_0.shape[0])
		
		# DNN
		## Define the layers
		self.dense_layers = []
		self.dense_layers.append(nn.Linear(2, self.number_neurons).to(self.device))
		for i in range(0, self.number_hidden):
			layer = nn.Linear(self.number_neurons, self.number_neurons).to(self.device)
			self.dense_layers.append(layer)
		layer_final = nn.Linear(self.number_neurons, 3).to(self.device)
		self.dense_layers.append(layer_final)
		## Initialize if considered
		if self.config["neural"]["general_parameters"]["init"] == "xavier_uniform_":
			for layer in self.dense_layers:
				torch.nn.init.xavier_uniform_( layer_final.weight, gain = 1.0 )
		else:
			sys.exit("Invalid method of 'init' for weights. Supported:\n-'xavier_uniform_'")
		## Register parameters hidden layers
		self.params_hidden = nn.ModuleList(self.dense_layers)
		# Add extra variables:
		## Related with the activation function
		### Hyperparameters "n"
		self.n_rho, self.n_u, self.n_p = self.config["neural"]["activation_functions"]["n"]["output"]
		self.n_hidden = self.config["neural"]["activation_functions"]["n"]["hidden"]
		### Initial slopes
		self.slope_rho, self.slope_u, self.slope_p = self.config["neural"]["activation_functions"]["initial_slopes"]["output"]
		self.slopes_hidden = self.config["neural"]["activation_functions"]["initial_slopes"]["hidden"]
		# Define initial 'a' set of trainable parameters
		self.a_output_rho = nn.Parameter( torch.tensor(self.slope_rho, dtype = self.DTYPE).to(self.device) )
		self.a_output_u = nn.Parameter( torch.tensor(self.slope_u, dtype = self.DTYPE).to(self.device) )
		self.a_output_p = nn.Parameter( torch.tensor(self.slope_p, dtype = self.DTYPE).to(self.device) )
		self.as_hidden = []
		for i in range(len(self.dense_layers) - 1):
			self.as_hidden.append( nn.Parameter( torch.tensor(self.slopes_hidden, dtype = self.DTYPE).to(self.device) ) )
		# Register parameters for independent activations of the hidden layers
		self.params_hidden_acts = nn.ParameterList(self.as_hidden)
		# Define lists to save 'slopes'
		self.a_rho_hist, self.a_u_hist, self.a_p_hist, self.as_hidden_hist = [], [], [], []
		self.as_hidden_hist = []
		for i in range( len(self.dense_layers) - 1 ):
			self.as_hidden_hist.append( [] )
		# Define lists to save relative l2 errors
		self.l2_total_hist, self.l2_rho_hist, self.l2_u_hist, self.l2_p_hist = [], [], [], []
		# Define lists to save physical losses
		self.loss_total_hist, self.loss_r_hist, self.loss_ic_hist = [], [], []
	def forward(self, x):
		# Initialize activation functions for output
		try:
			self.activation_output_rho = eval( self.config["neural"]["activation_functions"]["output"][0] )
			self.activation_output_u = eval( self.config["neural"]["activation_functions"]["output"][1] )
			self.activation_output_p = eval( self.config["neural"]["activation_functions"]["output"][2] )
		except:
			sys.exit("Bad syntaxis for the activation functions for the output. Supported trainable functions and vanilla ones from 'torch' module. Example:\n-Trainable function: 'TrainableSig(parameter = self.a_output_rho, n = self.n_rho)', considering the corresponding variable.\n-Non-trainable function (vanilla): 'nn.Sigmoid()', 'nn.Tanh()', etc.")
		self.activations_hidden = []
		for i in range(len(self.dense_layers) - 1):
			# Initialize activation functions for hidden layers
			try:
				self.activations_hidden.append( eval( self.config["neural"]["activation_functions"]["hidden_layers"] ) )
			except:
				sys.exit("Bad syntaxis for the activation functions for the hidden layers. Supported trainable functions and vanilla ones from 'torch' module. Example:\n-Trainable function: 'TrainableTanh(parameter = self.a_output_hidden, n = self.n_hidden)', considering the corresponding variable.\n-Non-trainable function (vanilla): 'nn.Tanh()', 'nn.ELU()', etc.")
			x = self.activations_hidden[i]( self.dense_layers[i](x) )
		x = self.dense_layers[len(self.dense_layers) - 1](x)
		rho, u, p = self.activation_output_rho(x[:,0:1]), self.activation_output_u(x[:,1:2]), self.activation_output_p(x[:,2:3])
		return torch.cat( (rho, u, p), dim = 1 )
	def compute_l2(self):
		self.pred = self( self.physical_space )
		l2_rho = torch.sqrt( ( ( self.ground_truth[:,0:1] - self.pred[:,0:1] )**2 ).sum() / (self.ground_truth[:,0:1]**2).sum() )
		l2_u = torch.sqrt( ( ( self.ground_truth[:,1:2] - self.pred[:,1:2] )**2 ).sum() / (self.ground_truth[:,1:2]**2).sum() )
		l2_p = torch.sqrt( ( ( self.ground_truth[:,2:3] - self.pred[:,2:3] )**2 ).sum() / (self.ground_truth[:,2:3]**2).sum() )
		return l2_rho + l2_u + l2_p, l2_rho, l2_u, l2_p
	def residual_ic(self):
		t_0, x_0 = self.X_0[:,0:1], self.X_0[:,1:2]
		self.prediction_0 = self( torch.cat((t_0,x_0), dim = 1) )
		
		self.w_IC = self.config["neural"]["loss_function_parameters"]["w_IC"]
		self.r_ic_1 = self.w_IC[0] * torch.square( self.prediction_0[:,0:1] - self.U_0[:,0:1] )
		self.r_ic_2 = self.w_IC[1] * torch.square( self.prediction_0[:,1:2] - self.U_0[:,1:2] )
		self.r_ic_3 = self.w_IC[2] * torch.square( self.prediction_0[:,2:3] - self.U_0[:,2:3] )
		
		self.r_ic = self.r_ic_1 + self.r_ic_2 + self.r_ic_3
		return self.r_ic
	def residual_r(self):
		t, x = self.X_r[:,0:1], self.X_r[:,1:2]
		prediction_r = self( torch.cat((t,x), dim = 1) )
		rho, u, p = prediction_r[:,0:1], prediction_r[:,1:2], prediction_r[:,2:3]
		W = 1 / torch.sqrt(1 - u**2)
		dens = rho * W
		s = (rho + p * self.gamma / (self.gamma - 1.0)) * W * W * u
		tau = (rho + p * self.gamma / (self.gamma - 1.0)) * (W**2) - p - dens
		
		j_1 = dens * u
		j_2 = s * u + p
		j_3 = s - dens * u
		
		rho_x = torch.autograd.grad(rho, x, grad_outputs = torch.ones_like(rho), create_graph = True)[0]
		u_x = torch.autograd.grad(u, x, grad_outputs = torch.ones_like(u), create_graph = True)[0]
		p_x = torch.autograd.grad(p, x, grad_outputs = torch.ones_like(p), create_graph = True)[0]
		dens_t = torch.autograd.grad(dens, t, grad_outputs = torch.ones_like(dens), create_graph = True)[0]
		s_t = torch.autograd.grad(s, t, grad_outputs = torch.ones_like(s), create_graph = True)[0]
		tau_t = torch.autograd.grad(tau, t, grad_outputs = torch.ones_like(tau), create_graph = True)[0]
		j_1_x = torch.autograd.grad(j_1, x, grad_outputs = torch.ones_like(j_1), create_graph = True)[0]
		j_2_x = torch.autograd.grad(j_2, x, grad_outputs = torch.ones_like(j_2), create_graph = True)[0]
		j_3_x = torch.autograd.grad(j_3, x, grad_outputs = torch.ones_like(j_3), create_graph = True)[0]
		
		self.rho_x, self.u_x, self.p_x = torch.abs(rho_x), torch.abs(u_x), torch.abs(p_x)
		
		r_1, r_2, r_3 = torch.square(dens_t + j_1_x), torch.square(s_t + j_2_x), torch.square(tau_t + j_3_x)
		alpha_set = self.config["neural"]["loss_function_parameters"]["alpha_set"]
		beta_set = self.config["neural"]["loss_function_parameters"]["beta_set"]
		w_R = self.config["neural"]["loss_function_parameters"]["w_R"]
		if self.config["neural"]["loss_function_parameters"]["lambda_to_use"] == "lambda_1":
			self.lambda_r_1 = torch.abs(1 / ( 1 + alpha_set[0] * self.rho_x ** beta_set[0] ) )
			self.lambda_r_2 = torch.abs(1 / ( 1 + alpha_set[1] * self.u_x ** beta_set[1] ) )
			self.lambda_r_3 = torch.abs(1 / ( 1 + alpha_set[2] * self.p_x ** beta_set[2] ) )
			self.factor_r = w_R * ( self.lambda_r_1 + self.lambda_r_2 + self.lambda_r_3 )
		elif self.config["neural"]["loss_function_parameters"]["lambda_to_use"] == "lambda_2":
			self.lambda_r_1 = self.rho_x
			self.lambda_r_2 = self.u_x
			self.lambda_r_3 = self.p_x
			self.factor_r = w_R * 1/( 1 + alpha_set[0] * self.lambda_r_1 ** beta_set[0] + alpha_set[1] * self.lambda_r_2 ** beta_set[1] + alpha_set[2] * self.lambda_r_3 ** beta_set[2] )
		else:
			sys.exit("Invalid 'lambda_to_use' function. Supported:\n'lambda_1'\n'lambda_2'")
		self.r = self.factor_r * (r_1 + r_2 + r_3)
		return self.r
	def train_step(self, optimizer, epoch):
		self.epoch = epoch
		# Optimizer: global
		loss_r, loss_ic = self.residual_r().mean(), self.residual_ic().mean()
		( loss_r + loss_ic ).backward( retain_graph = True )
		optimizer.step()
		optimizer.zero_grad(set_to_none = True)
		
		return loss_r + loss_ic, loss_r, loss_ic
























