# Import modules
import os
import sys
import numpy as np
from tqdm import tqdm
import h5py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import matplotlib.pyplot as plt
import pdb
from functools import reduce
sys.path.insert(1, "../../")
import utils



class GA_PINN(nn.Module):
	def __init__(self, config):
		"""Class for the model.
		
		Parameters
		----------
		config : dictionary
			Configuration file for the training.
		"""
		
		super(GA_PINN, self).__init__()
		self.config = config
		
		self.tmin, self.tmax = self.config["physical"]["parameters"]["temporal_range"]
		self.xmin, self.xmax = self.config["physical"]["parameters"]["spatial_range"]
		self.N_t, self.N_x = eval(self.config["physical"]["parameters"]["N_t"]), eval(self.config["physical"]["parameters"]["N_x"])
		self.size_hidden = self.config["neural"]["general_parameters"]["number_neurons"]
		self.num_hidden_layers = self.config["neural"]["general_parameters"]["number_hidden"]
		self.gamma = eval(self.config["physical"]["parameters"]["adiabatic_constant"])
		self.DTYPE, self.device = eval(self.config["training_process"]["DTYPE"]), torch.device(self.config["training_process"]["device"])
		self.num_inputs, self.num_outputs = 2, 3
		
		
		# Define the DNN.
		self.dense_layers = []
		self.dense_layers.append(nn.Linear(self.num_inputs, self.size_hidden).to(self.device))
		for i in range(0, self.num_hidden_layers):
			layer = nn.Linear(self.size_hidden, self.size_hidden).to(self.device)
			self.dense_layers.append(layer)
		layer_final = nn.Linear(self.size_hidden, self.num_outputs).to(self.device)
		self.dense_layers.append(layer_final)
		## Initialize the weights of the layers.
		for i in range(len(self.dense_layers)):
			torch.nn.init.xavier_uniform_(self.dense_layers[i].weight, gain = 1.0)
		## Now, register the parameters as trainable variables
		self.params_hidden = nn.ModuleList(self.dense_layers)
		
		# Related with the L2 computation.
		self.l2_hist, self.l2_rho_hist, self.l2_ux_hist, self.l2_p_hist = [], [], [], []
		# Define lists for the physical losses.
		self.loss_hist, self.loss_ic_hist = [], []
		self.loss_ic_rho, self.loss_ic_ux, self.loss_ic_p = [], [], []
		# Define activation functions.
		self.act_rho = eval(self.config["neural"]["activation_functions"]["output"][0])
		self.act_ux = eval(self.config["neural"]["activation_functions"]["output"][1])
		self.act_p = eval(self.config["neural"]["activation_functions"]["output"][2])
		self.act_hidden = eval(self.config["neural"]["activation_functions"]["hidden_layers"])
	def compute_l2(self):
		with torch.no_grad():
			prediction = self(self.analytical_space)
		rho_pred, ux_pred, p_pred = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3]
		rho_truth, ux_truth, p_truth = self.analytical_solution[:,0:1], self.analytical_solution[:,1:2], self.analytical_solution[:,2:3]
		
		l2_rho = torch.sqrt(torch.square(rho_truth - rho_pred).sum()/torch.square(rho_truth).sum()).item()
		l2_ux = torch.sqrt(torch.square(ux_truth - ux_pred).sum()/torch.square(ux_truth).sum()).item()
		l2_p = torch.sqrt(torch.square(p_truth - p_pred).sum()/torch.square(p_truth).sum()).item()
		self.l2_rho_hist.append(l2_rho)
		self.l2_ux_hist.append(l2_ux)
		self.l2_p_hist.append(l2_p)
		self.l2_hist.append(l2_rho + l2_ux + l2_p)
	def forward(self, X):
		# Training bucle.
		for i in range( len(self.dense_layers) - 1 ):
			X = self.act_hidden(self.dense_layers[i](X))
		X = self.dense_layers[-1](X)
		
		# Extract each primitive variable separately.
		rho = self.act_rho(X[:,0:1])
		ux = self.act_ux(X[:,1:2])
		p = self.act_p(X[:,2:3])
		return torch.cat((rho, ux, p), dim = 1)
	def compute_loss(self, X, X_0, U_0):
		t, x = X[:,0:1], X[:,1:2]
		prediction = self(torch.cat((t, x), dim = 1 ))
		rho, ux, p = prediction[:,0:1], prediction[:,1:2], prediction[:,2:3]
		ux = torch.clamp(ux, max = 0.9999, min = -0.9999)
		e = p/(rho * (self.gamma - 1.0) )
		
		W = 1 / torch.sqrt(1 - ux**2)
		
		D = rho * W
		Mx = ux * (rho + p * self.gamma / (self.gamma - 1.0) ) * (W ** 2)
		E = (rho + p * self.gamma/(self.gamma - 1.0)) * (W ** 2) - p
		
		F1 = D * ux
		F2x = Mx * ux + p
		F3 = (E + p) * ux
		
		D_t = torch.autograd.grad(D, t, grad_outputs = torch.ones_like(D), create_graph = True)[0]
		Mx_t = torch.autograd.grad(Mx, t, grad_outputs = torch.ones_like(Mx), create_graph = True)[0]
		E_t = torch.autograd.grad(E, t, grad_outputs = torch.ones_like(E), create_graph = True)[0]
		
		F1_x = torch.autograd.grad(F1, x, grad_outputs = torch.ones_like(F1), create_graph = True)[0]
		F2x_x = torch.autograd.grad(F2x, x, grad_outputs = torch.ones_like(F2x), create_graph = True)[0]
		F3_x = torch.autograd.grad(F3, x, grad_outputs = torch.ones_like(F3), create_graph = True)[0]

		rho_x = torch.autograd.grad(rho, x, grad_outputs = torch.ones_like(rho), create_graph = True)[0]
		ux_x = torch.autograd.grad(ux, x, grad_outputs = torch.ones_like(ux), create_graph = True)[0]
		p_x = torch.autograd.grad(p, x, grad_outputs = torch.ones_like(p), create_graph = True)[0]
		
		
		self.alpha_rho, self.alpha_ux, self.alpha_p = self.config["neural"]["loss_function_parameters"]["alpha_set"]
		self.beta_rho, self.beta_ux, self.beta_p = self.config["neural"]["loss_function_parameters"]["beta_set"]
		Lambda = ( 1 / (1 + ( self.alpha_rho * torch.abs(rho_x)**self.beta_rho + self.alpha_ux * torch.abs(ux_x)**self.beta_ux + self.alpha_p * torch.abs(p_x)**self.beta_p) ) ).view(self.N_t, self.N_x, 1) # Works fine
		self.Lambda = Lambda
		# Compute Losses
		# ================================================================================================================================
		## Losses of the equations conforming the system
		### These present shape of (N_t, N_x, 1)
		L_t_1 = torch.square(D_t + F1_x).view(self.N_t, self.N_x, 1)
		L_t_2 = torch.square(Mx_t + F2x_x).view(self.N_t, self.N_x, 1)
		L_t_3 = torch.square(E_t + F3_x).view(self.N_t, self.N_x, 1)
		
		## Total physical loss
		L_t = torch.mean( Lambda * (L_t_1 + L_t_2 + L_t_3), dim = 1 )
		# ================================================================================================================================
		
		## Compute loss for tmin (L_IC).
		prediction_tmin = self(X_0)
		# Consider a certain weight for the IC (hyperparameter) and for the collocation loss.
		w_rho, w_ux, w_p = self.config["neural"]["loss_function_parameters"]["w_IC"]
		w_R = self.config["neural"]["loss_function_parameters"]["w_R"]
		# Compute initial losses.
		L_IC_rho = w_rho * torch.square(U_0[:,0:1] - prediction_tmin[:,0:1]).mean()
		L_IC_ux = w_ux * torch.square(U_0[:,1:2] - prediction_tmin[:,1:2]).mean()
		L_IC_p = w_p * torch.square(U_0[:,2:3] - prediction_tmin[:,2:3]).mean()
		L_IC = L_IC_rho + L_IC_ux + L_IC_p
		L_t = torch.cat( (L_IC.view(1,1), L_t[1:]), dim = 0 )
		### Take advantage and save initial losses into lists
		self.loss_ic_hist.append(L_IC.item())
		self.loss_ic_rho.append(torch.square(U_0[:,0:1] - prediction_tmin[:,0:1]).mean().item())
		self.loss_ic_ux.append(torch.square(U_0[:,1:2] - prediction_tmin[:,1:2]).mean().item())
		self.loss_ic_p.append(torch.square(U_0[:,2:3] - prediction_tmin[:,2:3]).mean().item())
		## Compute and save l2 errors
		self.compute_l2()
		return L_t.mean()
	def train_step(self, X, X_0, U_0, optimizer):
		optimizer.zero_grad(set_to_none = True)
		# Compute the loss
		loss = self.compute_loss(X, X_0, U_0)
		loss.backward(retain_graph = False)
		# Apply gradient clipping and update parameters
		#torch.nn.utils.clip_grad_value_(self.parameters(), 1.0)
		#torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
		optimizer.step()
		# Save data
		self.loss_hist.append(loss.item())
		return loss



































