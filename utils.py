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
import matplotlib.pyplot as plt

def save_results(model, path):
	hf = h5py.File(path + "data_" + str(model.epoch) + ".h5", "w")
	
	hf.create_dataset("loss_total_hist", data = np.array(model.loss_total_hist))
	hf.create_dataset("loss_r_hist", data = np.array(model.loss_r_hist))
	hf.create_dataset("loss_ic_hist", data = np.array(model.loss_ic_hist))
	
	hf.create_dataset("l2_total_hist", data = np.array(model.l2_total_hist))
	hf.create_dataset("l2_rho_hist", data = np.array(model.l2_rho_hist))
	hf.create_dataset("l2_u_hist", data = np.array(model.l2_u_hist))
	hf.create_dataset("l2_p_hist", data = np.array(model.l2_p_hist))
	
	hf.create_dataset("X_r", data = model.X_r.detach().cpu().numpy())
	hf.create_dataset("X_0", data = model.X_0.detach().cpu().numpy())
	hf.create_dataset("rho_x", data = model.rho_x.detach().cpu().numpy())
	hf.create_dataset("u_x", data = model.u_x.detach().cpu().numpy())
	hf.create_dataset("p_x", data = model.p_x.detach().cpu().numpy())
	
	hf.create_dataset("a_rho_hist", data = np.array(model.a_rho_hist))
	hf.create_dataset("a_u_hist", data = np.array(model.a_u_hist))
	hf.create_dataset("a_p_hist", data = np.array(model.a_p_hist))
	for i in range( len(model.as_hidden_hist) ):
		hf.create_dataset("as_hidden_hist_" + str(i), data = np.array(model.as_hidden_hist[i]))
	hf.close()

def plot_results(model, path):
	t = torch.tensor(model.tmax, dtype = model.DTYPE).repeat((50,1)).to(model.device)
	x = torch.reshape(torch.tensor(np.linspace(model.xmin, model.xmax, 50), dtype = model.DTYPE), (50,1)).to(model.device)
	
	prediction_tmax = model( torch.cat( (t, x), dim = 1 ) ).detach().cpu().numpy()
	rho_pred, u_pred, p_pred = prediction_tmax[:,0:1], prediction_tmax[:,1:2], prediction_tmax[:,2:3]
	t, x = t.detach().cpu().numpy(), x.detach().cpu().numpy()
	
	# FINAL VARIABLES PLOT
	fig, ax = plt.subplots(1, 3, figsize = (12, 4), constrained_layout = True)
	## Final density plot
	ax[0].scatter(x, rho_pred, color = "blue", marker = "o", facecolors = "none", s = 30, label = r'$\rho_{GA-PINN}$')
	ax[0].plot(model.physical_space[:,1:2].detach().cpu().numpy(), model.ground_truth[:,0:1].detach().cpu().numpy(), "k-", label = r'$\rho_{analytical}$')
	ax[0].set_title("Epoch = " + str(model.epoch) + ", t = " + str(model.tmax))
	ax[0].set_xlabel("x")
	ax[0].legend()
	## Final velocity plot
	ax[1].scatter(x, u_pred, color = "blue", marker = "o", facecolors = "none", s = 30, label = r'$u_{GA-PINN}$')
	ax[1].plot(model.physical_space[:,1:2].detach().cpu().numpy(), model.ground_truth[:,1:2].detach().cpu().numpy(), "b-", label = r'$u_{analytical}$')
	ax[1].set_title("Epoch = " + str(model.epoch) + ", t = " + str(model.tmax))
	ax[1].set_xlabel("x")
	ax[1].legend()
	## Final pressure plot
	ax[2].scatter(x, p_pred, color = "blue", marker = "o", facecolors = "none", s = 30, label = r'$p_{GA-PINN}$')
	ax[2].plot(model.physical_space[:,1:2].detach().cpu().numpy(), model.ground_truth[:,2:3].detach().cpu().numpy(), "r-", label = r'$p_{analytical}$')
	ax[2].set_title("Epoch = " + str(model.epoch) + ", t = " + str(model.tmax))
	ax[2].set_xlabel("x")
	ax[2].legend()
	
	plt.suptitle("Epoch = " + str(model.epoch))
	plt.savefig(path + "Results_Epoch_" + str(model.epoch) + ".png")
	plt.close()
	
	# PLOT OF THE LOSSES AND RELATIVE l2 ERRORS
	fig, ax = plt.subplots(1, 2, figsize = (9, 3), constrained_layout = True)
	## Plot of physical losses
	ax[0].semilogy( range(len(model.loss_total_hist)), model.loss_total_hist, "k-", label = r'$\mathcal{L}$' )
	ax[0].semilogy( range(len(model.loss_r_hist)), model.loss_r_hist, "b--", label = r'$\mathcal{L}_{\mathcal{R}}$' )
	ax[0].semilogy( range(len(model.loss_total_hist)), model.loss_total_hist, "r--", label = r'$\mathcal{L}_{\mathcal{IC}}$' )
	ax[0].legend()
	ax[0].set_xlabel("Epoch")
	ax[0].set_ylabel("Physical loss")
	## Plot of relative l2 errors
	ax[1].semilogy( range(len(model.l2_total_hist)), model.l2_total_hist, "k-", label = r'$l^{2}$' )
	ax[1].semilogy( range(len(model.l2_rho_hist)), model.l2_rho_hist, "k--", label = r'$l^{2}_{\rho}$' )
	ax[1].semilogy( range(len(model.l2_u_hist)), model.l2_u_hist, "b--", label = r'$l^{2}_{u}$' )
	ax[1].semilogy( range(len(model.l2_p_hist)), model.l2_p_hist, "r--", label = r'$l^{2}_{p}$' )
	ax[1].legend()
	ax[1].set_xlabel("Epoch")
	ax[1].set_ylabel("Relative error")
	plt.savefig(path + "Losses_and_L2.png")
	plt.close()
	# PLOT OF THE SLOPES (ACTIVATIONS)
	fig, ax = plt.subplots(1, 2, figsize = (11, 4), constrained_layout = True)
	## Slopes of output activations
	ax[0].plot( range(len(model.a_rho_hist)), model.a_rho_hist, "k-", label = r'$a_{\rho}$' )
	ax[0].plot( range(len(model.a_u_hist)), model.a_u_hist, "b-", label = r'$a_{u}$' )
	ax[0].plot( range(len(model.a_p_hist)), model.a_p_hist, "r-", label = r'$a_{p}$' )
	ax[0].legend()
	ax[0].set_xlabel("Epoch")
	ax[0].set_ylabel("Slopes parameter")
	## Slopes of hidden activations
	for i in range(len(model.dense_layers) - 1):
		str_i = str(i)
		ax[1].plot( range(len(model.as_hidden_hist[i])), model.as_hidden_hist[i], label = rf'$a_{{hidden}}^{{{str_i}}}$' )
	ax[1].legend(fontsize = 5)
	ax[1].set_xlabel("Epoch")
	plt.savefig(path + "Slopes_Activations.png")
	plt.close()


































