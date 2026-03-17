# Import modules
import json
import math
import os
import pdb
import sys
from functools import reduce

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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
        self.N_t, self.N_x = (
            eval(self.config["physical"]["parameters"]["N_t"]),
            eval(self.config["physical"]["parameters"]["N_x"]),
        )
        self.size_hidden = self.config["neural"]["general_parameters"]["number_neurons"]
        self.num_hidden_layers = self.config["neural"]["general_parameters"][
            "number_hidden"
        ]
        self.gamma = eval(self.config["physical"]["parameters"]["adiabatic_constant"])
        self.DTYPE, self.device = (
            eval(self.config["training_process"]["DTYPE"]),
            torch.device(self.config["training_process"]["device"]),
        )
        self.num_inputs, self.num_outputs = 2, 3

        # Define the DNN.
        self.dense_layers = []
        self.dense_layers.append(
            nn.Linear(self.num_inputs, self.size_hidden).to(self.device)
        )
        for i in range(0, self.num_hidden_layers):
            layer = nn.Linear(self.size_hidden, self.size_hidden).to(self.device)
            self.dense_layers.append(layer)
        layer_final = nn.Linear(self.size_hidden, self.num_outputs).to(self.device)
        self.dense_layers.append(layer_final)
        ## Initialize the weights of the layers.
        for i in range(len(self.dense_layers)):
            torch.nn.init.xavier_uniform_(self.dense_layers[i].weight, gain=1.0)
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
        self.act_hidden = eval(
            self.config["neural"]["activation_functions"]["hidden_layers"]
        )

    def forward(self, X):
        # Training bucle.
        for i in range(len(self.dense_layers) - 1):
            X = self.act_hidden(self.dense_layers[i](X))
        X = self.dense_layers[-1](X)

        # Extract each primitive variable separately.
        rho = self.act_rho(X[:, 0:1])
        ux = self.act_ux(X[:, 1:2])
        p = self.act_p(X[:, 2:3])
        return torch.cat((rho, ux, p), dim=1)
