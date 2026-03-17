# Import modules
import json
import math
import os
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import qmc
from tqdm import tqdm

sys.path.insert(1, "../../")
import models
import pytorch_optimizer


def define_optimizer(model, config):
    """
    Function to define the optimizer.
    """

    base_lr = config["training_process"]["parameters"]["learning_rate"]

    optimizer = pytorch_optimizer.SOAP(model.parameters(), lr=base_lr)
    # Change for ADAM if needed:
    # optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    return optimizer


def load_analytical(config):
    """Load the physical analytical solution.

    Parameters
    ----------
    config : dictionary
            Configuration file for the training.

    Returns
    -------
    primitive_analytical, space_analytical : 2 torch.Tensor
            PyTorch tensors containing the analytical primitive variables and physical space.
    """

    tmin, tmax = config["physical"]["parameters"]["temporal_range"]
    hf = h5py.File(
        config["training_process"]["import"]["analytical_solution_path"], "r"
    )
    x_analytical = torch.tensor(np.array(hf.get("x_space"))).view(-1, 1)
    rho_analytical = torch.tensor(np.array(hf.get("dens_calculated"))).view(-1, 1)
    ux_analytical = torch.tensor(np.array(hf.get("ur_calculated"))).view(-1, 1)
    p_analytical = torch.tensor(np.array(hf.get("p_calculated"))).view(-1, 1)
    hf.close()
    t_analytical = torch.tensor(tmax).repeat((x_analytical.shape[0], 1))

    analytical_space = torch.cat((t_analytical, x_analytical), dim=1)
    analytical_variables = torch.cat(
        (rho_analytical, ux_analytical, p_analytical), dim=1
    )

    return analytical_space, analytical_variables


def initial_conditions(x, config):
    """Compute the initial conditions.

    Parameters
    ----------
    x : torch.Tensor
            Torch tensor containing the 'x' dimension for t=0, namely 'X_0[:,1:2]'.
    config : dictionary
            Configuration file for the training.

    Returns
    -------
    output_ICs : torch.Tensor
            Tensor containing the initial conditions for the primitive variables.
    """

    rhoL, rhoR = config["physical"]["initial_conditions"]["density"]
    uxL, uxR = config["physical"]["initial_conditions"]["velocity"]
    pL, pR = config["physical"]["initial_conditions"]["pressure"]

    x_numpy = x.detach().cpu().numpy()
    ic_rho = lambda x: (rhoL) * (x <= 0.5) + (rhoR) * (x > 0.5)
    ic_ux = lambda x: (uxL) * (x <= 0.5) + (uxR) * (x > 0.5)
    ic_p = lambda x: (pL) * (x <= 0.5) + (pR) * (x > 0.5)

    W_tensor = torch.tensor(
        1 / (1 - (ic_ux(x_numpy) ** 2)) ** (1 / 2), requires_grad=True
    )
    rho_tensor = torch.tensor(ic_rho(x_numpy), requires_grad=True)
    ux_tensor = torch.tensor(ic_ux(x_numpy), requires_grad=True)
    p_tensor = torch.tensor(ic_p(x_numpy), requires_grad=True)

    output_ICs = torch.cat((rho_tensor, ux_tensor, p_tensor), dim=1)
    return output_ICs


def generate_domain(config):
    """Compute the physical domain.

    Parameters
    ----------
    config : dictionary
            Configuration file for the training.

    Returns
    -------
    X_0, X_r : 2 torch.Tensor
            PyTorch tensors containing the physical space, for initial data and collocation points, respectively.
    """
    tmin, tmax = config["physical"]["parameters"]["temporal_range"]
    xmin, xmax = config["physical"]["parameters"]["spatial_range"]
    N_t, N_x = (
        eval(config["physical"]["parameters"]["N_t"]),
        eval(config["physical"]["parameters"]["N_x"]),
    )
    N_0 = eval(config["physical"]["parameters"]["N_0"])

    # Generate data (internal)
    ## Define list to save tensors
    X_list = []
    ## Define main temporal domain
    sampler = qmc.Sobol(d=1, scramble=False)
    sample = sampler.random_base2(m=int(np.log2(N_t)))
    l_bounds, u_bounds = [tmin], [tmax]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    t = torch.tensor(sample_scaled)

    for value in t:
        sampler = qmc.Sobol(d=1, scramble=False)
        sample = sampler.random_base2(m=int(np.log2(N_x)))
        l_bounds, u_bounds = [xmin], [xmax]
        sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
        x = torch.tensor(sample_scaled)
        t_repeated = torch.tensor(float(value.detach().cpu().numpy())).repeat(
            (x.shape[0], 1)
        )
        X_list.append(torch.cat((t_repeated, x), dim=1))
    X = torch.stack(X_list)
    X.requires_grad = True

    # Generate data (initial)
    t_0 = torch.tensor(tmin).repeat((N_0, 1)).view(-1, 1)
    sampler = qmc.Sobol(d=1, scramble=False)
    sample = sampler.random_base2(m=int(np.log2(N_0)))
    l_bounds, u_bounds = [xmin], [xmax]
    sample_scaled = qmc.scale(sample, l_bounds, u_bounds)
    x_0 = torch.tensor(sample_scaled)
    X_0 = torch.cat((t_0, x_0), dim=1)
    X_0.requires_grad = True

    return X, X_0


def plot_results(model, config):
    """Function to plot the results of the training process.

    Parameters
    ----------
    model : 'GA_PINN' object from 'nn.Module'.
            Model object.
    config : dictionary
            Configuration file for the training.
    """

    # Data for t=tmax.
    t_final = (
        torch.tensor(model.tmax, dtype=model.DTYPE, requires_grad=True)
        .repeat((30, 1))
        .to(model.device)
    )
    x_final = (
        torch.tensor(
            np.linspace(model.xmin, model.xmax, 30),
            dtype=model.DTYPE,
            requires_grad=True,
        )
        .view(-1, 1)
        .to(model.device)
    )
    X_final = torch.cat((t_final, x_final), dim=1)
    prediction_tmax = model(X_final).detach().cpu().numpy()
    rho_final, ux_final, p_final = (
        prediction_tmax[:, 0:1],
        prediction_tmax[:, 1:2],
        prediction_tmax[:, 2:3],
    )
    X_final = X_final.detach().cpu().numpy()

    # Plot of the final variables.
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.5), constrained_layout=True)
    ## Final density plot
    ax[0].scatter(
        X_final[:, 1:2],
        rho_final,
        color="blue",
        marker="o",
        facecolors="none",
        s=30,
        label=r"$\rho_{GA-PINN}$",
    )
    ax[0].plot(
        model.analytical_space[:, 1:2].detach().cpu().numpy(),
        model.analytical_solution[:, 0:1].detach().cpu().numpy(),
        "k-",
        label=r"$\rho_{analytical}$",
    )
    ax[0].set_xlabel(r"$x$")
    ax[0].set_title("density")
    ax[0].legend()
    ## Final velocity plot
    ax[1].scatter(
        X_final[:, 1:2],
        ux_final,
        color="blue",
        marker="o",
        facecolors="none",
        s=30,
        label=r"$ux_{GA-PINN}$",
    )
    ax[1].plot(
        model.analytical_space[:, 1:2].detach().cpu().numpy(),
        model.analytical_solution[:, 1:2].detach().cpu().numpy(),
        "k-",
        label=r"$ux_{analytical}$",
    )
    ax[1].set_xlabel(r"$x$")
    ax[1].set_title("velocity")
    ax[1].legend()
    ## Final density plot
    ax[2].scatter(
        X_final[:, 1:2],
        p_final,
        color="blue",
        marker="o",
        facecolors="none",
        s=30,
        label=r"$p_{GA-PINN}$",
    )
    ax[2].plot(
        model.analytical_space[:, 1:2].detach().cpu().numpy(),
        model.analytical_solution[:, 2:3].detach().cpu().numpy(),
        "k-",
        label=r"$p_{analytical}$",
    )
    ax[2].set_xlabel(r"$x$")
    ax[2].set_title("pressure")
    ax[2].legend()

    plt.suptitle("t=" + str(model.tmax) + ", " + str(model.epoch) + " epochs")
    plt.savefig(config["training_process"]["export"]["path_images"] + "results.png")
    plt.close()

    # Plot of the losses and relative L2.
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    ax[0].semilogy(range(len(model.loss_hist)), model.loss_hist, "k-")
    ax[0].set_xlabel("epoch")
    ax[0].set_title(r"$\mathcal{L}$")
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)
    ax[1].semilogy(
        range(len(model.l2_rho_hist)),
        model.l2_rho_hist,
        "k-",
        label=r"$l_{\rho_{\theta}}^{2}$",
    )
    ax[1].semilogy(
        range(len(model.l2_ux_hist)),
        model.l2_ux_hist,
        "b-",
        label=r"$l_{u_{\theta}}^{2}$",
    )
    ax[1].semilogy(
        range(len(model.l2_p_hist)),
        model.l2_p_hist,
        "g-",
        label=r"$l_{p_{\theta}}^{2}$",
    )
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_title(r"$l^{2}$")
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)
    plt.savefig(config["training_process"]["export"]["path_images"] + "losses.png")

    plt.close("all")


def save_results(model, config):
    """Function to plot the results of the training process.

    Parameters
    ----------
    model : 'GA_PINN' object from 'nn.Module'.
            Model object.
    config : dictionary
            Configuration file for the training.
    """

    hf = h5py.File(
        config["training_process"]["export"]["path_data"]
        + "data_"
        + str(model.epoch)
        + ".h5",
        "w",
    )

    hf.create_dataset("loss_hist", data=np.array(model.loss_hist))
    hf.create_dataset("loss_ic_hist", data=np.array(model.loss_ic_hist))

    hf.create_dataset("loss_ic_rho", data=np.array(model.loss_ic_rho))
    hf.create_dataset("loss_ic_ux", data=np.array(model.loss_ic_ux))
    hf.create_dataset("loss_ic_p", data=np.array(model.loss_ic_p))

    hf.create_dataset("l2_rho_hist", data=np.array(model.l2_rho_hist))
    hf.create_dataset("l2_ux_hist", data=np.array(model.l2_ux_hist))
    hf.create_dataset("l2_p_hist", data=np.array(model.l2_p_hist))
    hf.create_dataset("l2_hist", data=np.array(model.l2_hist))

    hf.close()


def compute_loss(model, X, X_0, U_0):
    """Function to compute the physical loss used for training."""
    t, x = X[:, 0:1], X[:, 1:2]
    prediction = model(torch.cat((t, x), dim=1))
    rho, ux, p = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
    ux = torch.clamp(ux, max=0.9999, min=-0.9999)
    e = p / (rho * (model.gamma - 1.0))

    W = 1 / torch.sqrt(1 - ux**2)

    D = rho * W
    Mx = ux * (rho + p * model.gamma / (model.gamma - 1.0)) * (W**2)
    E = (rho + p * model.gamma / (model.gamma - 1.0)) * (W**2) - p

    F1 = D * ux
    F2x = Mx * ux + p
    F3 = (E + p) * ux

    D_t = torch.autograd.grad(D, t, grad_outputs=torch.ones_like(D), create_graph=True)[
        0
    ]
    Mx_t = torch.autograd.grad(
        Mx, t, grad_outputs=torch.ones_like(Mx), create_graph=True
    )[0]
    E_t = torch.autograd.grad(E, t, grad_outputs=torch.ones_like(E), create_graph=True)[
        0
    ]

    F1_x = torch.autograd.grad(
        F1, x, grad_outputs=torch.ones_like(F1), create_graph=True
    )[0]
    F2x_x = torch.autograd.grad(
        F2x, x, grad_outputs=torch.ones_like(F2x), create_graph=True
    )[0]
    F3_x = torch.autograd.grad(
        F3, x, grad_outputs=torch.ones_like(F3), create_graph=True
    )[0]

    rho_x = torch.autograd.grad(
        rho, x, grad_outputs=torch.ones_like(rho), create_graph=True
    )[0]
    ux_x = torch.autograd.grad(
        ux, x, grad_outputs=torch.ones_like(ux), create_graph=True
    )[0]
    p_x = torch.autograd.grad(p, x, grad_outputs=torch.ones_like(p), create_graph=True)[
        0
    ]

    model.alpha_rho, model.alpha_ux, model.alpha_p = model.config["neural"][
        "loss_function_parameters"
    ]["alpha_set"]
    model.beta_rho, model.beta_ux, model.beta_p = model.config["neural"][
        "loss_function_parameters"
    ]["beta_set"]
    Lambda = 1 / (
        1
        + (
            model.alpha_rho * torch.abs(rho_x) ** model.beta_rho
            + model.alpha_ux * torch.abs(ux_x) ** model.beta_ux
            + model.alpha_p * torch.abs(p_x) ** model.beta_p
        )
    ).view(model.N_t, model.N_x, 1)
    model.Lambda = Lambda
    # Compute Losses
    # ================================================================================================================================
    ## Losses of the equations conforming the system
    ### These present shape of (N_t, N_x, 1)
    L_t_1 = torch.square(D_t + F1_x).view(model.N_t, model.N_x, 1)
    L_t_2 = torch.square(Mx_t + F2x_x).view(model.N_t, model.N_x, 1)
    L_t_3 = torch.square(E_t + F3_x).view(model.N_t, model.N_x, 1)

    ## Total physical loss
    L_t = torch.mean(Lambda * (L_t_1 + L_t_2 + L_t_3), dim=1)
    # ================================================================================================================================

    ## Compute loss for tmin (L_IC).
    prediction_tmin = model(X_0)
    # Consider a certain weight for the IC (hyperparameter) and for the collocation loss.
    w_rho, w_ux, w_p = model.config["neural"]["loss_function_parameters"]["w_IC"]
    w_R = model.config["neural"]["loss_function_parameters"]["w_R"]
    # Compute initial losses.
    L_IC_rho = w_rho * torch.square(U_0[:, 0:1] - prediction_tmin[:, 0:1]).mean()
    L_IC_ux = w_ux * torch.square(U_0[:, 1:2] - prediction_tmin[:, 1:2]).mean()
    L_IC_p = w_p * torch.square(U_0[:, 2:3] - prediction_tmin[:, 2:3]).mean()
    L_IC = L_IC_rho + L_IC_ux + L_IC_p
    L_t = torch.cat((L_IC.view(1, 1), L_t[1:]), dim=0)
    ### Take advantage and save initial losses into lists
    model.loss_ic_hist.append(L_IC.item())
    model.loss_ic_rho.append(
        torch.square(U_0[:, 0:1] - prediction_tmin[:, 0:1]).mean().item()
    )
    model.loss_ic_ux.append(
        torch.square(U_0[:, 1:2] - prediction_tmin[:, 1:2]).mean().item()
    )
    model.loss_ic_p.append(
        torch.square(U_0[:, 2:3] - prediction_tmin[:, 2:3]).mean().item()
    )
    return L_t.mean()


def compute_l2(model):
    """Function to compute the l2 w.r.t. analytical solution."""
    with torch.no_grad():
        prediction = model(model.analytical_space)
        rho_pred, ux_pred, p_pred = (
            prediction[:, 0:1],
            prediction[:, 1:2],
            prediction[:, 2:3],
        )
        rho_truth, ux_truth, p_truth = (
            model.analytical_solution[:, 0:1],
            model.analytical_solution[:, 1:2],
            model.analytical_solution[:, 2:3],
        )

        l2_rho = torch.sqrt(
            torch.square(rho_truth - rho_pred).sum() / torch.square(rho_truth).sum()
        ).item()
        l2_ux = torch.sqrt(
            torch.square(ux_truth - ux_pred).sum() / torch.square(ux_truth).sum()
        ).item()
        l2_p = torch.sqrt(
            torch.square(p_truth - p_pred).sum() / torch.square(p_truth).sum()
        ).item()
        model.l2_rho_hist.append(l2_rho)
        model.l2_ux_hist.append(l2_ux)
        model.l2_p_hist.append(l2_p)
        model.l2_hist.append(l2_rho + l2_ux + l2_p)
