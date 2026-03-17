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
    # Change to ADAM if needed:
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
    ρ_analytical = torch.tensor(np.array(hf.get("dens_calculated"))).view(-1, 1)
    ux_analytical = torch.tensor(np.array(hf.get("ur_calculated"))).view(-1, 1)
    p_analytical = torch.tensor(np.array(hf.get("p_calculated"))).view(-1, 1)
    hf.close()
    t_analytical = torch.tensor(tmax).repeat((x_analytical.shape[0], 1))

    analytical_space = torch.cat((t_analytical, x_analytical), dim=1)
    analytical_variables = torch.cat((ρ_analytical, ux_analytical, p_analytical), dim=1)

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

    ρL, ρR = config["physical"]["initial_conditions"]["density"]
    uxL, uxR = config["physical"]["initial_conditions"]["velocity"]
    pL, pR = config["physical"]["initial_conditions"]["pressure"]

    x_numpy = x.detach().cpu().numpy()
    ic_ρ = lambda x: (ρL) * (x <= 0.5) + (ρR) * (x > 0.5)
    ic_ux = lambda x: (uxL) * (x <= 0.5) + (uxR) * (x > 0.5)
    ic_p = lambda x: (pL) * (x <= 0.5) + (pR) * (x > 0.5)

    W_tensor = torch.tensor(
        1 / (1 - (ic_ux(x_numpy) ** 2)) ** (1 / 2), requires_grad=True
    )
    ρ_tensor = torch.tensor(ic_ρ(x_numpy), requires_grad=True)
    ux_tensor = torch.tensor(ic_ux(x_numpy), requires_grad=True)
    p_tensor = torch.tensor(ic_p(x_numpy), requires_grad=True)

    output_ICs = torch.cat((ρ_tensor, ux_tensor, p_tensor), dim=1)
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

    # ==== Generate data (internal) ====
    ## ==== Define list to save tensors ====
    X_list = []
    ## ==== Define main temporal domain ====
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

    # ==== Generate data (initial) ====
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

    # ==== Data for t=tmax ====
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
    ρ_final, ux_final, p_final = (
        prediction_tmax[:, 0:1],
        prediction_tmax[:, 1:2],
        prediction_tmax[:, 2:3],
    )
    X_final = X_final.detach().cpu().numpy()

    # ==== Plot of the final variables ====
    fig, ax = plt.subplots(1, 3, figsize=(9, 3.5), constrained_layout=True)
    ## ==== Final density plot ====
    ax[0].scatter(
        X_final[:, 1:2],
        ρ_final,
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
    ## ==== Final velocity plot ====
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
    ## ==== Final density plot ====
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
    plt.savefig(config["training_process"]["export"]["path_images"] + "results.png", format="png", dpi=300)
    plt.close()

    # ==== Plot of the losses and relative L2 ====
    fig, ax = plt.subplots(1, 2, figsize=(9, 3), constrained_layout=True)
    ax[0].semilogy(range(len(model.histories['ℒ_hist'])), model.histories['ℒ_hist'], "k-")
    ax[0].set_xlabel("epoch")
    ax[0].set_title(r"$\mathcal{L}$")
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)
    ax[1].semilogy(
        range(len(model.histories['l2_ρ_hist'])),
        model.histories['l2_ρ_hist'],
        "k-",
        label=r"$l_{\rho_{\theta}}^{2}$",
    )
    ax[1].semilogy(
        range(len(model.histories['l2_ux_hist'])),
        model.histories['l2_ux_hist'],
        "b-",
        label=r"$l_{u_{\theta}}^{2}$",
    )
    ax[1].semilogy(
        range(len(model.histories['l2_p_hist'])),
        model.histories['l2_p_hist'],
        "g-",
        label=r"$l_{p_{\theta}}^{2}$",
    )
    ax[1].legend()
    ax[1].set_xlabel("epoch")
    ax[1].set_title(r"$l^{2}$")
    for tick in ax[1].get_xticklabels():
        tick.set_rotation(45)
    plt.savefig(config["training_process"]["export"]["path_images"] + "losses.png", format="png", dpi=300)

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

    DTYPE  = eval(config["training_process"]["DTYPE"])
    device = torch.device(config["training_process"]["device"])

    path_data = config["training_process"]["export"]["path_data"]

    # Helper to convert whatever is stored to a NumPy array
    def to_numpy(x):
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        else:
            return np.array(x)

    # ==== Export training histories ====
    if hasattr(model, "histories") and isinstance(model.histories, dict):
        with h5py.File(path_data + "data_training.h5", "w") as hf_train:
            for key, value in model.histories.items():
                try:
                    data_np = to_numpy(value)
                    hf_train.create_dataset(key, data=data_np)
                except Exception as e:
                    print(f"[export_data] Not saving histories['{key}']: {e}")

    # ==== Export evaluation metrics ====
    if hasattr(model, "metrics") and isinstance(model.metrics, dict):
        with h5py.File(path_data + "data_eval.h5", "w") as hf_eval:
            for key, value in model.metrics.items():
                try:
                    data_np = to_numpy(value)
                    hf_eval.create_dataset(key, data=data_np)
                except Exception as e:
                    print(f"[export_data] Not saving metrics['{key}']: {e}")

    # ==== Export model weights at this point ====
    torch.save(model.state_dict(), config['training_process']['export']['path_models'] + 'model_weights.pt')


def compute_ℒ(model, X, X_0, U_0):
    """Function to compute the physical loss used for training."""

    # ==== Extract time and space, and predict ====
    N_t, N_x = model.N_t, model.N_x
    t, x = X[:, 0:1], X[:, 1:2]
    out = model(torch.cat((t, x), dim=1))
    ρ, ux, p = out[:, 0:1], out[:, 1:2], out[:, 2:3]

    W = 1 / torch.sqrt(1 - ux**2)

    # ==== Compute relativistic magnitudes ====
    D = ρ * W
    Mx = ux * (ρ + p * model.𝛾 / (model.𝛾 - 1.0)) * (W**2)
    E = (ρ + p * model.𝛾 / (model.𝛾 - 1.0)) * (W**2) - p

    # ==== Compute fluxes ====
    F1 = D * ux
    F2x = Mx * ux + p
    F3 = (E + p) * ux

    # ==== Compute gradients with autograd ====
    dD_dt = torch.autograd.grad(
        D, t, grad_outputs=torch.ones_like(D), create_graph=True
    )[0]
    dMx_dt = torch.autograd.grad(
        Mx, t, grad_outputs=torch.ones_like(Mx), create_graph=True
    )[0]
    dE_dt = torch.autograd.grad(
        E, t, grad_outputs=torch.ones_like(E), create_graph=True
    )[0]

    dF1_dx = torch.autograd.grad(
        F1, x, grad_outputs=torch.ones_like(F1), create_graph=True
    )[0]
    dF2x_dx = torch.autograd.grad(
        F2x, x, grad_outputs=torch.ones_like(F2x), create_graph=True
    )[0]
    dF3_dx = torch.autograd.grad(
        F3, x, grad_outputs=torch.ones_like(F3), create_graph=True
    )[0]

    dρ_dx = torch.autograd.grad(
        ρ, x, grad_outputs=torch.ones_like(ρ), create_graph=True
    )[0]
    dux_dx = torch.autograd.grad(
        ux, x, grad_outputs=torch.ones_like(ux), create_graph=True
    )[0]
    dp_dx = torch.autograd.grad(
        p, x, grad_outputs=torch.ones_like(p), create_graph=True
    )[0]

    model.α_ρ, model.α_ux, model.α_p = model.config["neural"]["loss_function_parameters"]["α_set"]
    model.β_ρ, model.β_ux, model.β_p = model.config["neural"]["loss_function_parameters"]["β_set"]
    Lambda = 1 / (
        1
        + (
            model.α_ρ * torch.abs(dρ_dx) ** model.β_ρ
            + model.α_ux * torch.abs(dux_dx) ** model.β_ux
            + model.α_p * torch.abs(dp_dx) ** model.β_p
        )
    ).view(N_t, N_x, 1)
    model.Lambda = Lambda
    # ==== Compute Losses ====
    # ================================================================================================================================
    ## ==== Losses of the equations conforming the system ====
    ### These present shape of (N_t, N_x, 1)
    ℒ_t_1 = (dD_dt + dF1_dx).pow(2).view(N_t, N_x, 1)
    ℒ_t_2 = (dMx_dt + dF2x_dx).pow(2).view(N_t, N_x, 1)
    ℒ_t_3 = (dE_dt + dF3_dx).pow(2).view(N_t, N_x, 1)

    ## ==== Total physical loss ====
    ℒ_t = torch.mean(Lambda * (ℒ_t_1 + ℒ_t_2 + ℒ_t_3), dim=1)
    # ================================================================================================================================

    ## ==== Compute loss for tmin (L_IC) ====
    prediction_tmin = model(X_0)
    # ==== Consider a certain weight for the IC (hyperparameter) and for the collocation loss ====
    w_ρ, w_ux, w_p = model.config["neural"]["loss_function_parameters"]["w_IC"]
    w_R = model.config["neural"]["loss_function_parameters"]["w_R"]
    # ==== Compute initial losses ====
    ℒ_IC_ρ = w_ρ * torch.square(U_0[:, 0:1] - prediction_tmin[:, 0:1]).mean()
    ℒ_IC_ux = w_ux * torch.square(U_0[:, 1:2] - prediction_tmin[:, 1:2]).mean()
    ℒ_IC_p = w_p * torch.square(U_0[:, 2:3] - prediction_tmin[:, 2:3]).mean()
    ℒ_IC = ℒ_IC_ρ + ℒ_IC_ux + ℒ_IC_p
    # ==== Compute total loss ====
    ## ==== 'ℒ_t' is a column vector (N_t,1), where the first element corresponds to the ICs loss ====
    ℒ_t = torch.cat((ℒ_IC.view(1, 1), w_R * ℒ_t[1:]), dim=0)
    # ==== If ε_t != 0, then causality is enforced; otherwise, put this parameter equal to zero in the config.json file ====
    ## ==== (OPTIONAL): This procedure was not implemented in the original paper; this is a direct improvement of the original methodology ====
    if model.ε_t != 0:
        zeros_t = torch.zeros(1, 1, device=X.device, dtype=ℒ_t.dtype)
        ℒ_t_shifted = torch.cat((zeros_t, ℒ_t[:-1]), dim=0)
        ℒ_t_cumsum = torch.cumsum(ℒ_t_shifted, dim=0)
        w_t = torch.exp(-model.ε_t * ℒ_t_cumsum)
        ℒ = (w_t * ℒ_t).mean()
        # ==== Compute 'AUC': Area under the weights curve. AUC is in [0,1], having AUC --> 1.0 when causality has been totally enforced. ====
        dx = 1.0 / float(X.shape[0]-1)
        AUC = torch.trapz(w_t.view(-1), dx=dx).item()
        model.histories['AUC_hist'].append(AUC)
    else:
        ℒ = ℒ_t.mean()


    ### ==== Log losses ====
    model.histories['ℒ_hist'].append(ℒ.item())
    model.histories['ℒ_ic_ρ'].append(
        torch.square(U_0[:, 0:1] - prediction_tmin[:, 0:1]).mean().item()
    )
    model.histories['ℒ_ic_ux'].append(
        torch.square(U_0[:, 1:2] - prediction_tmin[:, 1:2]).mean().item()
    )
    model.histories['ℒ_ic_p'].append(
        torch.square(U_0[:, 2:3] - prediction_tmin[:, 2:3]).mean().item()
    )
    # ==== Save IC losses without the weight scaling: raw MSE ====
    model.histories['ℒ_ic_hist'].append(
        model.histories['ℒ_ic_ρ'][-1] +
        model.histories['ℒ_ic_ux'][-1] +
        model.histories['ℒ_ic_p'][-1]
    )
    return ℒ


def compute_l2(model):
    """Function to compute the l2 w.r.t. analytical solution."""
    with torch.no_grad():
        prediction = model(model.analytical_space)
        ρ_pred, ux_pred, p_pred = (
            prediction[:, 0:1],
            prediction[:, 1:2],
            prediction[:, 2:3],
        )
        ρ_truth, ux_truth, p_truth = (
            model.analytical_solution[:, 0:1],
            model.analytical_solution[:, 1:2],
            model.analytical_solution[:, 2:3],
        )

        l2_ρ = torch.sqrt(
            torch.square(ρ_truth - ρ_pred).sum() / torch.square(ρ_truth).sum()
        ).item()
        l2_ux = torch.sqrt(
            torch.square(ux_truth - ux_pred).sum() / torch.square(ux_truth).sum()
        ).item()
        l2_p = torch.sqrt(
            torch.square(p_truth - p_pred).sum() / torch.square(p_truth).sum()
        ).item()
        model.histories['l2_ρ_hist'].append(l2_ρ)
        model.histories['l2_ux_hist'].append(l2_ux)
        model.histories['l2_p_hist'].append(l2_p)
        model.histories['l2_hist'].append(l2_ρ + l2_ux + l2_p)
