# Gradient-Annihilated PINNs (GA-PINNs) for Solving Riemann Problems: Application to Relativistic Hydrodynamics
### Authors: Antonio Ferrer-Sánchez[^1][^2], José D. Martín-Guerrero[^1][^2], Roberto Ruiz de Austri[^3], Alejandro Torres-Forné[^4][^5], José A. Font[^4][^5] ###

[^1]: IDAL, Electronic Engineering Department, ETSE-UV, University of Valencia, Avgda. Universitat s/n, 46100 Burjassot, Valencia, Spain.
[^2]: Valencian Graduate School and Research Network of Artificial Intelligence (ValgrAI), Spain.
[^3]: Instituto de Física Corpuscular CSIC-UV, c/Catedrático José Beltrán 2, 46980 Paterna, Valencia, Spain.
[^4]: Departamento de Astronomía y Astrofísica, Universitat de Valencia, Dr. Moliner 50, 46100, Burjassot (València), Spain.
[^5]: Observatori Astronòmic, Universitat de València, Catedrático José Beltrán 2, 46980, Paterna (València), Spain.

---------------------------------------------------------------------------------------------------------------------------------

The main purpose of this repository is to provide reproducible code for the GA-PINN methodology described in the paper [Gradient-Annihilated PINNs for Solving Riemann Problems: Application to Relativistic Hydrodynamics](https://www.sciencedirect.com/science/article/pii/S0045782524001622). The folders and scripts are distributed as follows:

- **Ground_Data/**: Folder containing the corresponding *ground truth* solutions used through the numerical examples of the paper. This folder contains one subfolder for the Problem 3 that has been studied. Inside it, one can find two *.h5* files corresponding to the analytical and the HRSC (numerical) resolutions for the physical variables at final time. Both, the analytical and the numerical files, contain the following keys:
    - *dens_calculated*, *dens_initial*: Arrays of the density ($\rho$) at final time ($tmax$) and at initial time, respectively.
    - *ur_calculated*, *ur_initial*: Arrays of the velocity ($u$) at final time ($tmax$) and at initial time, respectively.
    - *p_calculated*, *p_initial*: Arrays of the pressure ($p$) at final time ($tmax$) and at initial time, respectively.
    - *w_calculated*, *w_initial*: Arrays of the Lorentz factor ($W$) at final time ($tmax$) and at initial time, respectively.
    - *x_space*: Array of the spatial domain, going from $xmin$ to $xmax$ depending on the problem under consideration.
- **Example_Problem/**: This folder contains example Python code using PyTorch to train a GA-PINN, corresponding to the *training_script.py* file. By default, the images and exported data are saved in the *Images/* and *Models_Data/* subfolders respectively, while the *.pt* files corresponding to the weights of the neural model are stored inside the *Models_Data/Model_Saved/* folder.
- **custom_activations.py**: Script containing the custom activation functions that may be needed in the training procedures. These functions consider the slope parameter that could be trainable, as explained in the paper. More can be defined if needed, but as it is right now the script contains functions such as the hyperbolic tangent, the sigmoid, Heaviside, and softplus functions.
- **models.py**: This script contains all the necessary code to define the *GA_PINN* class as reading from the *torch.nn.Module*. This class is imported into the aforementioned *training_script.py* of the example. As input, it admits the configuration file which is imported as a JSON within the training script.
- **utils.py**: Script containing some additional functions such as a function to save the results as *.h5* files, a function to plot them, and also functions to generate the physical domain and the initial conditions.
- **config.json**: This file corresponds to the configuration JSON of the training and the model. It contains several fields that can be modified in order to train the GA-PINN.
    - *physical*:
        - *parameters*:
            - *adiabatic_constant*: $\Gamma$ factor of the equations of the hydrodynamics.
            - *temporal_range*: List containing the *[tmin,tmax]* values.
            - *spatial_range*: List containing the *[xmin,xmax]* values.
            - *N_t* (string): Amount of points sampled for the temporal dimension ($t$). When using Sobol sampling method it must be a power of two.
            - *N_x* (string): Amount of points sampled for the spatial dimension ($x$). When using Sobol sampling method it must be a power of two.
            - *N_0* (string): Number containing the amount of points sampled for the initial physical domain. For *X_0* we do not Sobol since it can be understood as a linear grid (time is fixed to *tmin*), but we consider also a power of two just for simplicity.
            - *sampling_method*: Only available method at this moment is 'sobol'. More may be considered in the future.
        - *initial_conditions*: This field contains the initial conditions for each one of the variables. The first and second values of the lists correspond to the variables on the left and on the right sides of the initial discontinuity, respectively, at the initial time.
    - *neural*:
        - *general_parameters*:
            - *number_hidden*: Number of hidden layers of the GA-PINN model (it does not include the input and output layers).
            - *number_neurons*: Number of neurons per layer. At this moment it is the same value for all of the hidden layers.
            - *init*: Method to initialize the parameters of the network. At this point, only 'xavier_uniform_' with $gain=1.0$ is considered.
        - *activation_functions*:
            - *hidden_layers*: Activation function for the hidden layers. It must be a string that will be evaluated by the script so it needs to be written as a literal string indicating which activation function we want to use. For example, in case that we want to use a vanilla hyperbolic tangent in PyTorch, we should write 'nn.Tanh()'. In case we want to use a function with trainable slopes we would need to call the custom functions inside *custom_activations.py*, as the one that is already written as example.
            - *output*: In a similar way the same happens for the output variables, except that we have a list of 3 activation functions, corresponding to the density, velocity and pressure, respectively.
        - *loss_function_parameters*:
            - *w_R*: Weight of the residual (collocation) part of the loss, $\hat{\mathcal{L}}_{\mathcal{R}}$.
            - *w_IC*: List of the weights of the initial part of the loss, $\mathcal{L}_{\mathcal{IC}}$, corresponding to each one of the variables (density,velocity,pressure).
            - *alpha_set*: Set of hyperparameters corresponding to ($\alpha_{\rho}$, $\alpha_{u}$, $\alpha_{p}$), that is, the weight of the respective gradients in the GA-PINN methodology.
            - *beta_set*: Analogously, this defines the set of hyperparameters ($\beta_{\rho}$, $\beta_{u}$, $\beta_{p}$) corresponding to the exponents of the respective gradients.
    - *training_process*:
        - *device*: Defines the device that *PyTorch* is going to use (e.g. "cpu" or "cuda").
        - *DTYPE*: General DTYPE for the data. It is recommended to use *torch.float32*.
        - *import*: This key contains the *analytical_solution_path* parameter referring to the path pointing to the *.h5* file containing the analytical / ground data.
        - *export*: This field contains some configurations about the folders where we want the data and images to get stored into. In addition, the *save_each* parameter corresponds to the number of epochs how often we desire to save the information.
        - *parameters*: Here we can define the optimizer to use (Adam, AdamW, RAdam and SGD are available), as well as the learning rate, the number of epochs and the random seed.

---------------------------------------------------------------------------------------------------------------------------------

### Methodology ###

<img src=./figures/Methodology_Diagram.svg scale="100%"/>

The methodology involves a neural network processing temporal and spatial physical variables $(t,x)$. Its architecture, including layers, activation functions, and neurons, is flexible. The network outputs primitive variables $(\rho,u,p)$, differentiated automatically to compute physical constraints and residuals of the initial conditions. These results combine using weights $(\omega_{\mathcal{IC}},\omega_{\mathcal{R}})$, evaluating the loss function $\mathcal{L}$ to update parameters $\Theta$.

---------------------------------------------------------------------------------------------------------------------------------

### Some results ###

<img src=./figures/ICs_Problem_1.png scale="100%"/>

Physical variables for Sod Shock Tube initial conditions solved at $t=0.5$ with $\Gamma=5/3$. Analytical solution (solid black curve) serves as reference. GA-PINNs method in top row (**a**), base PINN model in bottom row (**b**). Predictions shown as blue circles. "SW", "CD", and "RW", indicate shock waves, contact discontinuities, and rarefactions, respectively, in the density plot in (**a**) for visual simplicity.

<img src=./figures/ICs_Problem_1_Losses.png scale="100%"/>

Graphic representation of loss metrics during training for the \textit{Sod Shock Tube} problem (Problem 1). (**a**) Total physical loss. (**b**) $l^{2}$ prediction errors in GA-PINN compared to the analytical solution, presented by independent variable, with the dashed green line indicating the error of the HRSC numerical method for reference. (**c**) Comparative illustration of the total $l^{2}$ error between the GA-PINN model and a baseline model.

---------------------------------------------------------------------------------------------------------------------------------

### Requirements ###

The entire code is written using PyTorch and fully tested with Python 3.8.10. Main used packages are:
* PyTorch 
* SciPy
* Matplotlib

The requirements can be installed using the available *requirements.txt* file. You may want to use:

`pip install -r requirements.txt`

Although some distributions as Anaconda may require to use *conda* instead of *pip* after activating a particular environment.

---------------------------------------------------------------------------------------------------------------------------------

### How to train the example problem ###

The *config.json* is already prepared with the initial conditions of the *Sod shock tube* problem. The rest of analytical solutions presented in the paper can be available upon a reasonable request.

In order to run the training procedure once the *config.json* has been prepared, is to execute the command:

`python3 training_script.py`

The script will take a GPU as device if it exists. The training process may take more or less time depending on the hardware used. For example, the entire procedure takes ~4h in a Quadro RTX 6000 for a total of 200,000 epochs.

### Contact ###

For any additional query please contact the corresponding author of the manuscript by email.

[Send email to Antonio Ferrer-Sánchez.](mailto:Antonio.Ferrer-Sanchez@uv.es)
