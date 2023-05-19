# Gradient-Annihilated PINNs for Solving Riemann Problems: Application to Relativistic Hydrodynamics
### Authors: Antonio Ferrer-Sánchez[^1][^2], José D. Martín-Guerrero[^1][^2], Roberto Ruiz de Austri[^3], Alejandro Torres-Forné[^4][^5], José A. Font[^4][^5] ###

[^1]: IDAL, Electronic Engineering Department, ETSE-UV, University of Valencia, Avgda. Universitat s/n, 46100 Burjassot, Valencia, Spain.
[^2]: Valencian Graduate School and Research Network of Artificial Intelligence (ValgrAI), Spain.
[^3]: Instituto de Física Corpuscular CSIC-UV, c/Catedrático José Beltrán 2, 46980 Paterna, Valencia, Spain.
[^4]: Departamento de Astronomía y Astrofísica, Universitat de Valencia, Dr. Moliner 50, 46100, Burjassot (València), Spain.
[^5]: Observatori Astronòmic, Universitat de València, Catedrático José Beltrán 2, 46980, Paterna (València), Spain.

---------------------------------------------------------------------------------------------------------------------------------

The main purpose of this repository is to provide reproducible code for the GA-PINN methodology described in the paper [Gradient-Annihilated PINNs for Solving Riemann Problems: Application to Relativistic Hydrodynamics](https://arxiv.org/abs/2305.08448). The folders and scripts are distributed as follows:

- **Ground_Data/**: Folder containing the corresponding *ground truth* solutions used through the numerical examples of the paper. This folder contains one subfolder the Problem 3 that has been studied. Inside that folder, one should find two *.h5* files corresponding to the analytical and the HRSC (numerical) resolutions for the physical variables at final time. Both, the analytical and the numerical files, contain the following keys:
    - *dens_calculated*, *dens_initial*: Arrays of the density ($\rho$) at final time ($tmax$) and at initial time, respectively.
    - *ur_calculated*, *ur_initial*: Arrays of the velocity ($u$) at final time ($tmax$) and at initial time, respectively.
    - *p_calculated*, *p_initial*: Arrays of the pressure ($p$) at final time ($tmax$) and at initial time, respectively.
    - *w_calculated*, *w_initial*: Arrays of the Lorentz factor ($W$) at final time ($tmax$) and at initial time, respectively.
    - *x_space*: Array of the spatial domain, going from $xmin$ to $xmax$ depending on the problem under consideration.
- **Example_Problem/**: This folder contains example Python code using PyTorch to train a GA-PINN, corresponding to the *training_script.py* file. By default, the images and exported data are saved in the *Images/* and *Models_Data/* subfolders respectively, while the *.pt* files corresponding to the weights of the neural model are stored inside the *Models_Data/Model_Saved/* folder. In particular, the code is already written in order to reproduce the initial conditions of the *Problem 3* presented in the [paper](https://arxiv.org/abs/2305.08448) (Section 4.2).
- **custom_activations.py**: Script containing the custom activation functions that may be needed in the training procedures. These functions consider the slope parameter that could be considered trainable, as explained in the paper. More can be defined if needed, but as it is right now the script contains functions such as the hyperbolic tangent, the sigmoid, Heaviside, and softplus functions.
- **models.py**: This script contains all the necessary code to define the *GA_PINN* class as reading from the *torch.nn.Module*. This class is imported into the aforementioned *training_script.py* of the example. As input, it admits:
    - *X_r*: Tensor of shape *(N_r,2)* where *N_r* corresponds to the number of points in the spatial internal domain $\Omega$, and *2* refers to 2 inputs since the first column will be the spatial tensor ($x$) and the second one will correspond to the temporal domain ($t$).
    - *X_0*: In a similar way, this tensor of shape *(N_0,2)* corresponds to the physical domain of the initial space, meaning that the time will be just a column tensor filled with $tmin$.
    - *U_0*: Tensor of shape *(N_0,3)* where each column corresponds to the initial condition of each one of the physical variables: density, velocity and pressure, respectively.
    - *ground_truth*: Tensor of 3 columns as well, representing the analytical solution. Thus, each column correspond, in order, to the *dens_calculated*, *ur_calculated* and *p_calculated* after transforming them into column torch tensors.
- **utils.py**: Script containing some additional functions to save the results obtained as *.h5* files and plotting them.
- **config.json**: This file corresponds to the configuration JSON of the training and the model. It contains several fields that can be modified in order to train the GA-PINN.
    - *physical*:
        - *parameters*:
            - *adiabatic_constant*: $\Gamma$ factor of the equations of the hydrodynamics.
            - *temporal_range*: List containing the *[tmin,tmax]* values.
            - *spatial_range*: List containing the *[xmin,xmax]* values.
            - *N_r* (string): Number containing the amount of points sampled for the internal physical domain. When using Sobol sampling method it must be a power of two. 
            - *N_0*: Number containing the amount of points sampled for the initial physical domain. For *X_0* we do not Sobol since it can be understood as a linear grid (time is fixed to *tmin*), but we consider also a power of two just for simplicity.
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
            - *n*: The *acceleration* hyperparameter of the trainable slopes, taking into account the one for the hidden layers and the list of parameters for the output variables (density, velocity, pressure).
            - *initial_slopes*: In a similar way, here we can define the initial slopes. These parameters will be multiplied by the *n* hyperparameter and will be trainable thorugh the process (in case that we have previously selected the custom activation functions; if we choose to use vanilla activations from PyTorch such as *nn.Sigmoid()*, these will not be used). We encourage the user to define both the initial slopes and the accelerations in order to get a product such as $n\cdot a=1.0$ or otherwise stability problems may arise.
        - *loss_function_parameters*:
            - *w_R*: Weight of the residual (collocation) part of the loss, $\hat{\mathcal{L}}_{\mathcal{R}}$.
            - *w_IC*: List of the weights of the initial part of the loss, $\mathcal{L}_{\mathcal{IC}}$, corresponding to each one of the variables (density,velocity,pressure).
            - *lambda_to_use*: $\Lambda$ function that we want to consider. It accepts *lambda_1* and *lambda_2* as defined in the paper.
            - *alpha_set*: Set of hyperparameters corresponding to ($\alpha_{\rho}$, $\alpha_{u}$, $\alpha_{p}$), that is, the weight of the respective gradients in the GA-PINN methodology.
            - *beta_set*: Analogously, this defines the set of hyperparameters ($\beta_{\rho}$, $\beta_{u}$, $\beta_{p}$) corresponding to the exponents of the respective gradients.
    - *training_process*:
        - *import*: This key contains the *analytical_solution_path* parameter referring to the path pointing to the *.h5* file containing the analytical / ground data.
        - *export*: This field contains some configurations about the folders where we want the data and images to get stored into. In addition, the *save_each* parameter corresponds to the number of epochs how often we desire to save the information.
        - *parameters*: Here we can define the optimizer to use (Adam, AdamW, RAdam and SGD are available), as well as the learning rate, the number of epochs and the random seed.

---------------------------------------------------------------------------------------------------------------------------------

### Methodology ###

<img src=./figures/Methodology_Diagram.png width="767" height="390"/>

General procedure of our methodology. The input to our neural network is a set of physical variables that include both time and space components, represented as the pair $(t, x)$. The neural architecture can vary in terms of the number of layers, internal activation functions, and the number of neurons per layer. Some general aspects, such as the adaptivity of the activation slopes ak will be dynamical, while there may be subtle variations depending on the problem at hand. The output of the network corresponds to the primitive variables $(\rho, u, p)$. These variables are subject to automatic differentiation to determine the residuals of the physical constraints and the initial conditions, which are then combined using weights ($\omega_{\mathcal{IC}}$, $\omega_{\mathcal{R}}$). Finally, the final loss function $\mathcal{L}$ is evaluated to determine whether the procedure should terminate after updating the network parameters, $\Theta$.

---------------------------------------------------------------------------------------------------------------------------------

### Some results ###

<img src=./figures/Results_Problem_2.png width="750" height="500"/>

Solution for the physical variables at $t=0.4$ for Problem 3 (Section 4.3). The analytical solution is represented by a solid black curve. The solutions obtained with our GA-PINNs method are shown in the top row while those for the base PINN model are displayed in
the bottom row. In the two cases the solutions are represented by blue circles.

<img src=./figures/Problem_2_extra_plots.png width="800" height="270"/>

Evolution of the physical losses (left plot) and relative l2 errors (middle plot) with the epoch number for Problem 3. The
analytical solution is used as *ground truth* to compute the relative errors. The dashed green line in the middle plot represents the
corresponding error for the numerical solution obtained with the second-order central HRSC scheme with a mesh of 400
zones ($\Delta x=0.0025$). The heat map on the right plot shows the $\Lambda$ function after training.

---------------------------------------------------------------------------------------------------------------------------------

### Requirements ###

The entire code is written using PyTorch for Python 3.8.10. Main used packages are:
* PyTorch 
* SciPy
* Matplotlib

The requirements can be installed using the available *requirements.txt* file. You may want to use:

`pip install -r requirements.txt`

Although some distributions as Anaconda may require to use *conda* instead of *pip* after activating a particular environment.

---------------------------------------------------------------------------------------------------------------------------------

### How to train the example problem ###

The *config.json* is already prepared to reproduce the Problem 3 of the original [paper](https://arxiv.org/abs/2305.08448), although the obtained results may not be identical as the resolutions presented in the project. In order to run the training procedure once the *config.json* has been prepared, is to execute the command:

`python3 training_script.py`

The script will take a GPU as device if it exists. The training process may take more or less time depending on the hardware used. For example, the entire procedure takes ~4h in a Quadro RTX 6000 for a total of 200,000 epochs.