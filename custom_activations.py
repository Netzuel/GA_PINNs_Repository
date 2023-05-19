import torch
import torch.nn as nn

# Define Heaviside activation.
class HeavisideAct(nn.Module):
	def __init__(self):
		super().__init__()
	def forward(self, x):
		return torch.where(x < 0, 0.0, 1.0)

# Define custom "slopped" sigmoid activation. Custom slope.
class TrainableSig(nn.Module):
	def __init__(self, parameter, n):
		super().__init__()
		self.parameter = n * parameter
	def forward(self, x):
		return torch.sigmoid(self.parameter * x)

# Define custom softplus activation. Custom slope.
class TrainableSoftplus(nn.Module):
	def __init__(self, parameter, n):
		super().__init__()
		self.parameter = parameter
	def forward(self, x):
		return (1 / self.parameter) * torch.log( 1 + torch.exp(self.parameter * x) )

# Define second version of softplus activation. The "self.parameter" is not multiplying. Custom slope.
class TrainableSoftplus2(nn.Module):
	def __init__(self, parameter, n):
		super().__init__()
		self.parameter = parameter
	def forward(self, x):
		return torch.log( 1 + torch.exp(self.parameter * x) )

# Define hyperbolic tangent with custom parameter. Custom slope.
class TrainableTanh(nn.Module):
	def __init__(self, parameter, n):
		super().__init__()
		self.parameter = n * parameter
	def forward(self, x):
		return torch.tanh(self.parameter * x)

# Define custom SiLU activation. Custom slope.
class TrainableSiLU(nn.Module):
	def __init__(self, parameter, n, a_act):
		super().__init__()
		self.parameter = n * parameter
		self.a_act = a_act
	def forward(self, x):
		return (self.a_act * x) * torch.sigmoid(self.parameter * x)

# Define custom ELU activation. Custom slope.
class TrainableELU(nn.Module):
	def __init__(self, parameter, n):
		super().__init__()
		self.parameter = n * parameter
	def forward(self, x):
		return torch.nn.ELU()(self.parameter * x)