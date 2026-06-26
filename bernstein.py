import torch 
import torch.nn as nn
import torch.nn.functional as F

#custom neural network layer whose output is based on a Bernstein polynomial
class BernsteinLayer(nn.Module):

  def __init__(self, in_shape, degree: int):
    super().__init__()
    
    ##print the bound statistics
    self._clamp_total = 0
    self._clamp_count = 0
    self._xnorm_min = float("inf")
    self._xnorm_max = float("-inf")
    self._xnorm_sum = 0.0
    self._xnorm_count = 0
    ##used to track out of bound values
    self.range_penalty = torch.tensor(0.0)


    #polynomial degree n of the bernstein basis functions
    self.degree = degree
    
    self.in_shape = in_shape
    
    #all possible k values from 0 to degree
    _basis_indices_tensor = torch.arange(degree + 1).reshape((-1, degree + 1))
    #fixed degree n
    _deg_tensor = torch.tensor([degree]).reshape(-1, 1)
    #compute binomial coefficiens used in basis functions using helper function 'binom' defined below
    nCk_tensor = self.binom(_deg_tensor, _basis_indices_tensor)

    #makes a tensor with the same shape as the input tensor but with an extra dimension for lower and upper bounds
    #filled w zeroes as a placeholder
    input_bounds = torch.zeros((*in_shape, 2))

    #Buffer: non-trainable tensors
    #bounds, degree, indices, binomial coeficients
    self.register_buffer("input_bounds", input_bounds)
    self.register_buffer("_basis_indices", _basis_indices_tensor)
    self.register_buffer("_deg_tensor", _deg_tensor)
    self.register_buffer("nCk", nCk_tensor)

    #used to determine if bounds have been set and can be used for input normalization
    
    self.use_bounds = True
    
    #initialize learnable parmameters: n+1 bernstein coefficients
    bern_coeffs = torch.empty(*in_shape, degree + 1)
    torch.nn.init.xavier_uniform_(bern_coeffs)
    self.bern_coeffs = nn.Parameter(bern_coeffs)

  def reset_xnorm_stats(self):
      self._xnorm_min = float("inf")
      self._xnorm_max = float("-inf")
      self._xnorm_sum = 0.0
      self._xnorm_count = 0

  def get_xnorm_stats(self):
      if self._xnorm_count == 0:
          return 0.0, 0.0, 0.0
      mean = self._xnorm_sum / self._xnorm_count
      return self._xnorm_min, mean, self._xnorm_max
  
  #range of possible Bernstein coefficient values for each input dimension
  @property
  def bern_bounds(self):
    #Convex Hull: Bernsetin polynomial is bounded by the minimum and maximum value of its coefficients
    lb, _ = self.bern_coeffs.min(axis = -1, keepdim=True)
    ub, _ = self.bern_coeffs.max(axis = -1, keepdim=True)
    return torch.concat((lb,ub), dim=-1)
 
  #computes binmoial coefficients: (n choose k) = (n!)/(k!(n-k)!)
  def binom(self, n, k):
    nCk = torch.lgamma(n + 1) - torch.lgamma(k + 1) - torch.lgamma(n - k + 1)
    nCk = torch.exp(nCk)
    nCk = torch.floor(nCk + 0.5)
    return nCk

  #Bernstein basis functions for a given input x
  def bern_basis(self, x):
    y = x.unsqueeze(-1)
    
    #computing basis functions using the bernstein basis formula
    basis = (
        self.nCk #precomputed binomial coeff
        * (y) ** self._basis_indices
        * (1 - y) ** (self._deg_tensor - self._basis_indices)
    )
    #basis contains all Bernstein basis values for each input x
    return basis


  #fwd pass of bernstein layer
  #computes the layer’s output bern poly value from the input x using the Bern basis and learned coefficients
  def forward(self, x):
      
    #first few "warmup" epochs, bounds are not set -> use_bounds = False
    if self.use_bounds:
        x_norm = (x - self.input_bounds[..., 0]) / (
            self.input_bounds[..., 1] - self.input_bounds[..., 0] + 1e-8
        )
    else:
        x_norm = x

    ###############################################33333
    # ----- DEBUG: accumulate clamp stats -----
    if self.training:
        with torch.no_grad():
            clamped = (x_norm < 0.0) | (x_norm > 1.0)
            self._clamp_total += clamped.sum().item()
            self._clamp_count += clamped.numel()
            
    # ---- accumulate bound stats ----
    if self.training:
        with torch.no_grad():
            self._xnorm_min = min(self._xnorm_min, x_norm.min().item())
            self._xnorm_max = max(self._xnorm_max, x_norm.max().item())
            self._xnorm_sum += x_norm.sum().item()
            self._xnorm_count += x_norm.numel()
    # ------- find out of bound penalty ------
    if self.training:
      below = F.relu(-x_norm)
      above = F.relu(x_norm - 1.0)
      self.range_penalty = (below + above).mean()
    else:
        # keep a tensor on the right device/type
        self.range_penalty = x_norm.new_zeros(())
    ###########################################################################33

    # clamp
    x = torch.clamp(x_norm, 0.0, 1.0)

    
    #compute basis functions
    basis = self.bern_basis(x)

    '''
    #check if sum of all basis function for input x = 1
    with torch.no_grad():
        basis_shape = torch.tensor(basis.shape)
        basis_sum_to_one = torch.isclose(
            torch.sum(basis, axis=-1).sum(), basis_shape[:-1].prod().float()
        )
        if not basis_sum_to_one:
            raise Exception(
                f"Basis doesn't sum to 1, {torch.sum(basis,axis = -1).sum()}"
            )
    '''

    #output = weighted sum with learned bernstein coeffs
    out = basis * self.bern_coeffs
    out = out.sum(axis=-1)
    return out
