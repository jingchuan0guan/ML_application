""" Standalone version of Structured (Sequence) State Space (S4) model. """

import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_only
from einops import rearrange, repeat
import opt_einsum as oe

contract = oe.contract
contract_expression = oe.contract_expression


_c2r = torch.view_as_real
_r2c = torch.view_as_complex
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 10):
    _resolve_conj = lambda x: x.conj().resolve_conj()
else:
    _resolve_conj = lambda x: x.conj()


""" simple nn.Module components """
def Activation(activation=None, dim=-1):
    if activation in [ None, 'id', 'identity', 'linear' ]:
        return nn.Identity()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'gelu':
        return nn.GELU()
    elif activation in ['swish', 'silu']:
        return nn.SiLU()
    elif activation == 'glu':
        return nn.GLU(dim=dim)
    elif activation == 'sigmoid':
        return nn.Sigmoid()
    else:
        raise NotImplementedError("hidden activation '{}' is not implemented".format(activation))

def LinearActivation(
        d_input, d_output, bias=True,
        transposed=False,
        activation=None,
        activate=False, # Apply activation as part of this module
        **kwargs,
    ):
    """ Returns a linear nn.Module with control over axes order, initialization, and activation """

    # Construct core module
    linear_cls = partial(nn.Conv1d, kernel_size=1) if transposed else nn.Linear
    if activation == 'glu': d_output *= 2
    linear = linear_cls(d_input, d_output, bias=bias, **kwargs)

    if activate and activation is not None:
        activation = Activation(activation, dim=-2 if transposed else -1)
        linear = nn.Sequential(linear, activation)
    return linear



def roots_of_unity(n):
    k = torch.arange(n)
    theta = 2 * torch.pi * k / n
    roots = torch.cos(theta) + 1j * torch.sin(theta)
    return roots

""" HiPPO utilities """
def A_eigvals(
    N, H=1, eigvals_name="conjugate_linear", n_conjugate=2,
    rho=0.9, dt=1, # specify both of them
    ):
    dtype = torch.cfloat # pi = torch.tensor(np.pi)
    n = N//2
    if eigvals_name in ["linear", "conjugate_linear"]:
        if eigvals_name == "linear":
            w = torch.arange(n)
        elif eigvals_name == "conjugate_linear":
            q = n % n_conjugate
            if q>0:
                p = 1 + n//n_conjugate
                w = [roots_of_unity(n_conjugate)*i/p for i in range(2, p+1)]    
                w.extend([roots_of_unity(q)])
            elif q==0:
                p = n//n_conjugate  
                w = [roots_of_unity(n_conjugate)*i/p for i in range(1, p+1)]    
            else:
                raise KeyError
            # print(w)
            w = torch.cat(w, dim=0).to(dtype)
            # print("n p q", n, p, q)
    w = repeat((rho*w)**(dt), 'n -> h n', h=H)
    
    B = torch.randn(H, N//2, dtype=dtype)
    norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
    zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
    B = B / zeta**.5
    return w, B



class SSKernelDiag(nn.Module):
    """ 
    Version using (complex) diagonal state matrix. Note that it is slower and less memory efficient than the NPLR kernel because of lack of kernel support.
    """

    def __init__(self, A_eigvals, C, H, lr=None, ):
        super().__init__()

        # Rank of low-rank correction
        assert A_eigvals.size(-1) == C.size(-1)
        self.H = H # log_dt.size(-1)
        self.N = A_eigvals.size(-1)
        assert self.H % A_eigvals.size(0) == 0
        self.copies = self.H // A_eigvals.size(0)

        # Broadcast everything to correct shapes
        C = C.expand(torch.broadcast_shapes(C.shape, (1, self.H, self.N))) # (H, C, N)

        # Register parameters
        self.C = nn.Parameter(_c2r(_resolve_conj(C)))
        # self.register("log_dt", log_dt, False)
        
        A_eigvals_abslog = np.log(torch.abs(A_eigvals) + 1e-6)
        A_eigvals_phase = torch.angle(A_eigvals)
        self.register("A_eigvals_abslog", A_eigvals_abslog, True, lr, 0.0)
        self.register("A_eigvals_phase", A_eigvals_phase, True, lr, 0.0)
        # log_w_real = torch.log(-A_eigvals.real + 1e-5)
        # w_imag = A_eigvals.imag
        # self.register("log_w_real", log_w_real, True, lr, 0.0)
        # self.register("w_imag", w_imag, True, lr, 0.0)

    def _w(self):
        # Get the internal A_eigvals (diagonal) parameter
        # w_real = -torch.exp(self.log_w_real)
        # w_imag = self.w_imag
        A_eigvals = torch.exp(self.A_eigvals_abslog + 1j*self.A_eigvals_phase) # w_real + 1j * w_imag
        A_eigvals = repeat(A_eigvals, 't n -> (v t) n', v=self.copies) # (H N)
        return A_eigvals

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """
        A_eigvals = self._w() # (H N)  == torch.exp(dtA)
        # print(A_eigvals)
        # Incorporate dt into A
        # dt = torch.exp(self.log_dt) # (H)
        # dtA = w * dt.unsqueeze(-1)  # (H N)
        dtA = torch.log(A_eigvals)
        
        C = _r2c(self.C) # (C H N)

        # Power up
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A_eigvals.device) # (H N L)
        C = C * (A_eigvals - 1.) / dtA
        K = contract('chn, hnl -> chl', C, torch.exp(K))
        K = 2*K.real

        return K

    def setup_step(self):
        # dt = torch.exp(self.log_dt) # (H)
        C = _r2c(self.C) # (C H N)
        self.dA = self._w() # (H N)
        dtA = torch.log(self.dA)
        
        # self.dA = torch.exp(dtA) # (H N)
        self.dC = C * (self.dA - 1.) / dtA # (C H N)
        self.dB = self.dC.new_ones(self.H, self.N) # (H N)

    def default_state(self, *batch_shape):
        C = _r2c(self.C)
        state = torch.zeros(*batch_shape, self.H, self.N, dtype=C.dtype, device=C.device)
        return state

    def step(self, u, state):
        next_state = contract("h n, b h n -> b h n", self.dA, state) \
                + contract("h n, b h -> b h n", self.dB, u)
        y = contract("c h n, b h n -> b c h", self.dC, next_state)
        return 2*y.real, next_state


    def register(self, name, tensor, trainable=False, lr=None, wd=None):
        """Utility method: register a tensor as a buffer or trainable parameter"""

        if trainable:
            self.register_parameter(name, nn.Parameter(tensor))
        else:
            self.register_buffer(name, tensor)

        optim = {}
        if trainable and lr is not None:
            optim["lr"] = lr
        if trainable and wd is not None:
            optim["weight_decay"] = wd
        if len(optim) > 0:
            setattr(getattr(self, name), "_optim", optim)

class S4DKernel(nn.Module):
    """Wrapper around SSKernelDiag that generates the diagonal SSM parameters
    """

    def __init__(
        self,
        H,
        N=64,
        eigvals_name="conjugate_linear", n_conjugate=2, rho=0.9, dt=0.1, 
        channels=1, # 1-dim to C-dim map; can think of C as having separate "heads"
        # dt_min=0.001, dt_max=0.1,
        lr=None, # Hook to set LR of SSM parameters differently
        n_ssm=1, # Copies of the ODE parameters A and B. Must divide H
        **kernel_args,
    ):
        super().__init__()
        self.N = N
        self.H = H
        dtype = torch.float
        cdtype = torch.cfloat
        self.channels = channels
        self.n_ssm = n_ssm

        # Generate dt
        # log_dt = torch.zeros(self.H, dtype=dtype)

        # Compute the preprocessed representation
        # Generate low rank correction p for the measure
        w, B = A_eigvals(self.N, H=n_ssm, eigvals_name=eigvals_name, n_conjugate=n_conjugate, rho=rho, dt=dt,)

        C = torch.randn(channels, self.H, self.N // 2, dtype=cdtype)

        # Broadcast tensors to n_ssm copies
        # These will be the parameters, so make sure tensors are materialized and contiguous
        B = repeat(B, 't n -> (v t) n', v=self.n_ssm // B.size(-2)).clone().contiguous()
        w = repeat(w, 't n -> (v t) n', v=self.n_ssm // w.size(-2)).clone().contiguous()

        # Combine B and C using structure of diagonal SSM
        C = C * repeat(B, 't n -> (v t) n', v=H//self.n_ssm)
        self.kernel = SSKernelDiag(
            w, C, self.H, 
            lr=lr,
            **kernel_args,
        )

    def forward(self, L=None):
        k = self.kernel(L=L)
        return k.float()

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state, **kwargs):
        u, state = self.kernel.step(u, state, **kwargs)
        return u.float(), state

    def default_state(self, *args, **kwargs):
        return self.kernel.default_state(*args, **kwargs)


class S4D(nn.Module):

    def __init__(
            self,
            d_model,
            d_state=64,
            channels=1, # maps 1-dim to C-dim
            bidirectional=False,
            # Arguments for FF
            activation='gelu', # activation in between SS and FF
            postact=None, # activation after FF
            dropout=0.0,
            transposed=True, # axis ordering (B, L, D) or (B, D, L)
            # SSM Kernel arguments
            **kernel_args,
        ):
        """
        d_state: the dimension of the state, also denoted by N
        channels: can be interpreted as a number of "heads"
        bidirectional: bidirectional
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, H) or (B, H, L) [B=batch size, L=sequence length, H=hidden dimension]

        Other options are all experimental and should not need to be configured
        """

        super().__init__()

        self.h = d_model
        self.n = d_state
        self.bidirectional = bidirectional
        self.channels = channels
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(channels, self.h))

        if self.bidirectional:
            channels *= 2

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, channels=channels, **kernel_args)

        # Pointwise
        self.activation = Activation(activation)
        dropout_fn = nn.Dropout2d if self.transposed else nn.Dropout
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = LinearActivation(
            self.h*self.channels,
            self.h,
            transposed=self.transposed,
            activation=postact,
            activate=True,
        )


    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing

        Returns: same shape as u
        """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SS Kernel
        k = self.kernel(L=L) # (C H L) (B C H L)

        # Convolution
        if self.bidirectional:
            k0, k1 = rearrange(k, '(s c) h l -> s c h l', s=2)
            k = F.pad(k0, (0, L)) \
                    + F.pad(k1.flip(-1), (L, 0)) \

        k_f = torch.fft.rfft(k, n=2*L) # (C H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y_f = contract('bhl,chl->bchl', u_f, k_f) # k_f.unsqueeze(-4) * u_f.unsqueeze(-3) # (B C H L)
        y = torch.fft.irfft(y_f, n=2*L)[..., :L] # (B C H L)


        # Compute D term in state space equation - essentially a skip connection
        y = y + contract('bhl,ch->bchl', u, self.D) # u.unsqueeze(-3) * self.D.unsqueeze(-1)

        # Reshape to flatten channels
        y = rearrange(y, '... c h l -> ... (c h) l')

        y = self.dropout(self.activation(y))

        if not self.transposed: y = y.transpose(-1, -2)

        y = self.output_linear(y)

        return y, None # Return a None to satisfy this repo's interface, but this can be modified

    def setup_step(self):
        self.kernel.setup_step()

    def step(self, u, state):
        """ Step one time step as a recurrent model. Intended to be used during validation.

        u: (B H)
        state: (B H N)
        Returns: output (B H), state (B H N)
        """
        assert not self.training

        y, next_state = self.kernel.step(u, state) # (B C H)
        y = y + u.unsqueeze(-2) * self.D
        y = rearrange(y, '... c h -> ... (c h)')
        y = self.activation(y)
        if self.transposed:
            y = self.output_linear(y.unsqueeze(-1)).squeeze(-1)
        else:
            y = self.output_linear(y)
        return y, next_state

    def default_state(self, *batch_shape, device=None):
        return self.kernel.default_state(*batch_shape)

    @property
    def d_state(self):
        return self.h * self.n

    @property
    def d_output(self):
        return self.h

    @property
    def state_to_tensor(self):
        return lambda state: rearrange('... h n -> ... (h n)', state)

