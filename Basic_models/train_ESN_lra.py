# import sys
# import itertools
# import scipy.optimize
import numpy as np
from tqdm import tqdm, trange
from scipy.sparse.linalg import eigs, ArpackNoConvergence

from .data_lra import*

class Module(object):
    def __init__(self, *_args, seed=None, rnd=None, dtype=np.float64, **_kwargs):
        if rnd is None:
            self.rnd = np.random.default_rng(seed)
        else:
            self.rnd = rnd
        self.dtype = dtype


class Linear(Module):
    def __init__(self, input_dim: int, output_dim: int, bound: float = None, bias: float = 0.0, **kwargs):
        """
        Linear model

        Args:
            input_dim (int): input dimension
            output_dim (int): output dimension
            bound (float, optional): sampling scale for weight. Defaults to None.
            bias (float, optional): sampling scale for bias. Defaults to 0.0.
        """
        super(Linear, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        if bound is None:
            bound = math.sqrt(1 / input_dim)
        self.weight = self.rnd.uniform(-bound, bound, (output_dim, input_dim)).astype(self.dtype)
        self.bias = self.rnd.uniform(-bias, bias, (output_dim,)).astype(self.dtype)

    def __call__(self, x: np.ndarray):
        # print(self.weight.shape, x.shape, self.bias.shape)
        out = x @ self.weight.T + self.bias
        return out


class ESN(Module):
    def __init__(
        self,
        dim: int,
        sr: float = 1.0,
        f=np.tanh,
        a: float | None = None,
        p: float = 1.0,
        init_state: np.ndarray | None = None,
        normalize: bool = True,
        **kwargs,
    ):
        """
        Echo state network [Jaeger, H. (2001). Bonn, Germany:
        German National Research Center for Information Technology GMD Technical Report, 148(34), 13.]

        Args:
            dim (int): number of the ESN nodes
            sr (float, optional): spectral radius. Defaults to 1.0.
            f (callable, optional): activation function. Defaults to np.tanh.
            a (float | None, optional): leaky rate. Defaults to None.
            p (float, optional): density of connection matrix. Defaults to 1.0.
            init_state (np.ndarray | None, optional): initial states. Defaults to None.
            normalize (bool, optional): decide if normalizing connection matrix. Defaults to True.
        """
        super(ESN, self).__init__(**kwargs)
        self.dim = dim
        self.sr = sr
        self.f = f
        self.a = a
        self.p = p
        if init_state is None:
            self.x_init = np.zeros(dim, dtype=self.dtype)
        else:
            self.x_init = np.array(init_state, dtype=self.dtype)
        self.x = np.array(self.x_init)
        # generating normalzied sparse matrix
        while True:
            try:
                self.w_net = self.rnd.normal(size=(self.dim, self.dim)).astype(self.dtype)
                if self.p < 1.0:
                    self.w_net *= self.rnd.normal(size=(self.dim, self.dim)) < self.p
                    # sparse matrix
                    # TODO (optional!) implement sparse matrix with density p
                if normalize:
                    spectral_radius = np.max(np.abs(np.linalg.eigvals(self.w_net) ) )
                    # TODO calculate `spectral_radius`
                    ...
                    self.w_net = self.w_net / spectral_radius
                break
            except ArpackNoConvergence:
                continue

    def __call__(self, x: np.ndarray, v: np.ndarray | None = None):
        # TODO calculate the next state
        x_next = self.f(x @ self.w_net.T*self.sr + v)
        if self.a is None:
            return x_next
        else:
            return (1 - self.a) * x + self.a * x_next

    def step(self, v: np.ndarray | None = None):
        self.x = self(self.x, v)


# class LRReadout(Linear):
#     def train(self, x: np.ndarray, y: np.ndarray):
#         assert (x.ndim == 2) and (x.shape[-1] == self.input_dim)
#         assert (y.ndim == 2) and (y.shape[-1] == self.output_dim)
#         print("x", x.shape)
#         T = x.shape[0]
#         x = np.concatenate((np.ones((T, 1)), x), axis=1)
#         print("x", x.shape)
#         w_out = np.linalg.pinv(x.T @ x) @ x.T @ y
#         print("w_out", w_out.shape)
#         self.bias = w_out[0, :][:, np.newaxis]
#         self.bias = self.bias.T
#         print("self.bias", self.bias.shape)
#         self.weight = w_out[1:,:]
#         self.weight = self.weight.T
#         print("self.weight", self.weight.shape)
#         return self.weight, self.bias

class BatchLRReadout(Linear):
    def train(self, x: np.ndarray, y: np.ndarray):
        assert (x.ndim > 1) and (x.shape[-1] == self.input_dim)
        assert (y.ndim > 1) and (y.shape[-1] == self.output_dim)
        x_biased = np.ones((*x.shape[:-1], x.shape[-1] + 1), dtype=self.dtype)
        x_biased[..., 1:] = x
        sol = np.matmul(np.linalg.pinv(x_biased), y)
        self.weight = sol[..., 1:, :].swapaxes(-2, -1)
        self.bias = sol[..., :1, :]
        return self.weight, self.bias


def calc_batch_nrmse(y, yhat):
    mse = y - yhat
    mse = (mse**2).mean(axis=-2)
    var = y.var(axis=-2)
    nrmse = (mse / var) ** 0.5
    return nrmse

def create_setup(seed, dim, rho, a=None, f=np.tanh, bound=1.0, bias=0.0, cls=BatchLRReadout):
    rnd = np.random.default_rng(seed)
    w_in = Linear(1, dim, bound=bound, bias=bias, rnd=rnd)
    net = ESN(dim, sr=rho, f=f, a=a, rnd=rnd)
    w_out = cls(dim, 1)
    return w_in, net, w_out


def eval_nrmse(xs, ys, w_out, time_info, return_out=False, **kwargs):
    t_washout, t_eval = time_info["t_washout"], time_info["t_eval"]
    x_train, y_train = xs[..., t_washout:-t_eval, :], ys[..., t_washout:-t_eval, :]
    x_eval, y_eval = xs[..., -t_eval:, :], ys[..., -t_eval:, :]
    out = w_out.train(x_train, y_train, **kwargs)
    y_out = w_out(x_eval)
    nrmse = calc_batch_nrmse(y_eval, y_out)
    if return_out:
        return nrmse, *out
    else:
        return nrmse

def sample_dynamics(x0, w_in, net, ts, vs, display=False):
    assert vs.shape[-2] == ts.shape[0]
    x = x0
    xs = np.zeros((*x.shape[:-1], ts.shape[0], x.shape[-1]))
    for idx in trange(ts.shape[0], display=display):
        x = net(x, w_in(vs[..., idx, :]))
        xs[..., idx, :] = x
    return xs

def sample_dataset(
    seed,
    dataloader,
    t_washout=1000,
    t_train=2000,
    t_eval=1000,
):
    rnd = np.random.default_rng(seed)
    t_total = t_washout + t_train + t_eval
    ts = np.arange(-t_washout, t_train + t_eval)
    # us = rnd.uniform(-1, 1, (t_total, 1))
    # ys = narma_func(us, np.zeros((10, 1)), **narma_parameters)
    time_info = dict(t_washout=t_washout, t_train=t_train, t_eval=t_eval)
    return ts, datas, labels, time_info


def train_and_eval(x0, w_in, net, w_out, ts, datas, labels, time_info, display=False):
    assert datas.shape[-2] == ts.shape[0]
    assert labels.shape[-2] == ts.shape[0]
    xs = sample_dynamics(x0, w_in, net, ts, datas, display=display)
    nrmse = eval_nrmse(xs, labels, w_out, time_info)
    return nrmse, xs

seed_setup, seed_dataset = 1234, 5678  # you can freely change here
dim, rho = 100, 0.9
batch_size = 32
dataset_info = dict(t_washout=dim, t_train=2000, t_eval=1000)
num_samples = dataset_info["t_train"]+dataset_info["t_eval"]

w_in, net, w_out = create_setup(seed_setup, dim, rho, f=np.tanh)

dataloader = PathFinderDataLoader(num_samples=num_samples, batch_size=batch_size)
ts, datas, labels, time_info = sample_dataset(seed_dataset, dataloader, **dataset_info)


sigmas = np.logspace(-2, 0, 21)  # 10^{-2.0}, 10^{-1.9}, ... 10^{0.0}
x0 = np.zeros((sigmas.shape[0], net.dim))

# symmetrical case (phi=0)
# vs_sym = convert_us_into_vs(us, sigmas, np.zeros_like(sigmas))
nrmse_sym, _xs_sym = train_and_eval(x0, w_in, net, w_out, ts, datas, labels, time_info)
best_sym = np.argmin(nrmse_sym[:, 0])
