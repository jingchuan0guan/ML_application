# import sys
# import itertools
# import scipy.optimize
import math
import numpy as np
from tqdm import tqdm, trange
from scipy.sparse.linalg import eigs, ArpackNoConvergence

# from data_lra import*

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
        dim_rv: int,
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
            dim_rv (int): number of the ESN nodes
            sr (float, optional): spectral radius. Defaults to 1.0.
            f (callable, optional): activation function. Defaults to np.tanh.
            a (float | None, optional): leaky rate. Defaults to None.
            p (float, optional): density of connection matrix. Defaults to 1.0.
            init_state (np.ndarray | None, optional): initial states. Defaults to None.
            normalize (bool, optional): decide if normalizing connection matrix. Defaults to True.
        """
        super(ESN, self).__init__(**kwargs)
        self.dim_rv = dim_rv
        self.sr = sr
        self.f = f
        self.a = a
        self.p = p
        if init_state is None:
            self.x_init = np.zeros(dim_rv, dtype=self.dtype)
        else:
            self.x_init = np.array(init_state, dtype=self.dtype)
        self.x = np.array(self.x_init)
        # generating normalzied sparse matrix
        while True:
            try:
                self.w_net = self.rnd.normal(size=(self.dim_rv, self.dim_rv)).astype(self.dtype)
                if self.p < 1.0:
                    self.w_net *= self.rnd.normal(size=(self.dim_rv, self.dim_rv)) < self.p
                    # sparse matrix
                    # TODO (optional!) implement sparse matrix with density p
                if normalize:
                    spectral_radius = np.max(np.abs(np.linalg.eigvals(self.w_net) ) )
                    # TODO calculate `spectral_radius`
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

def calc_batch_nrmse(y, yhat):
    mse = y - yhat
    mse = (mse**2).mean(axis=-2)
    var = y.var(axis=-2)
    nrmse = (mse / var) ** 0.5
    return nrmse


### newly defined
class BatchLR_Optimizer_Readout(Linear):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8,**kwargs):
        super().__init__(**kwargs)
        self.lr = lr
        
        ### for adam
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        
        self.m_w = np.zeros_like(self.weight)
        self.v_w = np.zeros_like(self.weight)
        self.m_b = np.zeros_like(self.bias)
        self.v_b = np.zeros_like(self.bias)

    def Adam_step_w(self, grad):
        self.t += 1
        self.m_w = self.beta1 * self.m_w + (1 - self.beta1) * grad
        self.v_w = self.beta2 * self.v_w + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m_w / (1 - self.beta1 ** self.t)
        v_hat = self.v_w / (1 - self.beta2 ** self.t)
        self.weight -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
        # print(self.weight.shape)
    
    def Adam_step_b(self, grad):
        self.m_b = self.beta1 * self.m_b + (1 - self.beta1) * grad
        self.v_b = self.beta2 * self.v_b + (1 - self.beta2) * (grad ** 2)
        m_hat = self.m_b / (1 - self.beta1 ** self.t)
        v_hat = self.v_b / (1 - self.beta2 ** self.t)
        self.bias -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def get_params(self):
        return self.self.weight
    
    def train(self, x: np.ndarray, label: np.ndarray):
        # print("train shapes xy", x.shape, label.shape)
        assert (x.ndim > 1) and (x.shape[-1] == self.input_dim)
        assert (label.ndim > 1) and (label.shape[-1] == self.output_dim)
        batch_size = x.shape[-2]
        
        y = x @ self.weight.T + self.bias
        error = y - label
        grad_w = (2 / batch_size) * (error.T @ x)
        grad_b = (2 / batch_size) * np.sum(error, axis=0)#, keepdims=True)
        self.Adam_step_w(grad_w)
        self.Adam_step_b(grad_b)
        return self.weight, self.bias


# choose one of the two
def net_out_batch_last_state(net, w_in, w_out, x0, T, batch_size, dataloader, num_patch, train=False):
    y_out_arr, pre_arr, acc_arr, nrmse_arr = [],[],[],[]
    for idx in trange(T):
        datas, labels = dataloader.__next__()
        x=x0
        for p_rep in range(num_patch):
            x = net(x, w_in(datas[..., p_rep, :]))
            # print("x", x.shape)
        
        if train:
            out = w_out.train(x, labels)
        
        y_out = w_out(x)
        pre = np.round(y_out).astype(int)
        acc = np.sum(pre == labels)/labels.shape[0]
        nrmse = calc_batch_nrmse(labels, y_out)
        y_out_arr.append(y_out), pre_arr.append(pre), acc_arr.append(acc), nrmse_arr.append(nrmse)
    return y_out_arr, pre_arr, acc_arr, nrmse_arr

def net_out_batch_all_states(net, w_in, w_out, x0, T, batch_size, dataloader, num_patch, train=False):
    y_out_arr, pre_arr, acc_arr, nrmse_arr = [],[],[],[]
    for idx in trange(T):
        xs = np.zeros((*x0.shape[:-1], num_patch, x0.shape[-1]))
        datas, labels = dataloader.__next__()
        x=x0
        for p_rep in range(num_patch):
            x = net(x, w_in(datas[..., p_rep, :]))
            # print("x", x.shape)
            xs[..., p_rep, :]=x
        xs = xs.reshape(batch_size, -1)
        if train:
            out = w_out.train(xs, labels)
        
        y_out = w_out(xs)
        pre = np.round(y_out).astype(int)
        acc = np.sum(pre == labels)/labels.shape[0]
        nrmse = calc_batch_nrmse(labels, y_out)
        y_out_arr.append(y_out), pre_arr.append(pre), acc_arr.append(acc), nrmse_arr.append(nrmse)
    return np.array(y_out_arr), np.array(pre_arr), np.array(acc_arr), np.array(nrmse_arr)

def train_and_eval(
    w_in, net, w_out, image_paths, labels, batch_size, num_patch, dataloader_cls, seed=0,
    learning_type=["all_states", "last_state"][0],
    t_washout=1000, t_train=2000, t_eval=1000,
    ):
    time_info = dict(t_washout=t_washout, t_train=t_train, t_eval=t_eval)
    dataloader = dataloader_cls(
        num_samples=(t_train+t_eval)*batch_size, image_paths=image_paths, labels=labels,
        batch_size=batch_size, seed=seed
        )
    
    x0 = np.zeros((batch_size, net.dim_rv))
    for idx in trange(t_washout):
        x0 = net(x0, w_in(np.zeros((batch_size, w_in.input_dim))) )
    
    if learning_type=="last_state":
        train_out=net_out_batch_last_state(net, w_in, w_out, x0, t_train, batch_size, dataloader, num_patch, train=True)
        valid_out=net_out_batch_last_state(net, w_in, w_out, x0, t_eval, batch_size, dataloader, num_patch, train=False)
    if learning_type=="all_states":
        train_out=net_out_batch_all_states(net, w_in, w_out, x0, t_train, batch_size, dataloader, num_patch, train=True)
        valid_out=net_out_batch_all_states(net, w_in, w_out, x0, t_eval, batch_size, dataloader, num_patch, train=False)
    return train_out, valid_out