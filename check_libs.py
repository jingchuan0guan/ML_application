import torch
# print(torch.version.cuda)  # PyTorchがサポートするCUDAのバージョン
# print(torch.cuda.is_available())  # CUDAが使用可能かどうか
# print(torch.backends.cudnn.version())  # cuDNNのバージョン

# from torch.utils.collect_env import get_pretty_env_info
# print(get_pretty_env_info())

# import pykeops
# pykeops.clean_pykeops()
# pykeops.test_numpy_bindings()    # perform the compilation
# pykeops.test_torch_bindings()    # perform the compilation
import numpy as np

# for N in range(10):
#     w = torch.arange(N//2)
#     print(N, w)


def roots_of_unity(n):
    k = torch.arange(n)
    theta = 2 * torch.pi * k / n
    roots = torch.cos(theta) + 1j * torch.sin(theta)
    return roots

def A_eigvals(N, H=1, imag_scaling='inverse', eigvals_name="conjugate_linear", rho=0.9, n_conjugate=3):
    dtype = torch.cfloat
    pi = torch.tensor(np.pi)
    
    # if imag_scaling in ['random', 'linear', 'inverse']:
    #     real_part = .5 * torch.ones(H, N//2)
    #     imag_part = repeat(torch.arange(N//2), 'n -> h n', h=H)
    #     if imag_scaling == 'random':
    #         imag_part = torch.randn(H, N//2)
    #     elif imag_scaling == 'linear':
    #         imag_part = pi * imag_part
    #     elif imag_scaling == 'inverse': # Based on asymptotics of the default HiPPO matrix
    #         imag_part = 1/pi * N * (N/(1+2*imag_part)-1)
    #     # else: raise NotImplementedError
    #     w = -real_part + 1j * imag_part
    
    n=N//2
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
            print("n p q", n, p,q)
            w = torch.cat(w, dim=0)
    
    B = torch.randn(H, N//2, dtype=dtype)
    norm = -B/w # (H, N) # Result if you integrate the kernel with constant 1 function
    zeta = 2*torch.sum(torch.abs(norm)**2, dim=-1, keepdim=True) # Variance with a random C vector
    B = B / zeta**.5
    return w, B


tensor_list = [torch.tensor([1, 2]), torch.tensor([3]), torch.tensor([4, 5, 6])]
result = torch.cat(tensor_list)
print(result)

N = 10
w, B =A_eigvals(N)
print(np.abs(w))
print(np.abs(w)*N)

N = 11
w, B =A_eigvals(N)
print(np.abs(w))
print(np.abs(w)*N)

N=12
w, B =A_eigvals(N)
print(np.abs(w))
print(np.abs(w)*N)