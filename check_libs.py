import torch
print(torch.version.cuda)  # PyTorchがサポートするCUDAのバージョン
print(torch.cuda.is_available())  # CUDAが使用可能かどうか
print(torch.backends.cudnn.version())  # cuDNNのバージョン

from torch.utils.collect_env import get_pretty_env_info
print(get_pretty_env_info())

import pykeops
pykeops.clean_pykeops()
# pykeops.test_numpy_bindings()    # perform the compilation
pykeops.test_torch_bindings()    # perform the compilation
