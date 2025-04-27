from data_lra import*
from ESN_lra import*

seed_setup, seed_dataset = 1234, 5678  # you can freely change here
dim_rv, rho = 100, 0.9
input_dim, batch_size = img_size, 32
num_patch = img_size
dataset_info = dict(t_washout=dim_rv, t_train=2000, t_eval=1000)

w_in, net, w_out = create_setup(seed_setup, input_dim, dim_rv, rho, f=np.tanh)
train_and_eval(
    w_in, net, w_out, batch_size,
    num_patch=img_size, dataloader_cls=PathFinderDataLoader, seed=seed_dataset,
    **dataset_info)
