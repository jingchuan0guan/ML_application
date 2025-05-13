import sys
from data_lra import*
from ESN_lra import*
np.set_printoptions(threshold=np.inf, precision=10, linewidth=10**4)

args=np.array(sys.argv)# 0:divs, 1:psid
args=args[1:]
print(args)

img_size = int(args[0]) # [32, 64, 128, 256][2]
difficulty = args[1] # ["E", "M", "H"][0] # easy, intermediate, hard

image_paths, labels = load_pathfider_metafiles(
    img_size = img_size, difficulty = difficulty,
    ROOT_DIR = "/home/kan/ML_application/s4/data/pathfinder/", ##### change here
    )
input_dim = [img_size, 1][1] # input_dim*num_patch == img_size**2
num_patch = int(img_size**2/input_dim)
print(input_dim, num_patch)

"""
you can freely change parameters in the below
Note that
" (dataset_info["t_train"]+dataset_info["t_eval"]) * batch_size "
should not exceed the number of "metadata load counter"
"""
seed_setup, seed_dataset = 1234, 5678
dim_rv, rho = 1024, 0.98
batch_size = [64, 64*1500, 128*8][1]
dataset_info = dict(
    t_washout=dim_rv,
    t_train=[2000, 1][1],
    t_eval=[1000, 1][1],
    )
learning_type = ["all_states", "last_state"][0]
optimizer = ["linreg", "adam"][0]
readout_f = ["1", "2", "tanh"][1]
state_expansion_ratio = {"1":1, "2":2, "tanh":1}[readout_f]
assert (dataset_info["t_train"]+dataset_info["t_eval"])*batch_size < 2*10**5 # == metadata load counter

rnd = np.random.default_rng(seed_setup)
w_in = Linear(input_dim, dim_rv, bound=1.0, bias=0.0, rnd=rnd)
net = ESN(dim_rv, sr=rho, f=np.tanh, a=None, rnd=rnd)
out_states_dim = {"last_state":dim_rv*state_expansion_ratio, "all_states":num_patch*dim_rv*state_expansion_ratio}[learning_type]
w_out_cls = {"adam":BatchLR_Optimizer_Readout, "linreg":BatchLRReadout}[optimizer]
w_out = w_out_cls( input_dim = out_states_dim, output_dim = 1 )

train_out, valid_out = train_and_eval(
    net, w_in, w_out, image_paths, labels,
    batch_size, num_patch=num_patch, dataloader_cls=PathFinderDataLoader, seed=seed_dataset,
    learning_type=learning_type, readout_f=readout_f, **dataset_info)

print(learning_type, optimizer, " activation function at the readout", readout_f)

y_out_arr, pre_arr, acc_arr, nrmse_arr = train_out
print("train")
# print("y_out_arr:", y_out_arr)
# print("pre_arr:", pre_arr.reshape(-1))
print("acc_arr:", acc_arr)
# print("nrmse_arr:", nrmse_arr)

y_out_arr, pre_arr, acc_arr, nrmse_arr = valid_out
print("valid")
# print("y_out_arr:", y_out_arr)
# print("pre_arr:", pre_arr.reshape(-1))
print("acc_arr:", acc_arr)
# print("nrmse_arr:", nrmse_arr)
