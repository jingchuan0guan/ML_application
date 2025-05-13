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
input_dim = img_size # input_dim*num_patch == img_size**2
num_patch = img_size

def net_out_batch_all_states_ablation(
    w_out,T,dataloader,batch_size, 
    # net, w_in, x0, num_patch, learning_type=None,
    readout_f="1", train=False,
    ):
    y_out_arr, pre_arr, acc_arr, nrmse_arr = [],[],[],[]
    for idx in trange(T):
        datas, labels = dataloader.__next__()
        target_state = np.einsum('bij,bik->bjk', datas, datas).reshape(batch_size, -1)
        print("target_state", target_state.shape)
        if readout_f=="2":
            target_state = np.concatenate([target_state, target_state**2], axis=1)
        if readout_f=="tanh":
            target_state = np.tanh(target_state)
        
        if train: # conduct training here
            learned_w = w_out.train(target_state, labels)
        
        y_out = w_out(target_state)
        pre = np.round(y_out).astype(int)
        acc = np.sum(pre == labels)/labels.shape[0]
        nrmse = calc_batch_nrmse(labels, y_out)
        y_out_arr.append(y_out), pre_arr.append(pre), acc_arr.append(acc), nrmse_arr.append(nrmse)
    return np.array(y_out_arr), np.array(pre_arr), np.array(acc_arr), np.array(nrmse_arr)

def train_and_eval_ablation(
    w_out, image_paths, labels, batch_size, dataloader_cls,
    net=None, w_in=None, num_patch=None, learning_type=["all_states", "last_state"][0], readout_f="1",
    seed=0, 
    t_washout=1000, t_train=2000, t_eval=1000, 
    ):
    dataloader = dataloader_cls(
        num_samples=(t_train+t_eval)*batch_size, image_paths=image_paths, labels=labels,
        batch_size=batch_size, seed=seed
        )
    # x0 = np.zeros((batch_size, net.dim_rv))
    # for idx in trange(t_washout):
    #     x0 = net(x0, w_in(np.zeros((batch_size, w_in.input_dim))) )
    
    train_out=net_out_batch_all_states_ablation(
        w_out, t_train, dataloader, batch_size=batch_size, train=True,
        # learning_type,
        readout_f=readout_f,
        )
    valid_out=net_out_batch_all_states_ablation(
        w_out, t_train, dataloader, batch_size=batch_size, train=False,
        # learning_type,
        readout_f=readout_f,
        )
    return train_out, valid_out

"""
you can freely change parameters in the below
Note that
" (dataset_info["t_train"]+dataset_info["t_eval"]) * batch_size "
should not exceed the number of "metadata load counter"
"""
seed_setup, seed_dataset = 1234, 5678
dim_rv, rho = 1024, 0.98
batch_size = [64, 64*1500, 128*8][2]
dataset_info = dict(
    t_washout=dim_rv,
    t_train=[2000, 1][1],
    t_eval=[1000, 1][1],
    )
learning_type = ["input", "all_states", "last_state"][0]
optimizer = ["linreg", "adam"][0]
readout_f = ["1", "2", "tanh"][1]
state_expansion_ratio = {"1":1, "2":2, "tanh":1}[readout_f]
assert (dataset_info["t_train"]+dataset_info["t_eval"])*batch_size < 2*10**5 # == metadata load counter

rnd = np.random.default_rng(seed_setup)
w_in = Linear(input_dim, dim_rv, bound=1.0, bias=0.0, rnd=rnd)
# net = ESN(dim_rv, sr=rho, f=np.tanh, a=None, rnd=rnd)
out_states_dim = {"input":img_size*img_size*state_expansion_ratio,"last_state":dim_rv*state_expansion_ratio, "all_states":num_patch*dim_rv*state_expansion_ratio}[learning_type]
w_out_cls = {"adam":BatchLR_Optimizer_Readout, "linreg":BatchLRReadout}[optimizer]
w_out = w_out_cls( input_dim = out_states_dim, output_dim = 1 )

train_out, valid_out = train_and_eval_ablation(
    w_out, image_paths, labels, batch_size, dataloader_cls=PathFinderDataLoader,
    # net=net, w_in=w_in, num_patch=img_size, learning_type=learning_type,
    readout_f=readout_f,
    seed=seed_dataset, **dataset_info)

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
