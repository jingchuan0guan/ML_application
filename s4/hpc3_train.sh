stseed=1001
fnseed=1006

for i in $( seq $((stseed)) $((fnseed)) ); do
    if [ "$i" -eq 0 ]; then
        top_k=-1
    else
        top_k=1
    fi
    # python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i loader.batch_size=16 model.layer.imag_scaling=inverse
    # python -m train experiment=s4-lra-cifar-new train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] callbacks.model_checkpoint.save_top_k=$top_k # model.layer.imag_scaling=inverse
    python -m train experiment=s4-lra-cifar-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=None model.layer.imag_scaling=inverse callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_cifar
    python -m train experiment=s4-lra-cifar-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=None model.layer.imag_scaling=linear callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_cifar
    python -m train experiment=s4-lra-cifar-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=conjugate_linear model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_cifar

    # python -m train experiment=s4-lra-cifar-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=None model.layer.imag_scaling=inverse callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_cifar
    # python -m train experiment=s4-lra-cifar-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=None model.layer.imag_scaling=linear callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_cifar
    # python -m train experiment=s4-lra-cifar-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=conjugate_linear model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_cifar
done

# nohup python -m train experiment=s4-lra-pathx model=s4d_mf trainer.devices="[0]" loader.batch_size=16 model.layer.rho=0.95 > log_s4d_mf95.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathx model=s4d_mf trainer.devices="[1]" loader.batch_size=16 model.layer.rho=0.99 > log_s4d_mf99.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathx model=s4d_mf trainer.devices="[2]" loader.batch_size=16 model.layer.rho=0.995 > log_s4d_mf995.out 2>&1 &
# # nohup python -m train experiment=s4-lra-pathx model=base trainer.devices="[3]" loader.batch_size=8 > log_base.out 2>&1 & ### no pool why??
# nohup python -m train experiment=s4-lra-pathx model=s4 trainer.devices="[4]" loader.batch_size=8 > log_s4.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathx model=s4d trainer.devices="[5]" loader.batch_size=16 model.layer.imag_scaling=linear > log_s4dlin.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathx model=s4d trainer.devices="[6]" loader.batch_size=16 model.layer.imag_scaling=inverse > log_s4dinv.out 2>&1 &

# nohup python -m train experiment=s4-lra-pathfinder model=s4d trainer.devices="[7]" layer.imag_scaling=linear > log_s4dlin_path.out 2>&1 &



# nohup python -m train experiment=s4-lra-pathx model=s4 trainer.devices="[0]" loader.batch_size=8 model.layer.dt_max=0.001 > log_s4_x.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathx model=s4d trainer.devices="[3]" loader.batch_size=16 model.layer.imag_scaling=linear model.layer.dt_max=0.001 > log_s4dlin_x.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathx model=s4d trainer.devices="[2]" loader.batch_size=16 model.layer.imag_scaling=inverse model.layer.dt_max=0.001 > log_s4dinv_x.out 2>&1 &

# nohup python -m train experiment=s4-lra-pathfinder model=s4 trainer.devices="[5]" model.layer.dt_max=0.001 > log_s4_path.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathfinder model=s4d trainer.devices="[7]" model.layer.imag_scaling=linear model.layer.dt_max=0.001 > log_s4dlin_path.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathfinder model=s4d trainer.devices="[4]" model.layer.imag_scaling=inverse model.layer.dt_max=0.001 > log_s4dinv_path.out 2>&1 &

# ## new configs
# python -m train experiment=s4-lra-pathx-new model=s4d train.seed=1100 trainer.devices=[0,1] loader.batch_size=16 model.layer.eigvals_name=None model.layer.imag_scaling=inverse callbacks.model_checkpoint.save_top_k=-1
# # linear checkpoint.
# python -m train experiment=s4-lra-pathx-new model=s4d train.seed=1100 trainer.devices=[0,1] loader.batch_size=16 model.layer.eigvals_name=None model.layer.imag_scaling=linear callbacks.model_checkpoint.save_top_k=-1
# python -m train experiment=s4-lra-pathx-new model=s4d train.seed=1100 trainer.devices=[0,1] loader.batch_size=16 model.layer.eigvals_name=conjugate_linear model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=-1
