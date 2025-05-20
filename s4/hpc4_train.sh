stseed=1001
fnseed=1006

for i in $( seq $((stseed)) $((fnseed)) ); do
    if [ "$i" -eq 0 ]; then
        top_k=-1
    else
        top_k=1
    fi
    # python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i loader.batch_size=16 model.layer.imag_scaling=inverse
    # python -m train experiment=s4-lra-aan-new train.seed=$i trainer.devices=[0,1,2,3] callbacks.model_checkpoint.save_top_k=$top_k # model.layer.imag_scaling=inverse
    
    # python -m train experiment=s4-lra-imdb-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=None model.layer.imag_scaling=inverse callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_imdb
    # python -m train experiment=s4-lra-imdb-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=None model.layer.imag_scaling=linear callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_imdb
    # python -m train experiment=s4-lra-imdb-new model=s4d train.seed=$i trainer.devices=[0,1,2,3,4,5,6,7] model.layer.eigvals_name=conjugate_linear model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_imdb

    ## s4 not diagonal
    # python -m train experiment=s4-lra-pathx-new train.seed=$i trainer.devices=[0,1,2,3] callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_pathx
    # python -m train experiment=s4-lra-listops-new train.seed=$i trainer.devices=[0,1,2,3] callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_listops
    # python -m train experiment=s4-lra-aan-new train.seed=$i trainer.devices=[0,1,2,3] callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_aan

    python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i trainer.devices=[0,1,2,3] model.layer.eigvals_name=None model.layer.imag_scaling=inverse callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_pathx
    python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i trainer.devices=[0,1,2,3] model.layer.eigvals_name=None model.layer.imag_scaling=linear callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_pathx
    python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i trainer.devices=[0,1,2,3] model.layer.eigvals_name=conjugate_linear model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=$top_k wandb.project=hippo_pathx
done

# python -m train experiment=s4-lra-pathx-new train.seed=1100 trainer.devices=[0,1,2,3] loader.batch_size=1 callbacks.model_checkpoint.save_top_k=-1
# python -m train experiment=s4-lra-pathx-new model=s4d train.seed=1100 trainer.devices=[0,1] loader.batch_size=16 model.layer.eigvals_name=conjugate_linear model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=-1 &
# python -m train experiment=s4-lra-pathx-new model=s4d train.seed=1100 trainer.devices=[2,3] loader.batch_size=16 model.layer.eigvals_name=conjugate_linear model.layer.n_conjugate=4 model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=-1 &
# python -m train experiment=s4-lra-pathx-new model=s4d train.seed=1100 trainer.devices=[4,5] loader.batch_size=16 model.layer.eigvals_name=conjugate_linear model.layer.n_conjugate=8 model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=-1 &
# python -m train experiment=s4-lra-pathx-new model=s4d train.seed=1100 trainer.devices=[6,7] loader.batch_size=16 model.layer.eigvals_name=conjugate_linear model.layer.n_conjugate=16 model.layer.imag_scaling=None callbacks.model_checkpoint.save_top_k=-1 &
