stseed=1001
fnseed=1006

for i in $( seq $((stseed)) $((fnseed)) ); do
    # python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i loader.batch_size=16 model.layer.imag_scaling=inverse
    python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i trainer.devices=[2,3,4,5,6,7] loader.batch_size=16 # model.layer.imag_scaling=inverse
done

# for i in $( seq $((stseed)) $((fnseed)) ); do
#     python -m train experiment=s4-lra-pathx-new model=s4d train.seed=$i loader.batch_size=16 model.layer.imag_scaling=inverse
# done

# for i in $( seq $((stseed)) $((fnseed)) ); do
#     python -m train experiment=s4-lra-pathx-new model=s4d_mf train.seed=$i loader.batch_size=16 model.layer.rho=0.99
# done

# nohup python MC_3exs_1ps_rho.py $div $i &
