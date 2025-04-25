stseed=1001
fnseed=1006

# for i in $( seq $((initseed)) $((fnseed)) ); do
#     python -m train experiment=s4-lra-pathx model=s4d train.seed=$i loader.batch_size=16 model.layer.imag_scaling=linear
# done

# for i in $( seq $((initseed)) $((fnseed)) ); do
#     python -m train experiment=s4-lra-pathx model=s4d train.seed=$i loader.batch_size=16 model.layer.imag_scaling=inverse
# done

for i in $( seq $((initseed)) $((fnseed)) ); do
    python -m train experiment=s4-lra-pathx model=s4d_mf train.seed=$i loader.batch_size=16 model.layer.rho=0.99
done

# nohup python MC_3exs_1ps_rho.py $div $i &
