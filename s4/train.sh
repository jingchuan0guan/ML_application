nohup python -m train experiment=s4-lra-pathx model=s4d_mf trainer.devices="[0]" loader.batch_size=16 model.layer.rho=0.95 > log_s4d_mf95.out 2>&1 &
nohup python -m train experiment=s4-lra-pathx model=s4d_mf trainer.devices="[1]" loader.batch_size=16 model.layer.rho=0.99 > log_s4d_mf99.out 2>&1 &
nohup python -m train experiment=s4-lra-pathx model=s4d_mf trainer.devices="[2]" loader.batch_size=16 model.layer.rho=0.995 > log_s4d_mf995.out 2>&1 &
# nohup python -m train experiment=s4-lra-pathx model=base trainer.devices="[3]" loader.batch_size=8 > log_base.out 2>&1 & ### no pool why??
nohup python -m train experiment=s4-lra-pathx model=s4 trainer.devices="[4]" loader.batch_size=8 > log_s4.out 2>&1 &
nohup python -m train experiment=s4-lra-pathx model=s4d trainer.devices="[5]" loader.batch_size=16 model.layer.imag_scaling=linear > log_s4dlin.out 2>&1 &
nohup python -m train experiment=s4-lra-pathx model=s4d trainer.devices="[6]" loader.batch_size=16 model.layer.imag_scaling=inverse > log_s4dinv.out 2>&1 &

nohup python -m train experiment=s4-lra-pathfinder model=s4d trainer.devices="[7]" layer.imag_scaling=linear > log_s4dlin_path.out 2>&1 &