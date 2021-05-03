env=CartPole-v1
algorithm=ppo
name=${env}_${algorithm}
gpu=0
seed=0

CUDA_VISIBLE_DEVICES=$gpu python -m spinup.algos.$algorithm.$algorithm  --exp_name $name --env $env  
    --hid [32,32] [64,32]  --seed $seed  --cpu auto
    