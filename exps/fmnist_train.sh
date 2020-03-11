python train.py \
--n_iter 100000 \
--batch_size 128 \
--n_chan 64 \
--n_gpus 2 \
--max_lr 1e-4 \
--min_noise 0.1 \
--max_noise 3 \
--noise_distribution 'exp' \
--save_every 5000 \
--dataset 'fmnist' \
--log 'fmnist_EBM' \
--lr_schedule 'cosine' \
--rand_seed 42 \

