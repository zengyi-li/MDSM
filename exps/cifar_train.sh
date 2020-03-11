python train.py \
--n_iter 300000 \
--batch_size 128 \
--n_chan 128 \
--n_gpus 2 \
--max_lr 5e-5 \
--min_noise 0.05 \
--max_noise 1.2 \
--noise_distribution 'lin' \
--save_every 5000 \
--dataset 'cifar' \
--log 'CIFAR_EBM' \
--lr_schedule 'cosine' \
--rand_seed 42 \


