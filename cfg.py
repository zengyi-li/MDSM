# Based on cfg.py file in AutoGAN repo 


import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--n_iter',
        type=int,
        default=300000,
        help='number of iterations of training')
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='batch size during training')
    
    parser.add_argument(
        '--n_chan',
        type=int,
        default=128,
        help='number of channels for first layer of network')
    
    parser.add_argument(
        '--n_gpus',
        type=int,
        default=2,
        help='number of gpus to use')
    
    parser.add_argument(
        '--max_lr',
        type=float,
        default=5e-5,
        help='maximum learning rate')
    

    parser.add_argument(
        '--min_noise',
        type=float,
        default=0.05,
        help='minimum noise level')
    
    parser.add_argument(
        '--max_noise',
        type=float,
        default=1.2,
        help='maximum noise level')
    
    parser.add_argument(
        '--noise_distribution',
        type=str,
        default='lin',
        help='how noise levels distribute on batch direction, choice between lin and exp')
    
    parser.add_argument(
        '--save_every',
        type=int,
        default=5000,
        help='number of iterations of between saving')
    
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar',
        help='dataset to use for training, choice between cifar, mnist, fmnist')
    
    parser.add_argument(
        '--log',
        type=str,
        default='Exp',
        help='logging folder name')
    
    parser.add_argument(
        '--time',
        type=str,
        default=None,
        help='time string of logging folder, only required when continuing training or sampling')
    
    parser.add_argument(
        '--lr_schedule',
        type=str,
        default='exp',
        help='learning rate schedule shape, choice between cosine and exp')
    
    parser.add_argument(
        '--rand_seed',
        type=float,
        default=42,
        help='costom random seed')
    
    
    parser.add_argument(
        '--cont',
        action='store_true',
        help='if continuing training')
        
    
    parser.add_argument(
        '--net_indx',
        type=int,
        default=0,
        help='if continuing training, the point of continuation. If sampling, the network to use')
    
    
    parser.add_argument(
        '--sample_mode',
        type=str,
        default='all',
        help='sampling mode, options: single, all')
    
    parser.add_argument(
        '--annealing_schedule',
        type=str,
        default='exp',
        help='shape of annealing schedule, options: exp, lin')
        
    parser.add_argument(
        '--file_name',
        type=str,
        default='netE_MDSM',
        help='filename, for sampling')
    
    parser.add_argument(
        '--n_samples_save',
        type=int,
        default=20000,
        help='number of samples to save during sampling')
        
    parser.add_argument(
        '--sample_step_size',
        type=float,
        default=0.02,
        help='step size during sampling')
    
    opt = parser.parse_args()

    return opt