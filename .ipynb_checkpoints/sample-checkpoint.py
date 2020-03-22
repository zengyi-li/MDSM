from pathlib import Path
import os
import sys
import shutil
import numpy as np
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torchvision
import cfg
from torchvision.utils import save_image, make_grid

from models.ResNet import Res12_Quadratic, Res18_Quadratic
from functions.sampling import Langevin_E, SS_denoise,  Annealed_Langevin_E,Reverse_AIS_sampling,AIS_sampling
from functions.analysis import save_sample_pdf

def main():
    
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.rand_seed)
    
    
    if args.dataset == 'cifar':
        sample_x = torch.zeros((args.batch_size,3,32,32))
        netE = Res18_Quadratic(3,args.n_chan,32,normalize=False,AF=nn.ELU())
        
        
    elif args.dataset == 'mnist':
        sample_x = torch.zeros((args.batch_size,1,32,32))
        netE = Res12_Quadratic(1,args.n_chan,32,normalize=False,AF=nn.ELU())
        
    elif args.dataset == 'fmnist':
        sample_x = torch.zeros((args.batch_size,1,32,32))
        netE = Res12_Quadratic(1,args.n_chan,32,normalize=False,AF=nn.ELU())
        
    else:
        NotImplementedError('{} unknown dataset'.format(args.dataset))
    #setup gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netE = netE.to(device)
    if args.n_gpus >1:
        netE = nn.DataParallel(netE)
    
    root = 'logs/' + args.log + '_'+ args.time
    
    #single sampling mode save a single file with custom number of images
    #all sampling mode save 
    
    #set annealing schedule
    if args.annealing_schedule =='exp':
        Nsampling = 2000 #exponential schedule with flat region in the beginning and end
        Tmax,Tmin = 100,1
        T = Tmax*np.exp(-np.linspace(0,Nsampling-1,Nsampling)*(np.log(Tmax/Tmin)/Nsampling))
        T = np.concatenate((Tmax*np.ones((500,)),T),axis=0)
        T = np.concatenate((T,Tmin*np.linspace(1,0,200)),axis=0)
        
    elif args.annealing_schedule == 'lin':
        Nsampling = 2000 #linear schedule with flat region in the beginning and end
        Tmax,Tmin = 100,1
        T = np.linspace(Tmax,Tmin,Nsampling)
        T = np.concatenate((Tmax*np.ones((500,)),T),axis=0)
        T = np.concatenate((T,Tmin*np.linspace(1,0,200)),axis=0)
    #sample
    
    if args.sample_mode == 'single':
        filename = args.file_name + str(args.net_indx) + '.pt'
        netE.load_state_dict(torch.load(root + '/models/'+filename))
        
        n_batches = int(np.ceil(args.n_samples_save/args.batch_size))
        
        denoise_samples = []
        print('sampling starts')
        for i in range(n_batches):
            initial_x = 0.5+torch.randn_like(sample_x).to(device)
            x_list,E_trace = Annealed_Langevin_E(netE,initial_x,args.sample_step_size,T,100)
            
            x_denoise = SS_denoise(x_list[-1][:].to(device),netE,0.1)
            denoise_samples.append(x_denoise)
            print('batch {}/{} finished'.format((i+1),n_batches))
            
        denoise_samples = torch.cat(denoise_samples,0)
        torch.save(denoise_samples,root + '/samples/' + args.dataset+'_'+str(args.n_samples_save)+'samples.pt')
        
        
    elif args.sample_mode == 'all':
        n_batches = int(np.ceil(256/args.batch_size))
        i = args.net_indx
        while True:
            filename = args.file_name + str(i) + '.pt'
            i += args.save_every
            try:
                netE.load_state_dict(torch.load(root + '/models/'+filename))
            except:
                print(root + '/models/'+filename)
                print('file not found or reached last file')
                break
            
            print('generating samples for '+ filename)
            denoise_samples = []
            for i in range(n_batches):
                initial_x = 0.5+torch.randn_like(sample_x).to(device)
                x_list,E_trace = Annealed_Langevin_E(netE,initial_x,args.sample_step_size,T,100)
                print(str(len(x_list)))
                x_denoise = SS_denoise(x_list[-1].to(device),netE,0.1)
                denoise_samples.append(x_denoise)
                print('batch {}/{} finished'.format((i+1),n_batches))
            denoise_samples = torch.cat(denoise_samples,0)
            save_sample_pdf(denoise_samples[0:256],(16,16),root + '/samples/' + args.dataset +'_256samples_'+str(i)+'knet_denoise.pdf')
            
            
if __name__ == '__main__':
    main()