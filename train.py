from pathlib import Path
import os
import sys
import shutil
import numpy as np
import time
import datetime
import torch
import torch.nn as nn
import torchvision
import cfg
from datetime import datetime

from torchvision.utils import save_image, make_grid
from tensorboardX import SummaryWriter

from models.ResNet import Res12_Quadratic, Res18_Quadratic, Res34_Quadratic
from models.SE_ResNet import SE_Res18_Quadratic, Swish

import pdb

def main():
    
    args = cfg.parse_args()
    torch.cuda.manual_seed(args.rand_seed)
    
    
    #switch datasets and models
    
    if args.dataset == 'cifar':
        from data.cifar import inf_train_gen
        itr = inf_train_gen(args.batch_size,flip=False)
        #netE = Res18_Quadratic(3,args.n_chan,32,normalize=False,AF=nn.ELU())
        netE = SE_Res18_Quadratic(3,args.n_chan,32,normalize=False,AF=Swish())
        
        
    elif args.dataset == 'mnist':
        from data.mnist_32 import inf_train_gen
        itr = inf_train_gen(args.batch_size)
        netE = Res12_Quadratic(1,args.n_chan,32,normalize=False,AF=nn.ELU())
        
    elif args.dataset == 'fmnist':
        #print(dataset+str(args.n_chan))
        from data.fashion_mnist_32 import inf_train_gen
        itr = inf_train_gen(args.batch_size)
        netE = Res12_Quadratic(1,args.n_chan,32,normalize=False,AF=nn.ELU())
        
    else:
        NotImplementedError('{} unknown dataset'.format(args.dataset))
        
    
    #setup gpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    netE = netE.to(device)
    if args.n_gpus >1:
        netE = nn.DataParallel(netE)
    
    #setup path
    
    now = datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    #pdb.set_trace()
    print(str(args.cont))
    #print(str(args.time))
    if args.cont==True:
        root = 'logs/' + args.log + '_'+ args.time #compose string for loading
        #load network
        file_name = 'netE_' + str(args.net_indx) + '.pt'
        netE.load_state_dict(torch.load(root + '/models/' +file_name))
    else: # start new will create logging folder
        root = 'logs/'+ args.log + '_' + timestamp #add timestemp
        #over write if folder already exist, not likely to happen as timestamp is used
        if os.path.isdir(root):
            shutil.rmtree(root)
        os.makedirs(root)
        os.makedirs(root+'/models')
        os.makedirs(root+'/samples')
    
    writer = SummaryWriter(root)
    
    # setup optimizer and lr scheduler
    params = {'lr':args.max_lr,'betas':(0.9,0.95)}
    optimizerE = torch.optim.Adam(netE.parameters(),**params)
    if args.lr_schedule == 'exp':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizerE,int(args.n_iter/6))
    
    elif args.lr_schedule == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerE,args.n_iter,eta_min=1e-6,last_epoch=-1)
        
    elif args.lr_schedule == 'const':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizerE,int(args.n_iter))
        
    #train
    print_interval = 50
    max_iter = args.n_iter+args.net_indx
    batchSize = args.batch_size
    sigma0 = 0.1
    sigma02 = sigma0**2
    
    if args.noise_distribution == 'exp':
        sigmas_np = np.logspace(np.log10(args.min_noise),np.log10(args.max_noise),batchSize)
    elif args.noise_distribution == 'lin':
        sigmas_np = np.linspace(args.min_noise,args.max_noise,batchSize)
        
    sigmas = torch.Tensor(sigmas_np).view((batchSize,1,1,1)).to(device)
    
    start_time = time.time()
    
    for i in range(args.net_indx,args.net_indx + args.n_iter):
        x_real = itr.__next__().to(device)
        x_noisy = x_real + sigmas*torch.randn_like(x_real)
        
        x_noisy = x_noisy.requires_grad_()
        E = netE(x_noisy).sum()
        grad_x = torch.autograd.grad(E,x_noisy,create_graph=True)[0]
        x_noisy.detach()
        
        optimizerE.zero_grad()
        
        LS_loss = ((((x_real-x_noisy)/sigmas/sigma02+grad_x/sigmas)**2)/batchSize).sum()
        
        LS_loss.backward()
        optimizerE.step()
        scheduler.step()
        
        if (i+1)%print_interval == 0:
            time_spent = time.time() - start_time
            start_time = time.time()
            netE.eval()
            E_real = netE(x_real).mean()
            E_noise = netE(torch.rand_like(x_real)).mean()
            netE.train()
            
            print('Iteration {}/{} ({:.0f}%), E_real {:e}, E_noise {:e}, Normalized Loss {:e}, time {:4.1f}'.format(i+1,max_iter,100*((i+1)/max_iter),E_real.item(),E_noise.item(),(sigma02**2)*(LS_loss.item()),time_spent))
                  
            writer.add_scalar('E_real',E_real.item(),i+1)
            writer.add_scalar('E_noise',E_noise.item(),i+1)
            writer.add_scalar('loss',(sigma02**2)*LS_loss.item(),i+1)
            del E_real, E_noise, x_real, x_noisy
            
        if (i+1)%args.save_every == 0:
            print("-"*50)
            file_name = args.file_name+str(i+1)+'.pt'
            torch.save(netE.state_dict(),root+'/models/'+file_name)
            
if __name__ == '__main__':
    main()