#utility functions used in analysis
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from torchvision.utils import save_image, make_grid



def NN_search(samples,data,N_nearest,device):
    #sample and data all torch tensors in torch convention
    #output NN_index is N_samples*N_nearest
    samples = samples.cuda(device=device)
    data = data.cuda(device=device)
    
    N_samples = samples.shape[0]
    N_data = data.shape[0]
    
    NN_vect_list = []
    for i in range(N_samples):
        sample_expand = samples[i].expand((N_data,-1,-1,-1))
        dist_vect = ((sample_expand-data)**2).sum((1,2,3))
        NN_vect_list.append(torch.argsort(dist_vect)[:N_nearest])
    
    NN_index = torch.stack(NN_vect_list,dim=0)
    
    
    
    return NN_index


def save_sample_pdf(samples,grid_size,filename,pad=2):
    #samples should be a torch tensor
    [Nsamples,Nchannels,Size_x,Size_y] = samples.shape
    
    
    last_step_denoise = make_grid(samples,normalize=False,nrow=grid_size[1],padding=pad).detach().cpu().numpy()
    last_step_denoise = np.moveaxis(last_step_denoise,0,2)
    
    n = colors.Normalize(vmin=0,vmax=1,clip=True)
    normalized=n(last_step_denoise)
    n_pix_column = grid_size[0]*Size_y + (grid_size[0]+1)*pad
    n_pix_row = grid_size[1]*Size_x + (grid_size[1]+1)*pad
    fig = plt.figure(figsize=(n_pix_row/100,n_pix_column/100 ), dpi=100)
    fig.figimage(normalized,xo=0,yo=0,vmin =0,vmax=1)
    fig.savefig(filename)

    
def get_inception_mean_cov(inception,data_in,gpu):
    import torch.nn.functional as F
    
    max_batch = 100
    N_batch = int(np.floor(data_in.shape[0]/max_batch))
    I_mean = torch.Tensor([0.5, 0.5, 0.5]).view((1,-1,1,1)).cuda(device=gpu)
    I_std = torch.Tensor([0.5, 0.5, 0.5]).view((1,-1,1,1)).cuda(device=gpu)
    
    
    act = [] #activation
    for i in range(N_batch):
        data = data_in[i*max_batch:(i+1)*max_batch]
        data = I_std*F.layer_norm(data,(32,32)) + I_mean
        act.append(inception(data)[0].cpu().squeeze().numpy())
    
    act = np.concatenate(act,0)
    mean = act.mean(axis=0)
    center_act = act-mean
    cov = center_act.T@center_act/act.shape[0]
    
    return mean, cov
    
def get_inception_logits(inception,data_in,gpu):
    import torch.nn.functional as F
    
    max_batch = 50
    N_batch = int(np.floor(data_in.shape[0]/max_batch))
    
    #I_mean = torch.Tensor([0.485, 0.456, 0.406]).view((1,-1,1,1)).cuda(device=gpu)
    #I_std = torch.Tensor([0.229, 0.224, 0.225]).view((1,-1,1,1)).cuda(device=gpu)
    I_mean = torch.Tensor([0.5, 0.5, 0.5]).view((1,-1,1,1)).cuda(device=gpu)
    I_std = torch.Tensor([0.5, 0.5, 0.5]).view((1,-1,1,1)).cuda(device=gpu)
    
    act = [] #activation
    for i in range(N_batch):
        data = data_in[i*max_batch:(i+1)*max_batch]
        data = I_std*F.layer_norm(data,(32,32)) + I_mean
        expanded_data = F.interpolate(data,
                              size=(299, 299),
                              mode='bilinear',
                              align_corners=False)
        #expanded_data = 2*expanded_data - 1
        act.append(inception(expanded_data.clamp(min=-1,max=1)).detach().cpu().squeeze().numpy())
    
    return np.concatenate(act,0)



def FID(mean1,mean2,cov1,cov2):
    #calculate FID
    from scipy import linalg
    
    out = ((mean1-mean2)**2).sum() + np.trace(cov1+cov2-2*linalg.sqrtm(cov1@cov2))
    
    return np.real(out)
    
def IS(logits):
    import torch.nn.functional as F
    
    logpmtx = F.log_softmax(logits,dim=1)
    pmtx = F.softmax(logits,dim=1)
    marginal_p = pmtx.mean(0)
    
    KL = F.kl_div(logpmtx,pmtx.expand(logits.shape[0],-1),reduction='mean')
    
    return KL.exp()
    
    
    
    
    
    
    