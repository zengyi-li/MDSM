#functions used in training
import torch
import numpy as np

def Langevin(netE,init_x,K,sigma):
    #perform langevin sampling and returns energy value along the way
    #netE : energy model
    #init_x : initial data
    #K : steps to be performed
    #sigma : temperature coefficient for controling the strength of gradient and noise
    
    sigma2 = sigma**2
    
    netE.eval()
    
    x = init_x

    for i in range(K):
        x.requires_grad_()
        E_values = netE(x)
        E = E_values.sum()
        grad = torch.autograd.grad(E,x)[0]
        x = x.detach() - 0.5*sigma2*grad + sigma*torch.randn_like(x)
    
    netE.train()
    
    return x

def Langevin_E(netE,init_x,K,sigma,noise=True):
    #perform langevin sampling and returns energy value along the way
    #netE : energy model
    #init_x : initial data
    #K : steps to be performed
    #sigma : temperature coefficient for controling the strength of gradient and noise
    
    sigma2 = sigma**2
    
    netE.eval()
    
    x = init_x
    
    E_list = []
    
    for i in range(K):
        x.requires_grad_()
        E_values = netE(x)
        E_list.append(E_values.squeeze().detach().cpu().numpy())
        E = E_values.sum()
        E.backward()
        if noise==False:
            x = x.detach() - 0.5*sigma2*x.grad
        else:
            x = x.detach() - 0.5*sigma2*x.grad + sigma*torch.randn_like(x)
    
    netE.train()
    
    E_mtx = np.vstack(E_list)
    return x.detach(), E_mtx

def Annealed_Langevin_E(netE,init_x,sigma,T_vect,Sample_every):
    #naive implementation of annealed Langevin sampling
    #T_vect is vector of temperatures, length equal to total sampling step
    #Sample_every is interval between saving a sample of X
    
    sigma2 = sigma**2
    
    T_vect = np.sqrt(T_vect)
    
    K = T_vect.shape[0]
    
    netE.eval()
    
    E_list = []
    
    x = init_x
    
    x_list = []
    for i in range(K):
        x.requires_grad_()
        E_values = netE(x)
        E_list.append(E_values.squeeze().detach())
        E = E_values.sum()
        E.backward()
        x = x.detach() - 0.5*sigma2*x.grad + T_vect[i]*sigma*torch.randn_like(x)
        
        if (i+1)%Sample_every==0:
            x_list.append(x.detach().cpu())
            print('langevin step {}'.format(i+1))
        
    netE.train()
    
    E_mtx = torch.stack(E_list,1)
    #print('sampling finished')
    return x_list, E_mtx.cpu().numpy()


def Annealed_Langevin_E_mask(netE,init_x,x_mask,sigma,T_vect,Sample_every,gpu):
    #naive implementation of annealed Langevin sampling
    #with step size scaling with energy, or temperature
    #T_vect is vector of temperatures, length equal to total sampling step
    #Sample_every is interval between saving a sample of X
    
    sigma2 = sigma**2
    
    T_vect = np.sqrt(T_vect)
    
    K = T_vect.shape[0]
    
    netE.eval()
    
    E_list = []
    
    x = init_x
    
    x_list = []
    for i in range(K):
        x.requires_grad_()
        E_values = netE(x+0.1*np.sqrt(T_vect[i])*torch.randn_like(x)*(1-x_mask).cuda(gpu))
        E_list.append(E_values.squeeze().detach().cpu().numpy())
        E = E_values.sum()
        E.backward()
        x = x.detach() - 0.5*sigma2*x.grad*x_mask + T_vect[i]*sigma*torch.randn_like(x)*x_mask
        
        if (i+1)%Sample_every==0:
            x_list.append(x.detach().cpu())
        
    netE.train()
    
    E_mtx = np.vstack(E_list)
    return x_list, E_mtx

def SS_denoise(x_noisy,netE,sigma):
    #single step denoising
    
    x_noisy = x_noisy.requires_grad_();
    E  = netE(x_noisy).sum()
    grad_x = torch.autograd.grad(E,x_noisy)[0]
    x_denoised = x_noisy.detach()-(sigma**2)*grad_x
    
    return x_denoised.detach()
    
    
    
    
    
    
class HMC(object):
    #Original HMC implemetation
    def __init__(self,netE,epsi,L,sigma):
        self.netE = netE 
        self.epsi = epsi #step size in Leaf frog integration
        self.L = L #step number in Leaf frog integration
        self.sigma = sigma #variance of momentum gaussian distribution
        self.sigma2= sigma**2 #here sigma2 is equivalent to m in original HMC
        
    def Sample(self,init_x,Nsteps,T_vect):
        #Sample with HMC initialized at init_x for Nsteps 
        #T_vect is Nsteps long, specifying temperature at each step

        self.netE.eval()
        
        self.E_list = []
        self.Ek_list = []
        self.x_prop_list = []
        self.p_prop_list = []
        self.accept_list = []
        
        x = init_x
        p = self.sigma*torch.randn_like(x)
        U_last = self.netE(x)+self.K(p)
        
        for i in range(Nsteps):
            x_prop,p_prop = self.Leap_frog(x,p)
            
            self.x_prop_list.append(x_prop)
            self.p_prop_list.append(p_prop)
            
            U = self.netE(x_prop)
            self.E_list.append(U.squeeze().detach())
            
            Ek = self.K(p_prop)
            self.Ek_list.append(Ek)
            
            U_now = U + Ek #overall energy
            
            p_accept = torch.exp(U_last-U_now)
            accept_vect = torch.rand_like(p_accept)<p_accept
            
            self.accept_list.append(accept_vect)
            
            x = x.detach()
            
            x[accept_vect]= x_prop[accept_vect]
            p[accept_vect]= self.sigma*torch.randn_like(p[accept_vect])
            
            
            
            U_last = U_now
          
        
    def Leap_frog(self,x,p):
        #perform leaf-frog integration for L steps
        for i in range(self.L):
            x.requires_grad_()
            E = self.netE(x).sum()
            grad_x = torch.autograd.grad(E,x)[0]
            
            if i == 0:
                p += -(self.epsi/2)*grad_x
            else:
                p += -self.epsi*grad_x
                
            
            x = x.detach() + self.epsi*p/self.sigma2
            
        return x.detach(), p
    
    def K(self,p):
        #kinetic energy
        return (p**2).sum((1,2,3))/(2*self.sigma2)
    
    
class AIS_sampling(object):
    
    #Use AIS sampling to estimate logZ, log of model partition function
    #Reference model is Gaussian distribution 
    def __init__(self,netE,shape,beta_vect,sigma,epsi,m,L,T,save_every,gpu):
    # initialization
        self.x_shape = shape #tuple of shape for x
        self.sigma = sigma #std of reference distribution
        self.sigma2= sigma**2
        self.epsi = epsi #step length of Langevin dynamics
        self.epsi2 = epsi**2
        self.m = m #mass in HMC simulation
        self.L = L #Langevin steps between energy value estimation
        
        self.gpu = gpu
        self.T = T #temperature on Energy function to evaluate at
        
        self.save_every = save_every #number of Langevin steps per beta
        self.netE = netE
        self.netE.eval()
        self.beta_vect = beta_vect # from 0 to 1 in some schedule
   
    def sample(self):
        #perform AIS sampling
        x = self.sigma*torch.randn(self.x_shape).cuda(device=self.gpu)
        
        x_list = []
        
        logw = 0
        
        last_beta = 0
        
        i = 0
        
        
        for beta in self.beta_vect[1:]:
            
            
           
            E_values = self.Et(x,beta)
            
            #update w with energy values
            logw +=  self.Et(x,last_beta).detach() - E_values.detach()
            
            #use langevin proposal
            #x = self.Langevin(x,beta)
            
            #use HMC proposal
            x = self.HMC(x,beta)
                
            
            last_beta = beta
            i+=1
            
            if i%self.save_every == 0:
                x_list.append(x.detach().cpu()) 
                print(str(i))
            
            
        
        return x_list, logw
    
    def E0(self,x):
        # default energy function of reference model is Gaussian
        return (x**2).sum((1,2,3))/(2*self.sigma2)
    
    def Et(self,x,beta):
        #total energy
        return beta*(self.netE(x)/self.T)+(1-beta)*self.E0(x) 
    
    def Langevin(self,x,beta):
        for i in range(self.L):
            x.requires_grad_()
            
            E_values = self.Et(x,beta)
            
            E = E_values.sum()
            E.backward()
         
            x = x.detach() - 0.5*self.epsi2*x.grad + self.epsi*torch.randn_like(x)
        
        return x
    
    def HMC(self,x_init,beta):
        
        x = x_init.clone()
        
        p = torch.randn_like(x)
        
        U_start = (p**2).sum((1,2,3))/(2*self.m) + self.Et(x,beta).detach()
        
        
        for i in range(self.L):
            x.requires_grad_()
            E = self.Et(x,beta).sum()
            grad_x = torch.autograd.grad(E,x)[0]
            
            if i == 0:
                p += -(self.epsi/2)*grad_x
            else:
                p += -self.epsi*grad_x
                
            
            x = x.detach() + self.epsi*p/self.m
            
        U_end = (p**2).sum((1,2,3))/(2*self.m) + self.Et(x,beta).detach()
        
        p_accept = torch.exp(U_start-U_end)
        accept_vect = torch.rand_like(p_accept)<p_accept
        
        x_init[accept_vect] = x[accept_vect]
        
        return x_init.detach()
        
    
    
    
    
class Reverse_AIS_sampling(object):
    #Use AIS sampling to estimate logZ, log of model partition function
    #Reference model is Gaussian distribution 
    def __init__(self,netE,init_x,beta_vect,sigma,x_offset,epsi,T,save_every,gpu):
    # initialization
        self.init_x = init_x
        self.x_shape = init_x.shape #tuple of shape for x
        
        self.sigma = sigma #std of reference distribution
        self.sigma2= sigma**2
        self.x_offset = x_offset #offset of reference distirbution
        
        self.epsi = epsi #step length of Langevin dynamics
        self.epsi2 = epsi**2
        self.gpu = gpu
        self.T = T #temperature on Energy function to evaluate at
        
        self.save_every = save_every #number of Langevin steps per beta
        self.netE = netE
        self.netE.eval()
        self.beta_vect = beta_vect # from 1 to 0 in some schedule
   
    def sample(self):
        #perform AIS sampling
        x = self.init_x
        
        x_list = []
        
        logw = 0
        
        last_beta = 1
        
        i = 0

        for beta in self.beta_vect[1:]:
            
            
            x.requires_grad_()
            E_values = self.Et(x,beta)
            
            #update w with energy values
            logw +=  self.Et(x,last_beta).detach() - E_values.detach()
            
            
            E = E_values.sum()
            E.backward()
            x = x.detach() - 0.5*self.epsi2*x.grad + self.epsi*torch.randn_like(x).cuda(device=self.gpu)
            
            
            
            last_beta = beta
            i+=1
            
            if (i+2)%self.save_every == 0:
                x_list.append(x.detach().cpu()) 
                print(str(i))
            
        
        return x_list, logw
    
    def E0(self,x):
        # default energy function of reference model is Gaussian
        return ((x-self.x_offset)**2).sum((1,2,3))/(2*self.sigma2)
    
    def Et(self,x,beta):
        #total energy
        return beta*(self.netE(x)/self.T)+(1-beta)*self.E0(x)     
    
    
    
    