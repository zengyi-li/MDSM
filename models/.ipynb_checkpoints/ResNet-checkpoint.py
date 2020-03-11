from torch import nn
from torch.autograd import grad
import torch


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.2)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.2)
        m.bias.data.fill_(0)

class MyConvo2d(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size,  stride = 1, bias = True):
        super(MyConvo2d, self).__init__()
        self.padding = int((kernel_size - 1)/2)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride=1, padding=self.padding, bias = bias)
        
        
    def forward(self, input):
        output = self.conv(input)
        return output
    
class Square(nn.Module):
    def __init__(self):
        super(Square,self).__init__()
        pass
    
    def forward(self,in_vect):
        return in_vect**2
    
class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
        pass
    
    def forward(self,in_vect):
        return in_vect*nn.functional.sigmoid(in_vect)
    
class MeanPoolConv(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(MeanPoolConv, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, input):
        output = input
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        output = self.conv(output)
        return output
    
class ConvMeanPool(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size):
        super(ConvMeanPool, self).__init__()
        self.conv = MyConvo2d(input_dim, output_dim, kernel_size)

    def forward(self, input):
        output = self.conv(input)
        output = (output[:,:,::2,::2] + output[:,:,1::2,::2] + output[:,:,::2,1::2] + output[:,:,1::2,1::2]) / 4
        return output
    
    
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, hw, resample=None, normalize=False,AF=nn.ELU()):
        super(ResidualBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.kernel_size = kernel_size
        self.resample = resample
        self.normalize = normalize
        self.bn1 = None
        self.bn2 = None
        self.relu1 = AF
        self.relu2 = AF
        if resample == 'down':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])
        elif resample == 'none':
            self.bn1 = nn.LayerNorm([input_dim, hw, hw])
            self.bn2 = nn.LayerNorm([input_dim, hw, hw])

        if resample == 'down':
            self.conv_shortcut = MeanPoolConv(input_dim, output_dim, kernel_size = 1)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = ConvMeanPool(input_dim, output_dim, kernel_size = kernel_size)
        elif resample == 'none':
            self.conv_shortcut = MyConvo2d(input_dim, output_dim, kernel_size = 1)
            self.conv_1 = MyConvo2d(input_dim, input_dim, kernel_size = kernel_size, bias = False)
            self.conv_2 = MyConvo2d(input_dim, output_dim, kernel_size = kernel_size)
            
            
    def forward(self, input):
        if self.input_dim == self.output_dim and self.resample == None:
                shortcut = input
        else:
            shortcut = self.conv_shortcut(input)
        
        if self.normalize == False:
            output = input
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.relu2(output)
            output = self.conv_2(output)
        else:
            output = input
            output = self.bn1(output)
            output = self.relu1(output)
            output = self.conv_1(output)
            output = self.bn2(output)
            output = self.relu2(output)
            output = self.conv_2(output)

        return shortcut + output

    




class Res18_Quadratic(nn.Module):
    #Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res18_Quadratic, self).__init__()
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rbc11 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc22 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.ln1 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.ln2 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.lq = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.Square = Square()
        

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output

class Res18_Linear(nn.Module):
    #Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res18_Linear, self).__init__()
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rbc11 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc22 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.nl = AF
        self.ln = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        
        

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = self.nl(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln(output)
        return output	
	
	
class Res12_Quadratic(nn.Module):
    #6 block resnet used in MIT EBM papaer for conditional Imagenet 32X32
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res12_Quadratic, self).__init__()
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.ln1 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.ln2 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.lq = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.Square = Square()
        
        
    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output        

class Res6_Quadratic(nn.Module):
    #3 block resnet for small MNIST experiment
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res6_Quadratic, self).__init__()

        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)        
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.ln1 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.ln2 = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.lq = nn.Linear(int(hw/8)*int(hw/8)*8*dim, 1)
        self.Square = Square()
        
        
    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)        
        output = self.rb2(output)       
        output = self.rb3(output)
        output = output.view(-1, int(self.hw/8)*int(self.hw/8)*8*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output    
    
    
class Res34_Quadratic(nn.Module):
    #Special super deep network used in MIT EBM paper for unconditional CIFAR 10
    def __init__(self,inchan,dim,hw,normalize=False,AF=None):
        super(Res34_Quadratic, self).__init__()
        #made first layer 2*dim wide to not provide bottleneck
        
        self.hw = hw
        self.dim = dim
        self.inchan = inchan
        self.conv1 = MyConvo2d(inchan,dim, 3)
        self.rb1 = ResidualBlock(dim, 2*dim, 3, int(hw), resample = 'down',normalize=normalize,AF=AF)
        self.rbc1 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rbc11 = ResidualBlock(2*dim, 2*dim, 3, int(hw/2), resample = 'none',normalize=normalize,AF=AF)
        self.rb2 = ResidualBlock(2*dim, 4*dim, 3, int(hw/2), resample = 'down',normalize=normalize,AF=AF)
        self.rbc2 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc22 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rbc222 = ResidualBlock(4*dim, 4*dim, 3, int(hw/4), resample = 'none',normalize=normalize,AF=AF)
        self.rb3 = ResidualBlock(4*dim, 8*dim, 3, int(hw/4), resample = 'down',normalize=normalize,AF=AF)
        self.rbc3 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc333 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc3333 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rbc33333 = ResidualBlock(8*dim, 8*dim, 3, int(hw/8), resample = 'none',normalize=normalize,AF=AF)
        self.rb4 = ResidualBlock(8*dim, 16*dim, 3, int(hw/8), resample = 'down',normalize=normalize,AF=AF)
        self.rbc4 = ResidualBlock(16*dim, 16*dim, 3, int(hw/16), resample = 'none',normalize=normalize,AF=AF)
        self.rbc44 = ResidualBlock(16*dim, 16*dim, 3, int(hw/16), resample = 'none',normalize=normalize,AF=AF)
        
        
        self.ln1 = nn.Linear(int(hw/16)*int(hw/16)*16*dim, 1)
        self.ln2 = nn.Linear(int(hw/16)*int(hw/16)*16*dim, 1)
        self.lq = nn.Linear(int(hw/16)*int(hw/16)*16*dim, 1)
        self.Square = Square()
        

    def forward(self, x_in):
        output = x_in
        output = self.conv1(output)
        output = self.rb1(output)
        output = self.rbc1(output)
        output = self.rbc11(output)
        output = self.rb2(output)
        output = self.rbc2(output)
        output = self.rbc22(output)
        output = self.rbc222(output)
        output = self.rb3(output)
        output = self.rbc3(output)
        output = self.rbc33(output)
        output = self.rbc333(output)
        output = self.rbc3333(output)
        output = self.rbc33333(output)
        output = self.rb4(output)
        output = self.rbc4(output)
        output = self.rbc44(output)
        output = output.view(-1, int(self.hw/16)*int(self.hw/16)*16*self.dim)
        output = self.ln1(output)*self.ln2(output)+self.lq(self.Square(output))
        output = output.view(-1)
        return output    
    
    
    
    
    
    
    
    