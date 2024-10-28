import torch

    
class ResConv(torch.nn.Module):
        def __init__(self, c):
            super().__init__()
            self.conv = torch.nn.Conv2d(c, c, 3, 1, 1)
            self.beta = torch.nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)        
            self.relu = torch.nn.Mish(True)

        def forward(self, x):
            return self.relu(self.conv(x) * self.beta + x)
    

class myNet(torch.nn.Module):
    def __init__(self, c=96):
        super().__init__()
        self.encode = torch.nn.Sequential(
            torch.nn.Conv2d(3, c//2, 3, 2, 1), 
            torch.nn.Mish(True),
            torch.nn.Conv2d(c//2, c, 3, 2, 1),
            torch.nn.Mish(True),
        )
        self.convblock = torch.nn.Sequential(
             ResConv(c),
             ResConv(c),
             ResConv(c),
             ResConv(c),
             ResConv(c),
             ResConv(c),
             ResConv(c),
             ResConv(c),
        )
        self.lastconv = torch.nn.Sequential(
             torch.nn.ConvTranspose2d(c, c//2, 4, 2, 1),
             torch.nn.Mish(True),
             torch.nn.ConvTranspose2d(c//2, 1, 4, 2, 1),
        )

    def forward (self, x):
         x = self.encode(x)
         x = self.convblock(x)
         x = self.lastconv(x)
         return x
