import torch
import torch.nn as nn

class Dropout(nn.Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p=p
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns a float32 tensor with the same shape as x.
        """
        
        if self.p>1.0 or self.p<0.0:
            raise ValueError("p should BE between 0 and 1")
        if not self.training :
            return x
        if self.p==1.0:
            return torch.zeros_like(x)
        else:
            U=torch.rand_like(x)
            m=torch.where(U>=self.p,1,0).float()
            return x * m / (1 - self.p)

        
