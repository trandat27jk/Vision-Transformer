import torch
from torch import nn
from einops.layers.torch import Rearrange
from einops import rearrange,repeat


pair= lambda x:x if isinstance(x,tuple) else (x,x)

class FeedForward(nn.Module):
    def __init__(self, dim, expansion_factor,dropout=0.):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,dim*expansion_factor),
                               nn.GELU(),
                               nn.Dropout(dropout),
                               nn.Linear(dim*expansion_factor,dim),
                               nn.Dropout(dropout))
    def forward(self,x):
        x1=self.norm(x)
        return self.mlp(x1)+x

class Attention(nn.Module):
    def __init__(self, dim,heads):
        super().__init__()
        self.norm=nn.LayerNorm(dim)
        self.to_qkv=nn.Linear(dim,dim*3)
        self.softmax=torch.nn.Softmax(dim=-1)
        self.scale=dim**(-0.5)
        self.heads=heads
    def forward(self,x):
        x1=self.norm(x)
        qkv=self.to_qkv(x1).chunk(3,dim=-1)
        q,k,v=map(lambda t: rearrange(t,'b n (h d) -> b h n d',h=self.heads),qkv)
        dots=torch.matmul(q,k.transpose(-1,-2))*self.scale
        atten=self.softmax(dots)
        out=torch.matmul(atten,v)
        out=rearrange(out,'b h n d -> b n (h d)')
        return x+out

class Transformer(nn.Module):
    def __init__(self, depth,dim,heads,expansion_factor,dropout):
        super().__init__()
        self.layers = nn.ModuleList([Attention(dim, heads), FeedForward(dim, expansion_factor, dropout)] for _ in range(depth))

        
    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x


#Vit
class Vit(nn.Module):
    def __init__(self,image_size,patch_size,channels,dim,depth,classes,heads,expansion_factor,dropout=0.,emb_dropout=0.,pool='cls'):
        super().__init__()
        image_height,image_width=pair(image_size)
        patch_height,patch_width=pair(patch_size)
        
        num_patchs=(image_height//patch_height)*(image_width//patch_width)
        patch_dim=(image_height*image_width*channels)
        self.to_patch_embedding=(Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',p1=patch_height,p2=patch_width),
                                 nn.LayerNorm(patch_dim)
                                 ,nn.Linear(patch_dim,dim))
        self.positional_embedding=nn.Parameter(torch.randn(1,num_patchs+1,dim))
        self.cls_token=nn.Parameter(torch.randn(1,1,dim))
        self.emb_dropout=nn.Dropout(emb_dropout)
        self.pool=pool
        self.transformer=Transformer(depth,dim,heads,expansion_factor,dropout)
        self.to_latent=nn.Identity()
        self.mlp=nn.Linear(dim,classes)
    def forward(self,x):
        x=self.to_patch_embedding(x)
        b,n,_=x.shape
        cls_token=repeat(self.cls_token,'1 1 d -> b 1 d')
        x=torch.cat((cls_token,x),dim=1)
        x+=self.positional_embedding[:,:(n+1)]
        x=self.emb_dropout(x)
        x=self.transformer(x)
        x=x.mean(dim=1) if self.pool=='mean' else x[:,0]
        x=self.to_latent(x)
        return self.mlp(x)
        