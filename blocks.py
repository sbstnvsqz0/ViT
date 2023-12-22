import torch
from torch import nn

class Embedding(nn.Module):
    def __init__(self, patch_size, n_patches,in_channels, embedding_size):
        super().__init__()
        #Se usa conv2d (hybrid arquitecture) para representar matriz entrenable que hace transformación lineal, 
        #stride de largo patch_size para que se aplique a cada patch. La dimensión final representa el embedding como tal
        #Al final una capa flatten(2) para aplanar dimensiones y dejar en patches en 2d
        """Entrada: batch_size,3,224,224
            Salida: batch_size,(n_patches+1),embedding_size"""
        self.patches = nn.Sequential(nn.Conv2d(in_channels = in_channels,   
                                               out_channels = embedding_size,
                                               kernel_size= patch_size,
                                               stride = patch_size),
                                               nn.Flatten(2))
        
        #embedding de posición entrenable (se suma a)
        self.position = nn.Parameter(torch.zeros(size=(1,n_patches+1,embedding_size),requires_grad=True), requires_grad = True)
        #class embedding entrenable (se concatena a salida de la conv2d), se inicializa como cero.
        #El 1 de la dimensión 0 luego es cambiado por el tamaño de batch
        self.class_embedding = nn.Parameter(torch.zeros(size = (1,1,embedding_size),requires_grad=True),requires_grad = True)

    def forward(self,x):
        clase = self.class_embedding.expand(x.shape[0],-1,-1)   #Se cambia la dimension 0 para ajustar a tamaño de x luego de salir de convolucion
        x = self.patches(x).permute(0,2,1)  #Se cambian posiciones para que encajen las dimensiones para la concatenación
        x = torch.cat([clase,x],dim=1)
        
        x = self.position + x

        x = nn.Dropout(p = 0.2)(x)  #Se agrega un dropout para mejorar rendimiento y evitar sobreajuste
        return x
    
class Encoder(nn.Module):
    def __init__(self,n_heads,input_dim,ff_dim,dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        
        self.self_attn = nn.MultiheadAttention(input_dim,n_heads,dropout=dropout,batch_first=True)

        self.mlp = nn.Sequential(nn.Linear(input_dim,ff_dim),
                                         nn.Linear(ff_dim,input_dim),
                                         nn.GELU())
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        r1 = x
        x = self.norm(x)
        attn_output, _ = self.self_attn(x,x,x)
        r2 = r1+attn_output
        x = self.norm(r2)
        x = self.mlp(x)
        x = self.dropout(x)
        x = x + r2
        return x
    
class ViT(nn.Module):
    def __init__(self,patch_size,n_patches,embedding,n_encoders,n_heads,hidden_dim,in_channels,n_classes):
        super().__init__()
        self.embedding_block = Embedding(patch_size,n_patches,in_channels,embedding)
        self.encoder_block = Encoder(n_heads, embedding,hidden_dim)
        self.mlp_head = nn.Sequential(nn.Linear(in_features = embedding, out_features=hidden_dim),
                                      nn.Linear(in_features=hidden_dim,out_features=n_classes),
                                      nn.Tanh())
        self.n_encoders = n_encoders
    def forward(self,x):
        x = self.embedding_block(x)
        for _ in range(self.n_encoders):
            x = self.encoder_block(x)
        x = self.mlp_head(x[:,0,:])
        return x
    
class MLP_fine_tunning(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(in_features=in_channels, out_features=out_channels),
                                    nn.Tanh())
        
    def forward(self,x):
        x = x[:,0,:]
        x = self.linear(x)
        return x