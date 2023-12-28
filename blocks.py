import torch
from torch import nn,matmul
from torch.nn.functional import softmax

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
        self.position = nn.Parameter(torch.rand(size=(1,n_patches+1,embedding_size)), requires_grad = True)
        #class embedding entrenable (se concatena a salida de la conv2d), se inicializa como cero.
        #El 1 de la dimensión 0 luego es cambiado por el tamaño de batch
        self.class_embedding = nn.Parameter(torch.rand(size = (1,1,embedding_size)),requires_grad = True)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self,x):
        clase = self.class_embedding.expand(x.shape[0],-1,-1)   #Se cambia la dimension 0 para ajustar a tamaño de x luego de salir de convolucion
        x = self.patches(x).permute(0,2,1)  #Se cambian posiciones para que encajen las dimensiones para la concatenación
        x = torch.cat([clase,x],dim=1)
        
        x = self.position + x

        x = self.dropout(x)  #Se agrega un dropout para mejorar rendimiento y evitar sobreajuste
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self,embedding,n_heads,heads_size):
        super().__init__()

        self.n_heads = n_heads  #Número de cabezas
        self.heads_size = heads_size    #Tamaño de las cabezas
        self.embedding = embedding  #largo de los embeddings
        assert self.embedding % self.n_heads == 0,"Embedding must be divisible by number of heads."

        #Para representar los pesos de matrices q, k y v se arman linear layer
        self.linear_q = nn.Linear(embedding,n_heads*heads_size, bias=True)

        self.linear_k = nn.Linear(embedding, n_heads*heads_size, bias=True)

        self.linear_v = nn.Linear(embedding, n_heads*heads_size, bias =True)

        self.final_layer = nn.Linear(n_heads*heads_size, embedding, bias=True)

    def forward(self, x):
        batch_size, seq, embedding = x.shape
        
        #Separa la dimensión final en 2 para expresar matrices
        #Luego se usa transpose para que para cada cabeza tenga su correspondiente matriz Q de dimensiones seq,heads_size
        q = self.linear_q(x).view(batch_size, seq, self.n_heads,self.heads_size).transpose(1,2)

        k = self.linear_k(x).view(batch_size, seq, self.n_heads,self.heads_size).transpose(1,2)

        v = self.linear_v(x).view(batch_size, seq, self.n_heads,self.heads_size).transpose(1,2)

        ######################################## SCALED DOT PRODUCT ATTENTION ####################################################

        attention = q/(self.heads_size**(1/2)) #Paper Attention all you need
        
        #Matriz K de cada (batch,head) se traspone
        attention = matmul(attention,k.transpose(-1,-2))
        attention = softmax(attention,dim=-1)
        attention = matmul(attention,v)
        #Aquí se tienen las matrices de attention para todos los batch,head

        #Se hace de nuevo un reshape y se unen las cabezas 
        output = attention.transpose(1,2).contiguous().view(batch_size,seq,self.n_heads*self.heads_size)
        output = self.final_layer(output)
        return output
    
class Encoder(nn.Module):
    def __init__(self,n_heads,input_dim,ff_dim,dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        
        self.self_attn = MultiHeadAttention(input_dim,n_heads,heads_size=64) #Head_size calculado como embedding/n_heads:768/8

        self.mlp = nn.Sequential(nn.Linear(input_dim,ff_dim),
                                 nn.GELU(),
                                 nn.Dropout(dropout),
                                nn.Linear(ff_dim,input_dim),
                                nn.Dropout(dropout))
                                         
        
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        r1 = x
        x = self.norm(x)
        x = self.self_attn(x)
        r2 = r1+x
        x = self.norm(r2)
        x = self.mlp(x)
        x = x + r2
        return x



class ViT(nn.Module):
    def __init__(self,patch_size,n_patches,embedding,n_encoders,n_heads,hidden_dim,in_channels,n_classes):
        super(ViT,self).__init__()
        self.embedding_block = Embedding(patch_size,n_patches,in_channels,embedding)
        
        self.encoder_blocks = nn.ModuleList([Encoder(n_heads, embedding, hidden_dim) for _ in range(n_encoders)])

        self.mlp_head = nn.Sequential(nn.Linear(in_features = embedding, out_features=512),
                                      nn.ReLU(),
                                      nn.Linear(in_features=512,out_features=n_classes),
                                      nn.Tanh())
        self.n_encoders = n_encoders
    def forward(self,x):
        x = self.embedding_block(x)
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
        x = self.mlp_head(x[:,0,:])
        return x
    

class MLP_fine_tunning(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.linear = nn.Sequential(nn.Linear(in_features=in_channels, out_features=out_channels),
                                    nn.Tanh())
        
    def forward(self,x):
        x = self.linear(x)
        return x