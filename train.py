import torch
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import random
from torch import nn
from torch.autograd import Variable
from blocks import ViT
from evaluate import evaluate
from Yoga82 import Yoga
import numpy as np
import os
import argparse
import logging
from torchvision.transforms import v2
import pandas as pd
from utils.scheduler import Scheduler, Scheduler2
from blocks import MLP_fine_tunning
from utils.ContarClase import contar_clase
from torch import nn,matmul
from torch.nn.functional import softmax


f = open("Yoga-82/yoga_train.txt");
lines = f.readlines();
f.close();

d = {}
for line in lines:
    folder_name,numbers = line.split("/")[0], [int(line.split("/")[1].split(",")[1]),int(line.split("/")[1].split(",")[2]),int(line.split("/")[1].split(",")[3].split("\n")[0])]
    if folder_name not in d.keys():
        d[folder_name] = numbers 
    else:
        if d[folder_name] != numbers:
            print("Error")
            break


torch.cuda.empty_cache() 
if (not os.path.isdir("dict_states")):
    os.mkdir("dict_states")

def train(epochs,val_percent,batch_size,patch_size,n_patches, embedding,n_encoders,n_heads, hidden_dim, in_channels,n_classes,learning_rate,save_dict=True,dictionary=None):
    # Se crea .txt con hiperparámetros
    while True:
        num = random.randint(0,100)
        if (not os.path.isdir("dict_states/"+str(num))):
            break
    os.mkdir("dict_states/"+str(num))
    print("Modelo número {}".format(num))
    parameters = dict(epochs=epochs,val_percent=val_percent,batch_size=batch_size,patch_size=patch_size,n_patches=n_patches, embedding=embedding,
                n_encoders=n_encoders,n_heads=n_heads, hidden_dim=hidden_dim, in_channels=in_channels,n_classes=n_classes,learning_rate=learning_rate,
                save_dict=save_dict)
    
    file = open("dict_states/"+str(num)+"/params.txt",'a')
    file.write(str(parameters))
    file.close()
    df_losses = pd.DataFrame(columns=["Train","Validation"])    #Guardar Losses

    device = "cuda" if torch.cuda.is_available() else "cpu"

    criterion = nn.CrossEntropyLoss()
    

    if dictionary is not None:  #Se carga modelo
        assert type(dictionary) == str, "Model must be a string for transfer learning"
        last_n_classes = 20 if n_classes==82 else 6 if n_classes==20 else None
        model = ViT(patch_size, n_patches, embedding, n_encoders, n_heads, hidden_dim, in_channels, last_n_classes)
        try:
            model.load_state_dict(torch.load(dictionary))
        except:
            model.mlp_head = MLP_fine_tunning(embedding,last_n_classes)
            model.load_state_dict(torch.load(dictionary))
        print("Model loaded correctly")
        #Se quita mlp head y se agrega al final una capa linear y una tanh
        model.mlp_head = MLP_fine_tunning(embedding,n_classes)
        #Se congelan todas las capas, excepto Linear final y class_embedding
        for name,param in model.named_parameters():
            if not ("mlp_head" in name):
                param.requires_grad = False
            if "class_embedding" in name:
                param.requires_grad = True
        optimizer = optim.SGD(model.parameters(),momentum =0.9, lr= learning_rate,weight_decay=0.1)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=50, eta_min=1e-7)
        
    else:
        model = ViT(patch_size, n_patches, embedding, n_encoders, n_heads, hidden_dim, in_channels, n_classes)
        optimizer = optim.Adam(model.parameters(),betas =(0.9, 0.999), lr = learning_rate,weight_decay=0.03)
        scheduler = Scheduler(optimizer=optimizer, dim_embed=embedding,warmup_steps=1000)

    model.to(device=device)
    
    dataset = Yoga(images_dir="Images", dictionary=d, n_classes=n_classes)
    dataset.transforms = v2.Compose([v2.Resize([128,128]), 
                                 v2.RandomHorizontalFlip(p=0.2),
                                 v2.RandomRotation(degrees=45),
                                 v2.RandomVerticalFlip(p=0.2),
                                 v2.ToImage(),v2.ToDtype(torch.float32,scale=True),
                                 v2.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    
    n_val = int(len(dataset)*val_percent)
    n_train = len(dataset)-n_val
    train_set, _ = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_set,batch_size=batch_size, shuffle=True)

    dataset0 = Yoga(images_dir="Images", dictionary=d, n_classes=n_classes)
    _, val_set = random_split(dataset0, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    
    best_val_loss = np.inf
    patience = 0

    for epoch in range(0,epochs):
        model.train()
        train_loss = 0
        batch_count = 0
        train_correct = 0

        if patience ==10 and epoch>50:
            break

        with tqdm(total=n_train,desc=f'Epoch {epoch}/{epochs}',unit="img") as pbar:
            for batch in train_loader:
                
                img = batch['image'].float().to(device=device)
                label = batch['label'].type(torch.uint8).to(device=device)
                t_pred = model(img)
                label_pred = torch.argmax(t_pred, dim = 1)
            
                t_loss = criterion(t_pred, label)
                train_loss += t_loss.item()
                batch_count+=1
                t_loss.backward()
                if batch_count%16==0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

                
                train_correct += (label_pred == label).sum()
                pbar.update(img.shape[0])
            if batch_count%16!=0:   #Optimiza luego de 512 imágenes 
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            train_epoch_loss = train_loss / (len(train_loader))
            train_acc = train_correct/ (n_train)
        
        val_epoch_loss,val_acc = evaluate(model,val_loader,device,criterion,n_val)
        
        dict_epoch ={"Train":train_epoch_loss,"Validation":val_epoch_loss}
        df_epoch = pd.DataFrame(dict_epoch,index=[0])
        df_losses = pd.concat([df_losses,df_epoch],ignore_index=True)
        df_losses.to_csv("dict_states/"+str(num)+"/losses.csv",index=False)

        if save_dict and val_epoch_loss<best_val_loss:
            torch.save(model.state_dict(), 'dict_states/{}/checkpoint.pth'.format(str(num)))
            best_val_loss = val_epoch_loss
            patience=0
        
        else:
            patience+=1
        print("Epoch : {},\tTrain Loss : {:.4f}, \tVal Loss : {:.4f},\tTrain Acc : {:.4f},\tVal Acc : {:.4f}".format(epoch,train_epoch_loss,val_epoch_loss,train_acc,val_acc))
    

def get_args():
    parser = argparse.ArgumentParser('Train a network')
    parser.add_argument('--epochs',"-e",metavar="E",type=int,default=150,help="Número de epocas")
    parser.add_argument("--val_percent","-v",type=float, default=0.1,help="Porcentaje del conjunto que se usa para validación")
    parser.add_argument('--batch_size','-b',dest="batch_size",metavar="B",type=int,default=32, help = "Batch Size")
    parser.add_argument("--patch_size","-p",type=int, default=16, help = "Tamaño de los patches")
    parser.add_argument("--n_patches","-n",type = int ,default =64, help = "Número de patches")
    parser.add_argument("--embedding","-emb",type = int ,default =768 , help = "Tamaño de vector de embedding")
    parser.add_argument("--n_encoders","-enc",type = int ,default = 8 , help = "Número de encoders")
    parser.add_argument("--n_heads","-head",type = int ,default = 8  , help = "Número de heads")
    parser.add_argument("--hidden_dim","-hidd",type = int ,default = 2048 , help = "Dimensión de hidden layer")
    parser.add_argument("--in_channels","-i",type = int ,default = 3 , help = "Número de canales de imagen de entrada")
    parser.add_argument("--n_classes","-class",type = int ,default = 6 , help = "Número de clases (6, 20 u 82)")
    parser.add_argument("--lr","-l", metavar="LR", type=float, default=0.001, help="Learning Rate")
    parser.add_argument("--save_dict","-save",type = bool ,default = True , help = "Si es True guarda modelo luego de cada época")
    parser.add_argument("--dictionary","-d",type = str ,default = None , help = "Carga modelo en caso de definirse")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    
    #Cambiar args para entrenamientos específicos
    train(epochs = args.epochs,
          val_percent = args.val_percent,
          batch_size = args.batch_size,
          patch_size = args.patch_size,
          n_patches = args.n_patches,
          embedding = args.embedding,
          n_encoders = args.n_encoders,
          n_heads = args.n_heads,
          hidden_dim = args.hidden_dim,
          in_channels = args.in_channels,
          n_classes = args.n_classes,
          learning_rate = args.lr,
          save_dict = args.save_dict,
          dictionary = args.dictionary
        )