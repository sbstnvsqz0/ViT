from blocks import ViT, MLP_fine_tunning
from Yoga82 import Yoga
import torch
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.transforms import v2
import cv2
import matplotlib.pyplot as plt

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


def predict(path_images,path_dir_model,d,transfer=False):
    f = open(path_dir_model+"/params.txt")
    lines = f.readlines()
    f.close()
    info = dict(eval(lines[0]))

    dataset = Yoga(path_images,d,info["n_classes"])
    loader = DataLoader(dataset, batch_size=32)
    model = ViT(patch_size = info["patch_size"],
                 n_patches = info["n_patches"],
                 embedding = info["embedding"],
                 n_encoders = info["n_encoders"],
                 n_heads = info["n_heads"],
                 hidden_dim = info["hidden_dim"],
                 in_channels = info["in_channels"],
                 n_classes = info["n_classes"])
    if transfer:
        model.mlp_head = MLP_fine_tunning(info["embedding"],info["n_classes"])
    model.load_state_dict(torch.load(path_dir_model +"/checkpoint.pth"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    original_labels = []
    preds_labels = []
    model.to(device=device)
    model.eval()
    with torch.no_grad():
        for batch in loader:
            img = batch["image"].float().to(device=device)
            labels = batch["label"].type(torch.uint8).to(device=device)
            original_labels+=list(labels.cpu().numpy())
            pred = model(img)
            label_pred = torch.argmax(pred, dim = 1).float()
            preds_labels+=list(label_pred.cpu().numpy())
            
    return original_labels,preds_labels

def predict_img(path_image,path_dir_model):
    f = open(path_dir_model+"/params.txt")
    lines = f.readlines()
    f.close()
    info = dict(eval(lines[0]))
    class_name = path_image.split("/")[-2] 
    number_classes = info["n_classes"]
    index = 0 if  number_classes ==6 else 1 if number_classes==20 else 2
    original_label = d[class_name][index]

    transforms = v2.Compose([v2.Resize([128,128]), 
                                      v2.ToImage(),
                                      v2.ToDtype(torch.float32,scale=True),
                                      v2.Normalize(mean = [0.5,0.5,0.5],std=[0.5,0.5,0.5])])

    model = ViT(patch_size = info["patch_size"],
                 n_patches = info["n_patches"],
                 embedding = info["embedding"],
                 n_encoders = info["n_encoders"],
                 n_heads = info["n_heads"],
                 hidden_dim = info["hidden_dim"],
                 in_channels = info["in_channels"],
                 n_classes = info["n_classes"])
    if path_dir_model[-1]!=0:
        model.mlp_head = MLP_fine_tunning(info["embedding"],info["n_classes"])
    model.load_state_dict(torch.load(path_dir_model +"/checkpoint.pth"))

    device = "cpu"

    image = Image.open(path_image)
    image = image.convert("RGB")
    
    image = transforms(image)
    image = image.unsqueeze(0)

    pred = model(image)
    label_pred = torch.argmax(pred,dim=1).type(torch.uint8).item()

    image = torch.squeeze(image)
    image = image.permute(1,2,0)
    image = image.numpy()
    image = image[...,::-1]
    
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Clase original: {}; Clase predecida: {}".format(str(original_label),str(label_pred)))
    plt.axis("off")
    plt.show()