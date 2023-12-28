from blocks import ViT, MLP_fine_tunning
from Yoga82 import Yoga
import torch
from torch.utils.data import DataLoader


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