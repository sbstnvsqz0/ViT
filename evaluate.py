import torch
from tqdm import tqdm

@torch.inference_mode()
def evaluate(net,dataloader,device,criterion,n_val):
    net.eval()
    num_val_batches=len(dataloader)
    loss = 0
    acc = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, total=num_val_batches, desc = "Validation", unit = "batch",leave=False):
            img = batch["image"].float().to(device=device)
            label = batch["label"].type(torch.uint8).to(device=device)

            pred = net(img)
            
            label_pred = torch.argmax(pred,dim=1).float()

            loss += criterion(pred,label).item()
            acc += (label_pred==label).sum()
        epoch_loss = loss/num_val_batches
        epoch_acc = acc.item()/n_val
    net.train()
    return epoch_loss, epoch_acc
