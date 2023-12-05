from tqdm.notebook import tqdm
import torch
from IPython.display import clear_output

def train_one_epoch(model, train_loader, epoch, device, optimizer, criterion):
    model.train()
    epoch_loss = 0

    with tqdm(train_loader, total=len(train_loader), leave=False) as t:
        for batch in t:
            t.set_description(f'Epoch {epoch+1}')
            g = batch.to(device)

            h = g.x[:,0].to(torch.long)

            precomputed_eigenvectors = g.x[:,1:]
            sign_flip = torch.rand(precomputed_eigenvectors.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            precomputed_eigenvectors = precomputed_eigenvectors * sign_flip.unsqueeze(0)

            y = g.y

            optimizer.zero_grad()
            out = model(g, h, precomputed_eigenvectors=precomputed_eigenvectors)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()
    
    epoch_loss = epoch_loss/len(train_loader)
    clear_output(wait=True)
    return epoch_loss, optimizer

def evaluate(model, val_loader, device, criterion):
    model.eval()
    val_loss = 0

    with tqdm(val_loader, total=len(val_loader), leave=False) as t:
        for batch in t:
            t.set_description('Validation')
            g = batch.to(device)

            h = g.x[:,0].to(torch.long)

            precomputed_eigenvectors = g.x[:,1:]
            sign_flip = torch.rand(precomputed_eigenvectors.size(1)).to(device)
            sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
            precomputed_eigenvectors = precomputed_eigenvectors * sign_flip.unsqueeze(0)

            y = g.y

            with torch.no_grad():
                out = model(g, h, precomputed_eigenvectors=precomputed_eigenvectors)
                loss = criterion(out, y)

            val_loss += loss.detach().item()

    val_loss = val_loss/len(val_loader)
    clear_output(wait=True)
    return val_loss
