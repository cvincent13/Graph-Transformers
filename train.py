from tqdm.notebook import tqdm
import torch
from IPython.display import clear_output

def train_one_epoch(model, train_loader, epoch, device, optimizer, criterion, pos_encoding='laplacian', accuracy=False):
    model.train()
    epoch_loss = 0
    if accuracy:
        epoch_acc = 0

    with tqdm(train_loader, total=len(train_loader), leave=False) as t:
        for batch in t:
            t.set_description(f'Epoch {epoch+1}')
            g = batch.to(device)

            h = g.x.to(torch.long)

            if pos_encoding == 'laplacian':
                precomputed_eigenvectors = g.laplacian_eigs
                sign_flip = torch.rand(precomputed_eigenvectors.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                precomputed_eigenvectors = precomputed_eigenvectors * sign_flip.unsqueeze(0)
            elif pos_encoding == 'wl':
                precomputed_eigenvectors = g.wl_encoding
            elif pos_encoding == 'both':
                precomputed_eigenvectors = (g.laplacian_eigs, g.wl_encoding)
            else:
                precomputed_eigenvectors = None

            y = g.y

            optimizer.zero_grad()
            out, _ = model(g, h, precomputed_eigenvectors=precomputed_eigenvectors)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.detach().item()

            if accuracy:
                pred = out.argmax(dim=-1)
                acc = (pred==y).sum()/len(y)
                epoch_acc += acc.detach().item()
    
    epoch_loss = epoch_loss/len(train_loader)
    if accuracy:
        epoch_acc = epoch_acc/len(train_loader)
    clear_output(wait=True)
    if accuracy:
        return epoch_loss, epoch_acc, optimizer
    return epoch_loss, optimizer

def evaluate(model, val_loader, device, criterion, pos_encoding='laplacian', accuracy=False):
    model.eval()
    val_loss = 0
    if accuracy:
        val_acc = 0

    with tqdm(val_loader, total=len(val_loader), leave=False) as t:
        for batch in t:
            t.set_description('Validation')
            g = batch.to(device)

            h = g.x.to(torch.long)

            if pos_encoding == 'laplacian':
                precomputed_eigenvectors = g.laplacian_eigs
                sign_flip = torch.rand(precomputed_eigenvectors.size(1)).to(device)
                sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
                precomputed_eigenvectors = precomputed_eigenvectors * sign_flip.unsqueeze(0)
            elif pos_encoding == 'wl':
                precomputed_eigenvectors = g.wl_encoding
            elif pos_encoding == 'both':
                precomputed_eigenvectors = (g.laplacian_eigs, g.wl_encoding)
            else:
                precomputed_eigenvectors = None

            y = g.y

            with torch.no_grad():
                out, _ = model(g, h, precomputed_eigenvectors=precomputed_eigenvectors)
                loss = criterion(out, y)

                val_loss += loss.detach().item()

                if accuracy:
                    pred = out.argmax(dim=-1)
                    acc = (pred==y).sum()/len(y)
                    val_acc += acc.detach().item()

    val_loss = val_loss/len(val_loader)
    if accuracy:
        val_acc = val_acc/len(val_loader)
    clear_output(wait=True)
    if accuracy:
        return val_loss, val_acc
    return val_loss
