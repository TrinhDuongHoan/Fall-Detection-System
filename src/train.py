import torch
from tqdm.notebook import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, epochs, patience, ckpt_path, device):
    
    best_val_loss = float("inf")
    patience_counter = 0

    print(f"Starting training on device: {device}")
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Train]")
        for batch_X, batch_y in train_pbar:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * batch_X.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == batch_y).sum().item()
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        train_loss /= len(train_loader.dataset)
        train_acc = train_correct / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{epochs} [Val]")
        with torch.no_grad():
            for batch_X, batch_y in val_pbar:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                
        val_loss /= len(val_loader.dataset)
        val_acc = val_correct / len(val_loader.dataset)

        print(f"Epoch {epoch+1:02d}/{epochs} - "
              f"loss: {train_loss:.4f} - acc: {train_acc:.4f} - "
              f"val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [*] val_loss improved to {val_loss:.4f}, model saved.")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}.")
                break
                
    print("Training finished.")
