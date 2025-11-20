import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

from model import LSTMAttentionModel, SimpleLSTMModel


def create_dataloader(df, seq_len, batch_size):
    data = df.values.astype(np.float32)
    X, y = [], []

    for i in range(len(data) - seq_len):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len, 0])  # predict first column

    X = torch.tensor(np.array(X))
    y = torch.tensor(np.array(y)).unsqueeze(1)  # shape (N,1)

    dataset = TensorDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_and_validate(train_df, val_df, seq_len=60, epochs=20,
                       batch_size=32, model_type="lstm",
                       params=None, device=None, save_path="best_model.pt"):

    params = params.copy() if params else {}

    lr = params.pop("lr", 0.001) 

    num_layers = params.get("num_layers", 1)
    if num_layers == 1 and params.get("dropout", 0) and params.get("dropout") > 0:
        print("Note: num_layers==1, setting dropout=0 to avoid PyTorch RNN warning.")
        params["dropout"] = 0.0

    input_dim = train_df.shape[1]

    if model_type == "attn":
        model = LSTMAttentionModel(input_dim, **params)
    else:
        model = SimpleLSTMModel(input_dim, **params)

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_loader = create_dataloader(train_df, seq_len, batch_size)
    val_loader = create_dataloader(val_df, seq_len, batch_size)

    best_loss = float("inf")

    for ep in range(epochs):
        model.train()
        train_loss_epoch = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)

            optimizer.zero_grad()
            out = model(Xb)

            if isinstance(out, tuple) or isinstance(out, list):
                preds = out[0]
            else:
                preds = out

            if preds.dim() == 1:
                preds = preds.unsqueeze(1)
            if yb.dim() == 1:
                yb = yb.unsqueeze(1)

            loss = loss_fn(preds, yb)
            loss.backward()
            optimizer.step()

            train_loss_epoch += loss.item() * Xb.size(0)

        train_loss_epoch = train_loss_epoch / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv, yv = Xv.to(device), yv.to(device)
                out = model(Xv)
                if isinstance(out, tuple) or isinstance(out, list):
                    preds_v = out[0]
                else:
                    preds_v = out

                if preds_v.dim() == 1:
                    preds_v = preds_v.unsqueeze(1)
                if yv.dim() == 1:
                    yv = yv.unsqueeze(1)

                val_loss += loss_fn(preds_v, yv).item() * Xv.size(0)

        val_loss = val_loss / len(val_loader.dataset)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), save_path)

        print(f"Epoch {ep+1}/{epochs}  Train Loss: {train_loss_epoch:.6f}  Val Loss: {val_loss:.6f}")

    return {"best_val_loss": best_loss, "save_path": save_path}

