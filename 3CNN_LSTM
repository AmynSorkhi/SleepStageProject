import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#################################################################
# 1. Dataset Creation (Per-file sequences, no sleep-efficiency)  #
#################################################################
class SleepDataset(Dataset):
    """
    Loads each CSV file as a sequence of epochs (features + label).
    Returns: (features_tensor [num_epochs, feat_dim], labels_tensor [num_epochs])
    """
    def __init__(self, folder_path, train_file_count=None):
        self.data = []  # list of (features, labels) per file
        # Gather CSV paths
        all_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path)
            if f.endswith('.csv')
        ])
        if train_file_count is not None:
            all_paths = all_paths[:train_file_count]
        if not all_paths:
            raise ValueError(f"No CSV files found in {folder_path}")

        # --- Global scaling fit ---
        all_feats = []
        for path in all_paths:
            df = pd.read_csv(path)
            feats = df.iloc[:, :-1].values
            all_feats.append(feats)
        all_feats = np.vstack(all_feats)
        self.scaler = StandardScaler().fit(all_feats)

        # --- Load, remap labels, scale, and store each file ---
        for path in all_paths:
            df = pd.read_csv(path)
            feats = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values
            # Remap: 0→0, 1–4→1, 5→2, others→-1
            labels = np.where(labels == 0, 0,
                      np.where(np.isin(labels, [1,2,3,4]), 1,
                      np.where(labels == 5, 2, -1)))
            # filter out unknowns
            mask = labels != -1
            feats = feats[mask]
            labels = labels[mask]
            feats = self.scaler.transform(feats)
            # to tensors
            feats_t = torch.tensor(feats, dtype=torch.float32)
            labels_t = torch.tensor(labels, dtype=torch.long)
            self.data.append((feats_t, labels_t))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

##########################################
# 2. Model Definition (CNN per-epoch → sliding windows → LSTM) #
##########################################
class SleepModel(nn.Module):
    def __init__(self, feat_dim, lstm_hidden=256, dropout=0.5,
                 win_len=12, win_stride=1):
        super(SleepModel, self).__init__()
        self.win_len = win_len
        self.win_stride = win_stride
        # CNN branches (pointwise conv across features)
        # Actigraphy: first 8 dims
        self.acti_conv = nn.Sequential(
            nn.Conv1d(in_channels=8, out_channels=32, kernel_size=1),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(), nn.Dropout(dropout)
        )
        # Heart rate: next 8 dims
        self.hr_conv = nn.Sequential(
            nn.Conv1d(8, 32, kernel_size=1),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(), nn.Dropout(dropout)
        )
        # Fusion
        self.layer_norm = nn.LayerNorm(128)
        self.fusion = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU()
        )
        # LSTM + classifier
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden,
                            batch_first=True)
        self.classifier = nn.Linear(lstm_hidden, 3)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, T, feat_dim)
        B, T, D = x.shape
        # split branches
        acti = x[:, :, :8]  # (B, T, 8)
        hr   = x[:, :, 8:16]
        # reshape for conv1d: (B, C, T)
        acti = acti.permute(0, 2, 1)
        hr   = hr.permute(0, 2, 1)
        # CNN per epoch (pointwise)
        acti_feat = self.acti_conv(acti).permute(0, 2, 1)  # (B, T, 64)
        hr_feat   = self.hr_conv(hr).permute(0, 2, 1)      # (B, T, 64)
        # combine
        combined = torch.cat([acti_feat, hr_feat], dim=2)  # (B, T, 128)
        combined = self.layer_norm(combined)
        combined = self.fusion(combined)                   # (B, T, 128)
        # sliding windows along time
        # windows: (B, num_wins, win_len, 128)
        wins = combined.unfold(dimension=1,
                               size=self.win_len,
                               step=self.win_stride)
        B, num_wins, win_len, C = wins.shape
        # reshape for LSTM: (B*num_wins, win_len, 128)
        wins = wins.contiguous().view(-1, win_len, C)
        # LSTM
        lstm_out, (h_n, _) = self.lstm(wins)
        h_last = h_n[-1]  # (B*num_wins, hidden)
        h_last = self.dropout(h_last)
        logits = self.classifier(h_last)  # (B*num_wins, 3)
        # reshape back: (B, num_wins, 3)
        return logits.view(B, num_wins, -1)

################################################
# 3. Training Function (handles windowed output) #
################################################
def train_model(model, train_loader, val_loader,
                epochs, lr, class_weights,
                save_dir, win_len, win_stride,
                optimizer_name='adam', patience=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    # optimizer
    opt = {
        'adam': optim.Adam,
        'sgd': optim.SGD,
        'rmsprop': optim.RMSprop
    }[optimizer_name.lower()](model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    best_loss = float('inf')
    early_count = 0
    history = {'train_loss':[], 'val_loss':[]}

    for epoch in range(1, epochs+1):
        # -- training --
        model.train()
        running_loss = 0.0
        for feats, labs in train_loader:
            feats, labs = feats.to(device), labs.to(device)
            # predict: (B, W, 3)
            logits = model(feats)
            B, W, _ = logits.shape
            # windowed labels: (B, W)
            win_labs = labs.unfold(dimension=1,
                                   size=win_len,
                                   step=win_stride)[..., -1]
            # flatten both
            logits_flat = logits.view(-1, 3)
            labs_flat = win_labs.reshape(-1)
            loss = criterion(logits_flat, labs_flat)
            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        # -- validation --
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for feats, labs in val_loader:
                feats, labs = feats.to(device), labs.to(device)
                logits = model(feats)
                win_labs = labs.unfold(1, win_len, win_stride)[..., -1]
                loss = criterion(logits.view(-1,3), win_labs.reshape(-1))
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        print(f"Epoch {epoch}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # early stopping
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss; early_count = 0
            torch.save(model.state_dict(), os.path.join(save_dir, 'best.pth'))
        else:
            early_count += 1
            if early_count >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
    # plot losses
    plt.figure(); plt.plot(history['train_loss'], label='train'); plt.plot(history['val_loss'], label='val')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

###################################################
# 4. Testing Function                              #
###################################################
def test_model(model, data_loader, win_len, win_stride, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labs in data_loader:
            feats = feats.to(device)
            logits = model(feats)  # (B, W, 3)
            preds = logits.argmax(dim=2).cpu().numpy()
            win_labs = labs.unfold(1, win_len, win_stride)[..., -1].cpu().numpy()
            # collect
            for p, l in zip(preds, win_labs):
                all_preds.extend(p); all_labels.extend(l)
    # metrics
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    print(f"Test Acc: {acc:.4f}, F1: {f1:.4f}")
    print(classification_report(all_labels, all_preds, target_names=['wake','nonrem','rem']))
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['wake','nonrem','rem'],
                yticklabels=['wake','nonrem','rem'])
    plt.xlabel('Pred'); plt.ylabel('True');
    plt.title('Confusion Matrix')
    plt.show()

###################################################
# 5. Main: Prepare DataLoaders, Model, and Run     #
###################################################
if __name__ == '__main__':
    data_folder = '/path/to/train_folder'
    test_folder = '/path/to/test_folder'
    save_dir = './results'
    os.makedirs(save_dir, exist_ok=True)

    # parameters
    epochs = 100
    lr = 5e-4
    win_len = 12
    win_stride = 1
    lstm_hidden = 256
    dropout = 0.5
    optimizer_name = 'adam'
    batch_size = 1  # per-file

    # datasets & loaders
    train_ds = SleepDataset(data_folder)
    test_ds  = SleepDataset(test_folder)
    # split train/val by files
    n_train = int(0.8 * len(train_ds))
    train_ds, val_ds = random_split(train_ds, [n_train, len(train_ds)-n_train])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    # compute class weights over train set
    all_labels = torch.cat([labs for (_, labs) in train_ds]).numpy()
    cw = compute_class_weight('balanced', classes=np.array([0,1,2]), y=all_labels)
    cw = torch.tensor(cw, dtype=torch.float32)

    # model
    feat_dim = train_ds[0][0].shape[1]
    model = SleepModel(feat_dim, lstm_hidden, dropout, win_len, win_stride)

    # train
    train_model(model, train_loader, val_loader,
                epochs, lr, cw, save_dir,
                win_len, win_stride, optimizer_name)

    # load best and test
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(os.path.join(save_dir,'best.pth')))
    model = model.to(device)
    test_model(model, test_loader, win_len, win_stride, device)
