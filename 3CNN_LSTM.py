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
# 1. Dataset Creation (Global Scaling, per-file sequences)      #
#################################################################
class SleepDataset(Dataset):
    def __init__(self, folder_path, train_file_count=None):
        self.features = []
        self.labels = []
        all_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path) if f.endswith('.csv')
        ])
        if train_file_count:
            all_paths = all_paths[:train_file_count]
        if not all_paths:
            raise ValueError(f"No CSV files in {folder_path}")
        # fit scaler on all
        all_feats = [pd.read_csv(p).iloc[:, :-1].values for p in all_paths]
        all_feats = np.vstack(all_feats)
        self.scaler = StandardScaler().fit(all_feats)
        # load each
        for p in all_paths:
            df = pd.read_csv(p)
            feats = df.iloc[:, :-1].values
            labs = df.iloc[:, -1].values
            labs = np.where(labs==0, 0,
                   np.where(np.isin(labs,[1,2,3,4]),1,
                   np.where(labs==5,2,-1)))
            mask = labs!=-1
            feats = feats[mask]; labs = labs[mask]
            feats = self.scaler.transform(feats)
            self.features.append(torch.tensor(feats, dtype=torch.float32))
            self.labels.append(torch.tensor(labs, dtype=torch.long))
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

##########################################
# 2. Model Definition (CNN → window → LSTM) #
##########################################
class SleepModelMultiBranch(nn.Module):
    def __init__(self, total_input_size, lstm_hidden_size=256, dropout_rate=0.5, win_len=12):
        super().__init__()
        self.win_len = win_len
        # CNN: same as original
        self.acti_conv1 = nn.Conv1d(8,32,kernel_size=9,stride=1,padding=4)
        self.acti_conv2 = nn.Conv1d(32,64,kernel_size=9,stride=1,padding=4)
        self.acti_conv3 = nn.Conv1d(64,64,kernel_size=9,stride=1,padding=4)
        self.hr_conv1   = nn.Conv1d(8,32,kernel_size=9,stride=1,padding=4)
        self.hr_conv2   = nn.Conv1d(32,64,kernel_size=9,stride=1,padding=4)
        self.hr_conv3   = nn.Conv1d(64,64,kernel_size=9,stride=1,padding=4)
        self.dropout    = nn.Dropout(dropout_rate)
        self.combined_ln = nn.LayerNorm(128)
        self.fusion_linear = nn.Linear(128,128)
        self.fusion_activation = nn.ReLU()
        self.lstm       = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, batch_first=True)
        self.fc         = nn.Linear(lstm_hidden_size,3)

    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        # split features
        acti = x[:,:,:8].permute(0,2,1)  # (B,8,T)
        hr   = x[:,:,8:16].permute(0,2,1)
        # CNN
        acti = self.dropout(torch.relu(self.acti_conv1(acti)))
        acti = self.dropout(torch.relu(self.acti_conv2(acti)))
        acti = self.dropout(torch.relu(self.acti_conv3(acti)))
        acti = acti.permute(0,2,1)       # (B,T,64)
        hr   = self.dropout(torch.relu(self.hr_conv1(hr)))
        hr   = self.dropout(torch.relu(self.hr_conv2(hr)))
        hr   = self.dropout(torch.relu(self.hr_conv3(hr)))
        hr   = hr.permute(0,2,1)         # (B,T,64)
        # fuse
        combined = torch.cat([acti, hr], dim=2)  # (B,T,128)
        combined = self.combined_ln(combined)
        combined = self.fusion_activation(self.fusion_linear(combined))
        # sliding windows
        wins = combined.unfold(1, self.win_len, 1)  # (B, num_wins, win_len, 128)
        B2, num_wins, wlen, C = wins.shape
        wins = wins.contiguous().view(-1, wlen, C)   # (B*num_wins, win_len, 128)
        # compact LSTM weights
        self.lstm.flatten_parameters()
        # LSTM
        lstm_out, (h_n, _) = self.lstm(wins)
        h_last = h_n[-1]
        out = self.dropout(h_last)
        logits = self.fc(out)                      # (B*num_wins, 3)
        return logits.view(B, num_wins, -1)        # (B, num_wins, 3)

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
    data_folder = "/home/amyn/projects/def-montazn/amyn/SleepStageProject/mesa/features_smoothed_all_reduced_manualr"
    test_folder = "/home/amyn/projects/def-montazn/amyn/SleepStageProject/mesa/test_smoothed_all_reduced_manual"
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
