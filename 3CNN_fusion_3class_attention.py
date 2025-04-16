import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

#################################################################
# 1. Dataset Creation (Global Scaling, Subset of Training Files) #
#################################################################
class SleepDataset(Dataset):
    def __init__(self, folder_path, window_size, train_file_count=None, efficiency_threshold=100):
        self.inputs = []
        self.labels = []
        self.window_size = window_size

        # Get list of all CSV files.
        all_paths = sorted([
            os.path.join(folder_path, f)
            for f in os.listdir(folder_path) if f.endswith(".csv")
        ])
        if train_file_count is not None:
            all_paths = all_paths[:train_file_count]

        # --- Filter files based on sleep efficiency ---
        filtered_paths = []
        for path in all_paths:
            df = pd.read_csv(path, header=0)
            original_labels = df.iloc[:, -1].values

            # For filtering: binary wake/sleep
            labels_binary = np.where(original_labels == 0, 0, 1)
            efficiency = np.mean(labels_binary) * 100
            if efficiency < efficiency_threshold:
                filtered_paths.append(path)
                print(f"Including file {path} with sleep efficiency: {efficiency:.2f}%")
            else:
                print(f"Excluding file {path} with sleep efficiency: {efficiency:.2f}%")
                
        if len(filtered_paths) == 0:
            raise ValueError("No files found with sleep efficiency below the threshold.")

        # --- Global Scaling ---
        all_features = []
        for path in filtered_paths:
            df = pd.read_csv(path, header=0)
            features = df.iloc[:, :-1].values
            all_features.append(features)
        all_features = np.vstack(all_features)
        self.scaler = StandardScaler()
        self.scaler.fit(all_features)

        # --- Create windows from each filtered file ---
        for path in filtered_paths:
            df = pd.read_csv(path, header=0)
            features = df.iloc[:, :-1].values
            labels = df.iloc[:, -1].values

            # Remap labels: 0 → 0, [1,2,3,4] → 1, 5 → 2
            labels = np.where(labels == 0, 0,
                      np.where(np.isin(labels, [1, 2, 3, 4]), 1,
                      np.where(labels == 5, 2, -1)))  # Optional: set unknowns as -1

            # Optional: remove unknown labels (if any)
            valid_indices = labels != -1
            features = features[valid_indices]
            labels = labels[valid_indices]

            features = self.scaler.transform(features)

            for i in range(len(features) - window_size):
                window = features[i : i + window_size]
                lbl = labels[i + window_size - 1]
                self.inputs.append(torch.tensor(window, dtype=torch.float32))
                self.labels.append(torch.tensor(lbl, dtype=torch.long))

        self.inputs = torch.stack(self.inputs)
        self.labels = torch.stack(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


##########################################
# 2. Modified Model Definition with Separate CNN Branches #
##########################################
import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        # Linear transformation applied to each hidden state.
        self.attn = nn.Linear(hidden_size, hidden_size)
        # Learnable context vector for computing scalar attention scores.
        self.context_vector = nn.Parameter(torch.randn(hidden_size))
    
    def forward(self, lstm_outputs):
        """
        Args:
            lstm_outputs: Tensor of shape (batch, seq_len, hidden_size)
        Returns:
            context: Tensor of shape (batch, hidden_size) - the weighted sum.
            attn_weights: Tensor of shape (batch, seq_len) - the attention weights.
        """
        # Apply a linear transformation and tanh nonlinearity.
        energy = torch.tanh(self.attn(lstm_outputs))  # (batch, seq_len, hidden_size)
        # Compute attention scores by dot product with the context vector.
        scores = energy.matmul(self.context_vector)   # (batch, seq_len)
        # Normalize the scores to obtain attention weights.
        attn_weights = F.softmax(scores, dim=1)         # (batch, seq_len)
        # Compute the context vector as the weighted sum of LSTM outputs.
        context = torch.sum(lstm_outputs * attn_weights.unsqueeze(-1), dim=1)  # (batch, hidden_size)
        return context, attn_weights

class SleepModelMultiBranchWithAttention(nn.Module):
    def __init__(self, total_input_size, lstm_hidden_size=256, dropout_rate=0.5):
        """
        Expects total_input_size = 16.
        Modified for 3-class classification: wake, nonrem, rem.
        This model uses 3 convolutional layers per branch.
        For the actigraphy branch, only the first feature is used.
        For the heart rate branch, 2 features are used from columns 9 and 16 
        (i.e., indices 8 and 15 in 0-indexed Python).
        An attention mechanism is applied over the LSTM outputs.
        """
        super(SleepModelMultiBranchWithAttention, self).__init__()
        
        # --- Actigraphy Branch --- (using 1 input feature: channels = 8 from slice)
        self.acti_conv1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=9, stride=1, padding=1)
        self.acti_conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=1)
        self.acti_conv3 = nn.Conv1d(64, 64, kernel_size=9, stride=1, padding=1)
        
        # --- Heart Rate Branch --- (using 2 input features: channels = 8 from slice)
        self.hr_conv1 = nn.Conv1d(in_channels=8, out_channels=32, kernel_size=9, stride=1, padding=1)
        self.hr_conv2 = nn.Conv1d(32, 64, kernel_size=9, stride=1, padding=1)
        self.hr_conv3 = nn.Conv1d(64, 64, kernel_size=9, stride=1, padding=1)
        
        # Dropout layer (you can enable/disable dropout as needed)
        self.dropout = nn.Dropout(dropout_rate)
        
        # --- Post-Concatenation Processing ---
        self.combined_ln = nn.LayerNorm(normalized_shape=128)
        
        # --- Linear Fusion ---
        self.fusion_linear = nn.Linear(128, 128)
        self.fusion_activation = nn.ReLU()
        
        # --- LSTM ---
        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=False)
        
        # --- Attention ---
        self.attention = Attention(lstm_hidden_size)
        
        # --- Final Classifier ---
        self.fc = nn.Linear(lstm_hidden_size, 3)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, window_size, total_input_size).
               Here, total_input_size=16.
               The model uses:
                  - The actigraphy branch receives the features from columns 0 to 7.
                  - The heart rate branch receives the features from columns 8 to 15.
        """
        # --- Step 1: Split the input for each branch ---
        # For actigraphy branch, select features from columns 0 to 7.
        acti_x = x[:, :, 0:8]  # shape: (batch, window_size, 8)
        # For heart rate branch, select features from columns 8 to 16.
        hr_x = x[:, :, 8:16]   # shape: (batch, window_size, 8)
        
        # --- Step 2: Permute for CNN processing ---
        # CNNs expect the channel dimension in second place.
        acti_x = acti_x.permute(0, 2, 1)  # shape: (batch, 8, window_size)
        hr_x = hr_x.permute(0, 2, 1)      # shape: (batch, 8, window_size)
        
        # --- Step 3: Process each branch via CNN layers ---
        # Actigraphy Branch:
        acti_x = F.relu(self.acti_conv1(acti_x))
        acti_x = self.dropout(acti_x)
        acti_x = F.relu(self.acti_conv2(acti_x))
        acti_x = self.dropout(acti_x)
        acti_x = F.relu(self.acti_conv3(acti_x))
        acti_x = self.dropout(acti_x)
        # Permute back to (batch, window_size, channels)
        acti_x = acti_x.permute(0, 2, 1)   # shape: (batch, window_size, 64)
        
        # Heart Rate Branch:
        hr_x = F.relu(self.hr_conv1(hr_x))
        hr_x = self.dropout(hr_x)
        hr_x = F.relu(self.hr_conv2(hr_x))
        hr_x = self.dropout(hr_x)
        hr_x = F.relu(self.hr_conv3(hr_x))
        hr_x = self.dropout(hr_x)
        # Permute back to (batch, window_size, channels)
        hr_x = hr_x.permute(0, 2, 1)       # shape: (batch, window_size, 64)
        
        # --- Step 4: Concatenate and fuse the branch outputs ---
        combined = torch.cat((acti_x, hr_x), dim=2)  # shape: (batch, window_size, 128)
        combined = self.combined_ln(combined)
        
        combined = self.fusion_linear(combined)
        combined = self.fusion_activation(combined)
        
        # --- Step 5: LSTM Processing ---
        # Here, the sequence length is window_size.
        lstm_out, (h_n, _) = self.lstm(combined)  # lstm_out: (batch, window_size, lstm_hidden_size)
        
        # --- Step 6: Apply Attention ---
        context, attn_weights = self.attention(lstm_out)  # context: (batch, lstm_hidden_size)
        # You can choose to apply dropout here if desired.
        context = self.dropout(context)
        
        # --- Step 7: Final Classification ---
        out = self.fc(context)  # shape: (batch, 3)
        return out, attn_weights
################################################
# 3. Training Function (with Variable Optimizer)  #
################################################
def train_model(model, train_loader, val_loader, epochs, learning_rate, class_weights, save_dir, window_size, threshold, batch_size, optimizer_name, patience=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    if optimizer_name.lower() == "rmsprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9,weight_decay=1e-4)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    best_val_loss = float("inf")
    train_metrics, val_metrics = [], []

    # Early stopping variables
    early_stop_counter = 0

    print("Training started...")
    for epoch in range(epochs):
        model.train()
        train_loss, train_preds, train_targets = 0.0, [], []
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs, attn_weights = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            train_targets.extend(labels.cpu().numpy())

        train_loss /= len(train_loader)
        train_acc = accuracy_score(train_targets, train_preds)
        train_f1 = f1_score(train_targets, train_preds, average="weighted")
        train_metrics.append((train_loss, train_acc, train_f1))

        model.eval()
        val_loss, val_preds, val_targets = 0.0, [], []
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs, attn_weights = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_targets.extend(labels.cpu().numpy())

        val_loss /= len(val_loader)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="weighted")
        val_metrics.append((val_loss, val_acc, val_f1))

        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Train F1: {train_f1:.4f} | Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

        # Early stopping check: save model if validation loss improves, otherwise increment counter.
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
            best_model_path = os.path.join(save_dir, f"best_model_{threshold}_ws_{window_size}_batch_{batch_size}_opt_{optimizer_name}.pth")
            torch.save(model.state_dict(), best_model_path)
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    plot_metrics(train_metrics, val_metrics, save_dir)

###################################################
# 4. Plot Metrics                                 #
###################################################
def plot_metrics(train_metrics, val_metrics, save_dir):
    train_loss, train_acc, train_f1 = zip(*train_metrics)
    val_loss, val_acc, val_f1 = zip(*val_metrics)
    epochs = range(1, len(train_loss) + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "loss_plot.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "accuracy_plot.png"))
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(epochs, train_f1, label="Train F1 Score")
    plt.plot(epochs, val_f1, label="Val F1 Score")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.title("F1 Score vs Epochs")
    plt.legend()
    plt.savefig(os.path.join(save_dir, "f1_score_plot.png"))
    plt.close()

###################################################
# 5. Testing Function with Detailed Reports       #
###################################################
def test_model_func(model, test_folder, window_size, scaler, device, save_dir, threshold, batch_size, optimizer_name):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds, all_targets = [], []
    classification_reports = []

    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) if f.endswith(".csv")]

    for file_path in test_files:
        df = pd.read_csv(file_path, header=0)
        features = scaler.transform(df.iloc[:, :-1].values)
        # Original labels
        original_labels = df.iloc[:, -1].values

        # Remap: 0 → 0 (wake), 1–4 → 1 (non-REM), 5 → 2 (REM)
        labels = np.where(original_labels == 0, 0,
            np.where(np.isin(original_labels, [1, 2, 3, 4]), 1,
            np.where(original_labels == 5, 2, -1)))  # Optional: unknowns → -1

        # Optional: remove invalid labels
        valid_indices = labels != -1
        features = features[valid_indices]
        labels = labels[valid_indices]

        windowed_inputs, windowed_labels = [], []
        for i in range(len(features) - window_size):
            windowed_inputs.append(torch.tensor(features[i : i + window_size], dtype=torch.float32))
            windowed_labels.append(torch.tensor(labels[i + window_size - 1], dtype=torch.long))

        if len(windowed_inputs) == 0:
            continue

        inputs_tensor = torch.stack(windowed_inputs).to(device)
        labels_tensor = torch.stack(windowed_labels).to(device)

        with torch.no_grad():
            outputs, attn_weights = model(inputs_tensor)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            true_labels = labels_tensor.cpu().numpy()

        all_preds.extend(preds)
        all_targets.extend(true_labels)

        report_dict = classification_report(true_labels, preds, target_names=["wake", "nonrem", "rem"], output_dict=True)
        classification_reports.append({
            "file_name": os.path.basename(file_path),
            "accuracy": report_dict["accuracy"],
            "wake_precision": report_dict["wake"]["precision"],
            "wake_recall": report_dict["wake"]["recall"],
            "wake_f1": report_dict["wake"]["f1-score"],
            "nonrem_precision": report_dict["nonrem"]["precision"],
            "nonrem_recall": report_dict["nonrem"]["recall"],
            "nonrem_f1": report_dict["nonrem"]["f1-score"],
            "rem_precision": report_dict["rem"]["precision"],
            "rem_recall": report_dict["rem"]["recall"],
            "rem_f1": report_dict["rem"]["f1-score"],
            "macro_avg_precision": report_dict["macro avg"]["precision"],
            "macro_avg_recall": report_dict["macro avg"]["recall"],
            "macro_avg_f1": report_dict["macro avg"]["f1-score"],
            "weighted_avg_precision": report_dict["weighted avg"]["precision"],
            "weighted_avg_recall": report_dict["weighted avg"]["recall"],
            "weighted_avg_f1": report_dict["weighted avg"]["f1-score"],
        })

    per_file_report_df = pd.DataFrame(classification_reports)
    per_file_report_csv = os.path.join(save_dir, f"test_classification_report_{threshold}_ws_{window_size}_batch_{batch_size}_opt_{optimizer_name}.csv")
    per_file_report_df.to_csv(per_file_report_csv, index=False)
    print(f"Per-file classification reports saved to {per_file_report_csv}")

    overall_report_dict = classification_report(all_targets, all_preds, target_names=["wake", "nonrem", "rem"], output_dict=True)
    overall_classification = {
        "accuracy": overall_report_dict["accuracy"],
        "wake_precision": overall_report_dict["wake"]["precision"],
        "wake_recall": overall_report_dict["wake"]["recall"],
        "wake_f1": overall_report_dict["wake"]["f1-score"],
        "nonrem_precision": overall_report_dict["nonrem"]["precision"],
        "nonrem_recall": overall_report_dict["nonrem"]["recall"],
        "nonrem_f1": overall_report_dict["nonrem"]["f1-score"],
        "rem_precision": overall_report_dict["rem"]["precision"],
        "rem_recall": overall_report_dict["rem"]["recall"],
        "rem_f1": overall_report_dict["rem"]["f1-score"],
        "macro_avg_precision": overall_report_dict["macro avg"]["precision"],
        "macro_avg_recall": overall_report_dict["macro avg"]["recall"],
        "macro_avg_f1": overall_report_dict["macro avg"]["f1-score"],
        "weighted_avg_precision": overall_report_dict["weighted avg"]["precision"],
        "weighted_avg_recall": overall_report_dict["weighted avg"]["recall"],
        "weighted_avg_f1": overall_report_dict["weighted avg"]["f1-score"],
    }
    overall_report_csv = os.path.join(save_dir, f"overall_classification_report_{threshold}_ws_{window_size}_batch_{batch_size}_opt_{optimizer_name}.csv")
    overall_report_df = pd.DataFrame([overall_classification])
    overall_report_df.to_csv(overall_report_csv, index=False)
    print(f"Overall classification report saved to {overall_report_csv}")

    cm = confusion_matrix(all_targets, all_preds)
    cm_sum = np.sum(cm)
    cm_percentage = 100 * cm / cm_sum

    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            c = cm[i, j]
            annot[i, j] = f"{c}"

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=annot, fmt="", cmap="Blues",
                xticklabels=["wake", "nonrem", "rem"],
                yticklabels=["wake", "nonrem", "rem"])
    plt.title(f"Cumulative Confusion Matrix (Threshold: {threshold}, WS: {window_size}, Batch: {batch_size}, Opt: {optimizer_name})")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_png = os.path.join(save_dir, f"cumulative_confusion_matrix_{threshold}_ws_{window_size}_batch_{batch_size}_opt_{optimizer_name}.png")
    plt.savefig(cm_png)
    plt.close()
    print(f"Cumulative confusion matrix saved to {cm_png}")

###################################################
# 6. Main Loop for Training and Testing           #
###################################################
# Assume you have imported or defined:
# SleepDataset, train_model, test_model_func, and SleepModelMultiBranch

if __name__ == "__main__":
    threshold = 0.02
    learning_rate = 0.0005
    epochs = 150
    window_size = 40

    # Define lists of hyperparameters for looping
    train_file_counts = [1000]         # Different training file counts
    batch_sizes = [128,256]               # Different batch sizes
    lstm_hidden_sizes = [256]         # Different LSTM hidden sizes
    dropout_rates = [0.5]             # Different dropout rates
    optimizer_names = ["sgd"]         # Different optimizers, adjust as needed

    data_folder = "/home/amyn/projects/def-montazn/amyn/SleepStageProject/mesa/features_smoothed_all_reduced_manualr"
    test_folder = "/home/amyn/projects/def-montazn/amyn/SleepStageProject/mesa/test_smoothed_all_reduced_manual"

    # Loop over the grid of hyperparameters
    for optimizer_name in optimizer_names:
        for train_file_count in train_file_counts:
            for batch_size in batch_sizes:
                for lstm_hidden_size in lstm_hidden_sizes:
                    for dropout_rate in dropout_rates:
                        print(f"\n--- Training started: Optimizer={optimizer_name}, Train Size={train_file_count}, Batch Size={batch_size}, LSTM Hidden Size={lstm_hidden_size}, Dropout={dropout_rate} ---")
                        parent_dir = "/home/amyn/projects/def-montazn/amyn/SleepStageProject/6CNN_fusion_3class_all"
                        os.makedirs(parent_dir, exist_ok=True)
                        # Create a unique directory for results
                        save_dir = os.path.join(parent_dir, f"3CNNfusion_0.0005attention_allker9_tr{train_file_count}_bs{batch_size}_lstm{lstm_hidden_size}")
                        os.makedirs(save_dir, exist_ok=True)
                        
                        # Prepare the dataset and dataloaders
                        dataset = SleepDataset(data_folder, window_size, train_file_count=train_file_count)
                        train_size = int(0.8 * len(dataset))
                        val_size = len(dataset) - train_size
                        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
                        
                        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                        
                        # Compute class weights for 3 classes.
                        labels_cpu = dataset.labels.cpu().numpy()
                        class_weights = compute_class_weight("balanced", classes=np.array([0, 1, 2]), y=labels_cpu)
                        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
                        
                        # Instantiate the model with current hyperparameters
                        model = SleepModelMultiBranchWithAttention(
                            total_input_size=dataset.inputs.shape[2],
                            lstm_hidden_size=lstm_hidden_size,
                            dropout_rate=dropout_rate
                        )
                        model.to(device)
                        
                        # Train the model
                        train_model(model, train_loader, val_loader, epochs, learning_rate,
                                    class_weights, save_dir, window_size, threshold, batch_size, optimizer_name)
                        print(f"--- Training completed: Optimizer={optimizer_name}, Train Size={train_file_count}, Batch Size={batch_size}, LSTM Hidden Size={lstm_hidden_size}, Dropout={dropout_rate} ---")
                        
                        # Load the best model and test
                        best_model_path = os.path.join(save_dir, f"best_model_{threshold}_ws_{window_size}_batch_{batch_size}_opt_{optimizer_name}.pth")
                        model.load_state_dict(torch.load(best_model_path, map_location=device))
                        model.to(device)
                        
                        print(f"--- Testing started for Optimizer={optimizer_name} ---")
                        test_model_func(model, test_folder, window_size, dataset.scaler, device, save_dir, threshold, batch_size, optimizer_name)
                        print(f"--- Testing and report generation completed for Optimizer={optimizer_name} ---\n")
