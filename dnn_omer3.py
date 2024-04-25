import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



file_path = "/mnt/12TB/jet_omer/cms_jet_data.h5"
key = "data"
data = pd.read_hdf(file_path, key=key)

df = pd.DataFrame(data)
df_filtered = df[(df["jetGenMatch"] != 0) & (df["genJetPt"] > 20) & (abs(df["jetEta"]) < 2.5)]

exclude_columns = ["genJetPt", "jetPt", "jetGenMatch"]

features = df_filtered[[col for col in df.columns if col not in exclude_columns]]
target = df_filtered["genJetPt"]


scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)




from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, features_scaled, target):
        self.features_scaled = torch.tensor(features_scaled, dtype=torch.float32)
        self.target = torch.tensor(target.to_numpy(), dtype=torch.float32)

    def __len__(self):
        return len(self.features_scaled)

    def __getitem__(self, idx):
        return self.features_scaled[idx], self.target[idx]



# Split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(features_scaled, target, test_size=0.4, random_state=42)
X_validation, X_test, y_validation, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)


X_jet_train, X_jet_temp, y_jet_train, y_jet_temp = train_test_split(df_filtered["jetPt"], target, test_size=0.4, random_state=42)
X_jet_validation, X_jet_test, y_jet_validation, y_jet_test = train_test_split(X_jet_temp, y_temp, test_size=0.5, random_state=42)


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, input_dim):
        super(MyModel, self).__init__()
    
        
        self.layer1 = nn.Linear(input_dim, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 256)  
        self.layer4 = nn.Linear(256, 128)
        self.layer5 = nn.Linear(128, 128) 
        self.layer6 = nn.Linear(128, 64) 
        self.layer7 = nn.Linear(64, 128)
        self.layer8 = nn.Linear(128, 256)
        self.layer9 = nn.Linear(256,256)
        self.layer10 = nn.Linear(256, 1) # Output layer with a single unit for regression

    def forward(self, x):
        x = F.relu(self.layer1(x))  # ReLU activation
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = F.relu(self.layer6(x))
        x = F.relu(self.layer7(x))
        x = F.relu(self.layer8(x))
        x = F.relu(self.layer9(x))
        output = self.layer10(x)
        return output



device = 'cuda'

model = MyModel(X_train.shape[1])
model = model.to(device)
model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003) 



import tqdm
from sklearn.metrics import r2_score
from IPython.display import clear_output



loss_fn = nn.L1Loss()


train_dataset = CustomDataset(X_train, y_train)
test_dataset = CustomDataset(X_test, y_test)
validation_dataset = CustomDataset(X_validation, y_validation)

batch_size = int(2**20)  


train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)
validation_loader = DataLoader(validation_dataset, batch_size=batch_size)



patience = 10
best_val_loss = float('inf')
counter = 0
EPOCHS = 100


# Training loop
loss_list = np.zeros((EPOCHS,))
r2_list = np.zeros((EPOCHS,))
val_loss_list = np.zeros((EPOCHS,))
val_r2_list = np.zeros((EPOCHS,))


# Create empty lists to store loss values for training and validation plotting
train_loss_history = []
val_loss_history = []


for epoch in tqdm.trange(EPOCHS):
    
    # Training phase
    model.train()  
    for inputs, targets in tqdm.tqdm(train_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        y_pred = model(inputs)
        loss = loss_fn(y_pred, targets.view(y_pred.shape))
        #loss = leaky_gaussian_loss(y_pred, targets.view(y_pred.shape), alpha, beta)
        loss_list[epoch] += loss.item()/len(targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Calculate R-squared for the current training batch
        r2_batch = r2_score(targets.cpu().numpy(), y_pred.cpu().detach().numpy())
        r2_list[epoch] += r2_batch

    
    # Compute the training loss for the epoch
    loss_list[epoch] /= len(train_loader)
    r2_list[epoch] /= len(train_loader)

    
    
    # Validation phase ------------------------------------------------------------------------------------------
    model.eval()  
    with torch.no_grad():
        val_loss = 0.0
        y_true_val = []
        y_pred_val = []
        for inputs, targets in validation_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            y_pred = model(inputs)
            val_loss += loss_fn(y_pred, targets.view(y_pred.shape)).item()/len(targets)
            #val_loss += leaky_gaussian_loss(y_pred, targets.view(y_pred.shape), alpha, beta).item()/len(targets)
            y_true_val.extend(targets.cpu().numpy())
            y_pred_val.extend(y_pred.cpu().numpy())

        # Compute the validation loss for the epoch
        val_loss /= len(validation_loader)

        # Calculate R-squared for validation
        r2_val = r2_score(y_true_val, y_pred_val)


        # Store the validation loss and R-squared values
        val_loss_list[epoch] = val_loss
        val_r2_list[epoch] = r2_val

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the model if needed
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping triggered.")
                break

    
    
    # Append the loss values to the history lists
    train_loss_history.append(loss_list[epoch])
    val_loss_history.append(val_loss_list[epoch])




# Testing loop -----------------------------------------------------------------------------------
test_loss = 0.0
y_true_test = []
y_pred_test = []


model.eval()  # Set the model to evaluation mode
with torch.no_grad():
    for inputs, targets in test_loader:
        inputs  = inputs.to(device)
        targets = targets.to(device) 
        y_pred = model(inputs)
        test_loss += loss_fn(y_pred, targets.view(y_pred.shape)).item()/len(targets)
        
        y_true_test.extend(targets.cpu().numpy())
        y_pred_test.extend(y_pred.cpu().numpy())

# Compute the test loss
test_loss /= len(test_loader)


# Calculate R-squared for the test set
r2_test = r2_score(y_true_test, y_pred_test)

# Print or store the test loss and R-squared
print(f"Test Loss: {test_loss}")
print(f"Test R-squared: {r2_test}")


# Plot the training and validation loss
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(range(EPOCHS), loss_list, label='Training Loss')
plt.plot(range(EPOCHS), val_loss_list, label='Validation Loss')
plt.plot(range(EPOCHS), [test_loss] * EPOCHS, label='Test Loss', linestyle='--', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training, Validation, and Test Loss')

# Plot the training and validation R-squared
plt.subplot(1, 2, 2)
plt.plot(range(EPOCHS), r2_list, label='Training R-squared')
plt.plot(range(EPOCHS), val_r2_list, label='Validation R-squared')
plt.plot(range(EPOCHS), [r2_test] * EPOCHS, label='Test R-squared', linestyle='--', color='red')
plt.xlabel('Epoch')
plt.ylabel('R-squared')
plt.legend()
plt.title('Training, Validation, and Test R-squared')

plt.savefig('/home/odokumaci/git_repo/correction-repo/dnn_full/test_loss_r_squared.png')



def bin_ratio(basejet_hist, genjet_hist):
    

    log_bins = np.logspace(np.log10(genjet_hist.min()), np.log10(genjet_hist.max()), 30)
    _, bin_edges = np.histogram(genjet_hist, bins=log_bins)

    ratios = []
    mean_list = []
    std_list = []
    iqr_list = []  # List to store interquartile ranges
    median_list = []  # List to store medians

    for ibin in range(len(bin_edges) - 1):
        idx = (basejet_hist >= bin_edges[ibin]) & (basejet_hist <= bin_edges[ibin + 1])  # boolean mask
        ratios.append((basejet_hist[idx] - 0 * genjet_hist[idx]) / genjet_hist[idx])

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    for hist in ratios:
        if len(hist) > 0:  # Check if hist is not empty
            mean_list.append(np.mean(hist))
            std_list.append(np.std(hist))
            iqr = np.percentile(hist, 75) - np.percentile(hist, 25)  # Interquartile range calculation
            iqr_list.append(iqr)
            median_list.append(np.median(hist))
        else:
            mean_list.append(np.nan)  # Append NaN if hist is empty
            std_list.append(np.nan)
            iqr_list.append(np.nan)
            median_list.append(np.nan)

    iqr_ratio = np.array(iqr_list) / np.array(median_list)  # Calculating IQR ratio


    return np.array(mean_list), np.array(std_list), bin_centers, bin_edges, iqr_ratio



basetogen_mean, basetogen_std, bin_centers, bin_edges, basetogen_iqr_ratio = bin_ratio(X_jet_test, y_test)
corrtogen_mean, corrtogen_std, _, _, corrtogen_iqr_ratio = bin_ratio(y_pred.flatten(), y_test)

# Check for division by zero and NaN values
valid_indices = (basetogen_iqr_ratio != 0) & (~np.isnan(corrtogen_iqr_ratio)) & (~np.isnan(basetogen_iqr_ratio))

# Calculate the ratio difference only for valid indices
beta = np.zeros_like(basetogen_iqr_ratio)
beta[valid_indices] = 1 - (corrtogen_iqr_ratio[valid_indices] / basetogen_iqr_ratio[valid_indices])

resolution_ratio = corrtogen_iqr_ratio/basetogen_iqr_ratio

np.save('/home/odokumaci/git_repo/correction-repo/dnn_full/dnn_corriqr.npy', corrtogen_iqr_ratio)
np.save('/home/odokumaci/git_repo/correction-repo/dnn_full/dnn_baseiqr.npy', basetogen_iqr_ratio)
np.save('/home/odokumaci/git_repo/correction-repo/dnn_full/dnn_corrtogen.npy', corrtogen_mean)
np.save('/home/odokumaci/git_repo/correction-repo/dnn_full/dnn_basetogen.npy', basetogen_mean)
np.save('/home/odokumaci/git_repo/correction-repo/dnn_full/dnn_bincenters.npy', bin_centers) 
np.save('/home/odokumaci/git_repo/correction-repo/dnn_full/dnn_corriqr_to_baseiqr.npy', resolution_ratio)




# Plotting the histogram and ratio plot
fig, (ax1, ax2) = plt.subplots(2, 1, sharex='col', gridspec_kw={'height_ratios': [4, 1]}, figsize=(12, 8))


ax1.hist(X_jet_test, bins=bin_edges, label='Baseline Jet Pt', fill=0, edgecolor='blue', linewidth=1.2)
ax1.hist(y_pred, bins=bin_edges, label='DNN Corrected Jet Pt', edgecolor="red", alpha=0.5, linewidth=1.2)
ax1.hist(y_test, bins=bin_edges, label='GenJet Pt', fill=0, edgecolor='green', linewidth=1.5)

ax1.set_xlabel('Jet Pt')
ax1.set_ylabel('Frequency')
ax1.set_title('Histogram of Base Jet Pt, DNN Corrected Jet Pt, and Gen Jet Pt')
ax1.set_yscale('log')
ax1.legend()


ax2.errorbar(bin_centers, basetogen_mean, yerr=0 * basetogen_std, fmt='o', markersize=3.0, color='blue', label='Baseline Data')
ax2.errorbar(bin_centers, corrtogen_mean, yerr=0 * corrtogen_std, fmt='o', markersize=3.0, alpha=0.5, color='red', label='DNN Corrected Data')
ax2.axhline(y=1.0, color='g', linestyle='--')

ax2.set_xlabel('jet pT (GeV)')
ax2.set_ylabel('Mean Ratio')
ax2.set_title('DNN Corrected Jet Pt and Baseline Jet pT')
ax2.set_xscale('log')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend(fontsize='small')

#ax1.set_xlim(0, 2500)
#ax2.set_xlim(0, 2500)

plt.tight_layout()
plt.savefig('/home/odokumaci/git_repo/correction-repo/dnn_full/ratio_plot.png')