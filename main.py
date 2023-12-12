# %%
import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
# %%

df = pd.read_csv('creditcard_train.csv')
test_df = pd.read_csv('creditcard_test.csv')

# %%
# splitting had been done
print('No Frauds', round(df['Class'].value_counts()[0]/len(df) * 100,2), '% of the dataset')
print('Frauds', round(df['Class'].value_counts()[1]/len(df) * 100,2), '% of the dataset')

print('No Frauds', round(test_df['Class'].value_counts()[0]/len(test_df) * 100,2), '% of the dataset')
print('Frauds', round(test_df['Class'].value_counts()[1]/len(test_df) * 100,2), '% of the dataset')

# %%
# scale amount and time
rob_scaler = RobustScaler()

df['scaled_amount'] = rob_scaler.fit_transform(df['Amount'].values.reshape(-1,1))
df['scaled_time'] = rob_scaler.fit_transform(df['Time'].values.reshape(-1,1))

df.drop(['Time','Amount'], axis=1, inplace=True)

test_df['scaled_amount'] = rob_scaler.fit_transform(test_df['Amount'].values.reshape(-1,1))
test_df['scaled_time'] = rob_scaler.fit_transform(test_df['Time'].values.reshape(-1,1))

test_df.drop(['Time','Amount'], axis=1, inplace=True)

# %%
# random undersampling to 1:2
df = df.sample(frac=1)

fraud_df = df.loc[df['Class'] == 1]
non_fraud_df = df.loc[df['Class'] == 0][:788]

normal_distributed_df = pd.concat([fraud_df, non_fraud_df])
undersampling_df = normal_distributed_df.sample(frac=1, random_state=42)

undersampling_df

# %%
# 篩選相關係數大於0.05的
correlation = undersampling_df.corr()['Class'].abs()
selected_features = correlation[correlation >= 0.05].index.tolist()
len(selected_features)

# %%
import seaborn as sns
import matplotlib.pyplot as plt

# 假設 correlation 是包含特徵與目標變量之間相關性的 Series 或 DataFrame

# 繪製相關性熱力圖
plt.figure(figsize=(10, 8))
sns.heatmap(correlation.to_frame(), annot=True, cmap='coolwarm', fmt='.2f', cbar=False)
plt.title('Correlation between features and Class')
plt.xlabel('Features')
plt.ylabel('Class')
plt.show()


# %%
# remove outlier on significant features
# # -----> V14 Removing Outliers (Highest Negative Correlated with Labels)
v14_fraud = undersampling_df['V14'].loc[undersampling_df['Class'] == 1].values
q25, q75 = np.percentile(v14_fraud, 25), np.percentile(v14_fraud, 75)
print('Quartile 25: {} | Quartile 75: {}'.format(q25, q75))
v14_iqr = q75 - q25
print('iqr: {}'.format(v14_iqr))
v14_cut_off = v14_iqr * 1.5
v14_lower, v14_upper = q25 - v14_cut_off, q75 + v14_cut_off
print('Cut Off: {}'.format(v14_cut_off))
print('V14 Lower: {}'.format(v14_lower))
print('V14 Upper: {}'.format(v14_upper))

outliers = [x for x in v14_fraud if x < v14_lower or x > v14_upper]
print('Feature V14 Outliers for Fraud Cases: {}'.format(len(outliers)))
print('V10 outliers:{}'.format(outliers))

undersampling_df = undersampling_df.drop(undersampling_df[(undersampling_df['V14'] > v14_upper) | (undersampling_df['V14'] < v14_lower)].index)
print('----' * 40)

# -----> V12 removing outliers from fraud transactions
v12_fraud = undersampling_df['V12'].loc[undersampling_df['Class'] == 1].values
q25, q75 = np.percentile(v12_fraud, 25), np.percentile(v12_fraud, 75)
v12_iqr = q75 - q25

v12_cut_off = v12_iqr * 1.5
v12_lower, v12_upper = q25 - v12_cut_off, q75 + v12_cut_off
print('V12 Lower: {}'.format(v12_lower))
print('V12 Upper: {}'.format(v12_upper))
outliers = [x for x in v12_fraud if x < v12_lower or x > v12_upper]
print('V12 outliers: {}'.format(outliers))
print('Feature V12 Outliers for Fraud Cases: {}'.format(len(outliers)))
undersampling_df = undersampling_df.drop(undersampling_df[(undersampling_df['V12'] > v12_upper) | (undersampling_df['V12'] < v12_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(undersampling_df)))
print('----' * 40)

# Removing outliers V10 Feature
v10_fraud = undersampling_df['V10'].loc[undersampling_df['Class'] == 1].values
q25, q75 = np.percentile(v10_fraud, 25), np.percentile(v10_fraud, 75)
v10_iqr = q75 - q25

v10_cut_off = v10_iqr * 1.5
v10_lower, v10_upper = q25 - v10_cut_off, q75 + v10_cut_off
print('V10 Lower: {}'.format(v10_lower))
print('V10 Upper: {}'.format(v10_upper))
outliers = [x for x in v10_fraud if x < v10_lower or x > v10_upper]
print('V10 outliers: {}'.format(outliers))
print('Feature V10 Outliers for Fraud Cases: {}'.format(len(outliers)))
undersampling_df = undersampling_df.drop(undersampling_df[(undersampling_df['V10'] > v10_upper) | (undersampling_df['V10'] < v10_lower)].index)
print('Number of Instances after outliers removal: {}'.format(len(undersampling_df)))

# %%
class FraudDetection(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetection, self).__init__()
        
        self.layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),  # Batch normalization after the first linear layer
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),  # Batch normalization after the second linear layer
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # Batch normalization after the second linear layer
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),  # Batch normalization after the second linear layer
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.BatchNorm1d(32),  # Batch normalization after the second linear layer
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.layer(x)
        return out

class FraudDetection2(nn.Module):
    def __init__(self, input_dim):
        super(FraudDetection2, self).__init__()
        
        self.initial_layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),  # Batch normalization after the first linear layer
            nn.ReLU()
        )
        
        # Transformer layers
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=64, nhead=4)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=2)
        
        self.final_layers = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),  # Batch normalization after the linear layer
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        out = self.initial_layers(x)
        
        # Reshape to fit transformer input (sequence length, batch size, input dimension)
        out = out.unsqueeze(0)
        
        # Apply transformer layers
        out = self.transformer(out)
        
        # Reshape back to the original shape
        out = out.squeeze(0)
        
        out = self.final_layers(out)
        return out


# %%
input_dim = len(selected_features)-1
lr = 0.001
# model = FraudDetection2(input_dim)
# loss_fuc = nn.BCELoss()
# optimizer = optim.AdamW(model.parameters(), lr=lr)

# Assuming the model is modified to output a 2-dimensional tensor
model = FraudDetection2(input_dim)  # Assuming the modified FraudDetection model outputs 2 dimensions
loss_func = nn.BCELoss()  # Cross Entropy Loss for multi-class classification
optimizer = optim.AdamW(model.parameters(), lr=lr)
# %%
X_train = undersampling_df[selected_features]
X_train = X_train.drop('Class', axis=1) 
y_train = undersampling_df['Class']
X_train = torch.tensor(X_train.values).float()
y_train = torch.tensor(y_train.values).float()

X_test = test_df[selected_features]
X_test = X_test.drop('Class', axis=1) 
y_test = test_df['Class']
X_test = torch.tensor(X_test.values).float()
y_test = torch.tensor(y_test.values).float()

# %%
# training
epochs = 60
batch_size = 32

for epoch in range(epochs):
    epoch_loss = 0.0
    num_batches = len(X_train) // batch_size

    for batch in range(num_batches):
        start = batch * batch_size
        end = start + batch_size

        # 前向传播
        outputs = model(X_train[start:end])
        loss = loss_func(outputs, y_train[start:end].view(-1, 1))

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # print(f"Epoch {epoch+1} Loss: {epoch_loss / num_batches}")

with torch.no_grad():
    model.eval()
    predictions = model(X_test)
    loss = loss_func(predictions, y_test.view(-1, 1))

    predicted_labels = (predictions > 0.5).float()

    confusion = confusion_matrix(y_test, predicted_labels)
    accuracy = accuracy_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)

print("Loss:", loss.item())
print("Confusion Matrix:")
print(confusion)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)
# 假设你已经有了混淆矩阵 `confusion`
labels = ['Non-Fraud', 'Fraud']

# 創建熱力圖
sns.heatmap(confusion, annot=True, cmap='Blues', fmt='d', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()

# %%
# 儲存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
}, 'models/model_undersampling.pth')

# %%
# load 模型
checkpoint = torch.load('models/model_undersampling.pth') # 改成你的模型名稱
model_test = FraudDetection(26)
adam = optim.Adam(model_test.parameters(), lr=lr)
model_test.load_state_dict(checkpoint['model_state_dict'])
adam.load_state_dict(checkpoint['optimizer_state_dict'])
# %%

with torch.no_grad():
    model_test.eval()
    predictions = model_test(X_test)
    loss = loss_func(predictions, y_test.view(-1, 1))

    predicted_labels = (predictions > 0.5).float()

    confusion = confusion_matrix(y_test, predicted_labels)
    accuracy = accuracy_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    precision = precision_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)
    
print("Loss:", loss.item())
print("Confusion Matrix:")
print(confusion)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 Score:", f1)