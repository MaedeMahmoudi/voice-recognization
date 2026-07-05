import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


with open('processed_data.pkl', 'rb') as f:
    data = pickle.load(f)

x = data['features']
y = data['labels']

print(f"total samples: {len(x)}")

#padding
#یعنی بیایم صفر یا یه مقدار مشخص به انتهای ارایه اضافه کنیم تا طول ها ثابت بشن 
lengths = [len(seq) for seq in x]
MAX_LEN = int(np.percentile(lengths, 95))
print(f"max length for padding: {MAX_LEN}")


def pad_sequence(seq, max_len=MAX_LEN):
    if len(seq) >= max_len:
        return seq[:max_len]
    else:
        pad_size = max_len - len(seq)
        return np.pad(seq, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)

x_padded = np.array([pad_sequence(seq) for seq in x])
print(f"padded shape: {x_padded.shape}")


# باعث میشه نسبت داده های تست به هر رقم یکسان باشه stratifyتقسیم داده با 
x_train, x_test, y_train, y_test = train_test_split(
    x_padded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"train data: {x_train.shape}")
print(f"test data: {x_test.shape}")

 
# نرمالایز کردن هر نمونه جداگانه
def normalize_sample(sample):
    mean = np.mean(sample, axis=0)
    std = np.std(sample, axis=0)
    return (sample - mean) / (std + 1e-8)

x_train_norm = np.array([normalize_sample(sample) for sample in x_train])
x_test_norm = np.array([normalize_sample(sample) for sample in x_test])

print(f"x_train_norm shape: {x_train_norm.shape}")
print(f"x_test_norm shape: {x_test_norm.shape}")


class VoiceDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

train_dataset = VoiceDataset(x_train_norm, y_train)
test_dataset = VoiceDataset(x_test_norm, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


class DigitLSTM(nn.Module):
    def __init__(self, input_size=39, hidden_size=64, num_layers=2, num_classes=10):
        super(DigitLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                            batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        return self.fc(out)

model = DigitLSTM()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


print(f"model parameters {sum(p.numel() for p in model.parameters()):,}")


#train
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005)  # نرخ یادگیری کمتر
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10, factor=0.5)


num_epochs = 80

best_accuracy = 0
patience_counter = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    #evaluation on test data
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

    accuracy = 100 * correct / total
    
    # کاهش نرخ یادگیری اگر دقت بالا نرفته
    scheduler.step(accuracy)
    
    # Early Stopping
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        patience_counter = 0
        torch.save(model.state_dict(), 'best_digit_model.pth')
    else:
        patience_counter += 1
    
    if (epoch + 1) % 5 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}, Test Accuracy = {accuracy:.2f}%, LR = {current_lr:.6f}")
    
    # اگر ۲۰ دوره پشت سر هم بهبود نداشت، متوقف کن
    if patience_counter >= 20:
        print(f"\n Early stopping at epoch {epoch+1}")
        break


model.load_state_dict(torch.load('best_digit_model.pth'))


#evaluation
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

final_acc = 100 * correct / total
print(f"\n accuracy: {final_acc:.2f}%")