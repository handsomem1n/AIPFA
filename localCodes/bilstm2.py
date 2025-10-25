# -*- coding: utf-8 -*-
"""
BiLSTM 운동 분류 모델 학습 스크립트
8가지 운동 분류 지원: Squat, Lunge, Side Lunge, Situp, High Knees, Bridge, Cobra, Jumping Jack

Original file is located at
    https://colab.research.google.com/drive/1Xzoe7YDtRF3jrsRcxWEReSrnFCpjGMrR

## 지원하는 운동 클래스 (8개)
- Squat (스쿼트)
- Lunge (런지)
- Side Lunge (사이드런지)
- Situp (윗몸일으키기)
- High Knees (하이니즈)
- Bridge (브릿지)
- Cobra (코브라)
- Jumping Jack (점핑잭)

## Train / Test 데이터셋 분리
- 데이터셋에서 각 운동별 (01. 02번째)영상을 테스트 데이터셋으로 사용
"""

import pandas as pd
import re

# 예시: CSV 파일에서 로딩
df = pd.read_csv("/content/Dataset_prev.csv")  # 열 이름이 'video_path'라고 가정
# 또는 리스트라면: df = pd.DataFrame({'video_path': your_list})

def extract_test_train(df):
    test_indices = []

    # 운동 별 그룹화
    for exercise_name, group in df.groupby(df['video_path'].apply(lambda x: x.split('/')[0])):
        # 숫자 추출: _XXXX에서 숫자만 추출
        group = group.copy()
        group['vid_num'] = group['video_path'].apply(
            lambda x: int(re.findall(r'_(\d{4})', x)[-1]) if re.findall(r'_(\d{4})', x) else -1
        )

        # 1, 2번째 해당되는 인덱스 추출
        test_vids = group[group['vid_num'].isin([1, 2])]
        test_indices.extend(test_vids.index.tolist())

    test_df = df.loc[test_indices].reset_index(drop=True)
    train_df = df.drop(index=test_indices).reset_index(drop=True)
    return train_df, test_df

# 실행
train_df, test_df = extract_test_train(df)

# 결과 확인
print("Train 개수:", len(train_df))
print("Test 개수:", len(test_df))

# 저장
train_df.to_csv("/content/train.csv", index=False)
test_df.to_csv("/content/test.csv", index=False)

"""## Train 데이터셋 확인"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# CSV 불러오기
df = pd.read_csv('/content/TrainData.csv')  # 또는 parquet, feather 등

# 그룹별로 나누기
group_df = df[['group_id', 'labels']].drop_duplicates()

# 운동별로 group_id를 train/val/test로 분할
train_ids, val_ids = train_test_split(group_df, test_size=0.2, stratify=group_df['labels'], random_state=0)

# group_id 기준으로 원본 df 필터링
train_df = df[df['group_id'].isin(train_ids['group_id'])]
val_df   = df[df['group_id'].isin(val_ids['group_id'])]




print("\n--- 데이터 개수 비교 ---")
print(f"원본 train 개수: {len(train_df)}")


# 데이터 준비
train_set = PoseSequenceDataset(train_df)
val_set = PoseSequenceDataset(val_df)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

"""# 모델 정의 및 학습"""

# --- 1. 라이브러리 ---
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.preprocessing import LabelEncoder
import joblib

# --- 2. 하이퍼파라미터 ---
BATCH_SIZE = 4
EPOCHS = 60
LR = 1e-3
HIDDEN_SIZE = 64
INPUT_SIZE = 132  # 33 joints × (x, y, z, v)

# --- 3. 커스텀 Dataset ---
class PoseSequenceDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.data, self.labels = [], []
        df['label_encoded'] = label_encoder.transform(df['labels'])
        self.label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        for group_id in df['group_id'].unique():
            group = df[df['group_id'] == group_id].sort_values('frame_id')
            pose_seq = group.loc[:, 'x0':'v32'].values
            self.data.append(torch.tensor(pose_seq, dtype=torch.float32))
            self.labels.append(torch.tensor(group['label_encoded'].iloc[0]))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), lengths

# --- 4. 모델 정의 ---
class PoseBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        h = torch.cat((h_fwd, h_bwd), dim=1)
        return self.fc(h)

le = LabelEncoder()
le.fit(train_df["labels"])
joblib.dump(le, "/content/label_encoder.pkl")  # 저장

train_set = PoseSequenceDataset(train_df, label_encoder=le)
val_set   = PoseSequenceDataset(val_df, label_encoder=le)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader   = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 6. 학습 루프 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PoseBiLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes=len(le.classes_)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for x, y, lengths in train_loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        output = model(x, lengths)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for x, y, lengths in val_loader:
            x, y = x.to(device), y.to(device)
            output = model(x, lengths)
            loss = criterion(output, y)
            total_val_loss += loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    print(f"[Epoch {epoch+1}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# --- 7. 저장 ---
torch.save(model.state_dict(), "/content/model.pt")
print("모델 저장 완료: /content/model.pt")

"""# 평가"""

# --- 1. 라이브러리 ---
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.multiclass import unique_labels

# --- 2. 하이퍼파라미터 ---
BATCH_SIZE = 4
HIDDEN_SIZE = 64
INPUT_SIZE = 132

# --- 3. Dataset & 모델 정의 (학습 코드와 동일하게 유지) ---
class PoseSequenceDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.data, self.labels = [], []
        df['label_encoded'] = label_encoder.transform(df['labels'])
        self.label_map = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
        for group_id in df['group_id'].unique():
            group = df[df['group_id'] == group_id].sort_values('frame_id')
            pose_seq = group.loc[:, 'x0':'v32'].values
            self.data.append(torch.tensor(pose_seq, dtype=torch.float32))
            self.labels.append(torch.tensor(group['label_encoded'].iloc[0]))

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(seq) for seq in sequences])
    padded = pad_sequence(sequences, batch_first=True)
    return padded, torch.tensor(labels), lengths

class PoseBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True, dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        h = torch.cat((h_fwd, h_bwd), dim=1)
        return self.fc(h)

# --- 4. 데이터 불러오기 ---
le = joblib.load("/content/label_encoder.pkl")
test_df = pd.read_csv("/content/TestData.csv")

test_set = PoseSequenceDataset(test_df, label_encoder=le)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# --- 5. 모델 로딩 ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PoseBiLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes=len(le.classes_)).to(device)
model.load_state_dict(torch.load("/content/model.pt"))
model.eval()

# --- 6. 평가 ---
criterion = nn.CrossEntropyLoss()
total_loss = 0
all_preds, all_labels = [], []

with torch.no_grad():
    for x, y, lengths in test_loader:
        x, y = x.to(device), y.to(device)
        output = model(x, lengths)
        loss = criterion(output, y)
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

avg_loss = total_loss / len(test_loader)
acc = sum([p == y for p, y in zip(all_preds, all_labels)]) / len(all_labels)
print(f"[Test] Loss: {avg_loss:.4f} | Accuracy: {acc:.4f}")

# # --- 7. Confusion Matrix ---
# plt.figure(figsize=(7, 6))
# cm = confusion_matrix(all_labels, all_preds)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#             xticklabels=le.classes_,
#             yticklabels=le.classes_)
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.title("Confusion Matrix")
# plt.show()

# --- 8. Classification Report ---
used_labels = unique_labels(all_labels, all_preds)
target_names = [le.classes_[i] for i in used_labels]

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds, labels=used_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names,
            yticklabels=target_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()

"""#Train, Val Loss시각화"""

# 학습 완료 후 손실 시각화
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training & Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

import pandas as pd
import torch
import torch.nn as nn
import joblib
from torch.nn.utils.rnn import pack_padded_sequence

# --- 하이퍼파라미터 ---
BATCH_SIZE = 1
HIDDEN_SIZE = 64
INPUT_SIZE = 132

# --- 모델 정의 ---
class PoseBiLSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, bidirectional=True,
                            dropout=0.3, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x, lengths):
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        h_fwd, h_bwd = h_n[-2], h_n[-1]
        h = torch.cat((h_fwd, h_bwd), dim=1)
        return self.fc(h)

# --- 라벨 인코더 & 모델 로딩 ---
le = joblib.load("/content/label_encoder.pkl")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = PoseBiLSTMClassifier(INPUT_SIZE, HIDDEN_SIZE, num_classes=len(le.classes_)).to(device)
model.load_state_dict(torch.load("/content/model.pt"))
model.eval()

# --- 테스트 데이터 불러오기 ---
df = pd.read_csv("/content/man_lunge.csv")

# --- 시퀀스별 예측 ---
results = []

for seq_id in df['sequence'].unique():
    seq_df = df[df['sequence'] == seq_id].sort_values("frame_id")
    pose_seq = seq_df.loc[:, 'x0':'v32'].values
    pose_tensor = torch.tensor(pose_seq, dtype=torch.float32).unsqueeze(0).to(device)
    length_tensor = torch.tensor([len(pose_seq)]).to(device)

    with torch.no_grad():
        output = model(pose_tensor, length_tensor)
        pred_idx = torch.argmax(output, dim=1).item()
        pred_label = le.classes_[pred_idx]
        results.append({"sequence": seq_id, "predicted_label": pred_label})

# --- 결과 출력 ---
print("✅ 예측 결과:")
for r in results:
    print(f"  - 시퀀스 {r['sequence']}: {r['predicted_label']}")