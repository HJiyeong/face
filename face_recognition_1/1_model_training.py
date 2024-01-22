import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split




# JSON 파일 경로
json_file_path = "/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/landmark/landmark_datas"

# JSON 파일 읽기
with open(json_file_path, 'r') as json_file:
    # JSON 파일을 파이썬 딕셔너리로 변환
    data = json.load(json_file)

data = sorted(data, key=lambda x: x.get('label', '').casefold())
print(data)


import json
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split




# JSON 파일 경로
json_file_path = "/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/landmark/landmark_datas"

# JSON 파일 읽기
with open(json_file_path, 'r') as json_file:
    # JSON 파일을 파이썬 딕셔너리로 변환
    data = json.load(json_file)

data = sorted(data, key=lambda x: x.get('label', '').casefold())
print(data)



class LandmarkModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LandmarkModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# 랜드마크 좌표를 사용하는 데이터셋 정의
class LandmarkDataset(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        landmarks = torch.tensor(item['landmarks'], dtype=torch.float32)  # 리스트를 Tensor로 변환
        label = item['label']

        # 훈련 데이터인 경우
        if self.train:
            return {'landmarks': landmarks, 'label': label}
        # 테스트 데이터인 경우
        else:
            return {'landmarks': landmarks, 'label': label, 'image_path': item['image_path']}





# 훈련 데이터와 테스트 데이터로 분리

# dami에 대한 데이터 추출
dami_data = [d for d in data if 'dami' in d['label']]
train_data_dami, test_data_dami = train_test_split(dami_data, test_size=0.25, random_state=42)

# hyunbin에 대한 데이터 추출
hyunbin_data = [d for d in data if 'hyunbin' in d['label']]
train_data_hyunbin, test_data_hyunbin = train_test_split(hyunbin_data, test_size=0.25, random_state=42)

# IU에 대한 데이터 추출
IU_data = [d for d in data if 'IU' in d['label']]
train_data_IU, test_data_IU = train_test_split(IU_data, test_size=0.25, random_state=42)

# JK에 대한 데이터 추출
JK_data = [d for d in data if 'JK' in d['label']]
train_data_JK, test_data_JK = train_test_split(JK_data, test_size=0.25, random_state=42)

# karina에 대한 데이터 추출
karina_data = [d for d in data if 'karina' in d['label']]
train_data_karina, test_data_karina = train_test_split(karina_data, test_size=0.25, random_state=42)

# rose에 대한 데이터 추출
rose_data = [d for d in data if 'rose' in d['label']]
train_data_rose, test_data_rose = train_test_split(rose_data, test_size=0.25, random_state=42)



train_data = train_data_dami + train_data_hyunbin + train_data_IU + train_data_JK + train_data_karina + train_data_rose
test_data = test_data_dami + test_data_hyunbin + test_data_IU + test_data_JK + test_data_karina + test_data_rose

# 훈련 데이터셋 및 테스트 데이터셋 생성
train_dataset = LandmarkDataset(train_data, train=True)
test_dataset = LandmarkDataset(test_data, train=False)

# DataLoader 설정
batch_size = 16  # 배치 크기 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 레이블을 숫자로 매핑
labels = ['dami', 'hyunbin', 'IU', 'JK', 'karina', 'rose']

label_to_idx = {label: idx for idx, label in enumerate(labels)}
labels = torch.tensor([label_to_idx[re.sub('\d+\.png$', '', label)] for label in labels])


# 모델의 입력 차원을 랜드마크 포인트의 수로 설정
input_dim = 68 * 2  # 랜드마크 포인트가 68개이고 각각의 포인트는 x, y 좌표를 가지므로 input_dim = 68 * 2
output_dim = 6

model = LandmarkModel(input_dim=input_dim, output_dim=output_dim)



# 손실 함수 및 최적화 기법 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_loader:
        landmarks, labels = batch['landmarks'], batch['label']

        landmarks = landmarks.view(-1, input_dim)  # 모델의 입력 차원에 맞게 형태 변환
        outputs = model(landmarks)

        labels = torch.tensor([label_to_idx[re.sub('\d+\.png$', '', label)] for label in labels])  # 레이블을 숫자로 변환
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 에폭마다 손실 출력
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# 테스트
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_loader:
        landmarks, labels = batch['landmarks'], batch['label']

        landmarks = landmarks.view(-1, input_dim)  # 모델의 입력 차원에 맞게 형태 변환
        outputs = model(landmarks)

        labels = torch.tensor([label_to_idx[re.sub('\d+\.png$', '', label)] for label in labels])  # 레이블을 숫자로 변환
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)  # 여기에 total 설정 추가
        correct += (predicted == labels).sum().item()

        # 예측 결과 출력
        print("Ground truth:", labels)
        print("Predicted:", predicted)

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')





class LandmarkModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LandmarkModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, output_dim)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc4(x)
        return x


# 랜드마크 좌표를 사용하는 데이터셋 정의
class LandmarkDataset(Dataset):
    def __init__(self, data, train=True):
        self.data = data
        self.train = train

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        landmarks = torch.tensor(item['landmarks'], dtype=torch.float32)  # 리스트를 Tensor로 변환
        label = item['label']

        # 훈련 데이터인 경우
        if self.train:
            return {'landmarks': landmarks, 'label': label}
        # 테스트 데이터인 경우
        else:
            return {'landmarks': landmarks, 'label': label, 'image_path': item['image_path']}





# 훈련 데이터와 테스트 데이터로 분리

# dami에 대한 데이터 추출
dami_data = [d for d in data if 'dami' in d['label']]
train_data_dami, test_data_dami = train_test_split(dami_data, test_size=0.25, random_state=42)

# hyunbin에 대한 데이터 추출
hyunbin_data = [d for d in data if 'hyunbin' in d['label']]
train_data_hyunbin, test_data_hyunbin = train_test_split(hyunbin_data, test_size=0.25, random_state=42)

# IU에 대한 데이터 추출
IU_data = [d for d in data if 'IU' in d['label']]
train_data_IU, test_data_IU = train_test_split(IU_data, test_size=0.25, random_state=42)

# JK에 대한 데이터 추출
JK_data = [d for d in data if 'JK' in d['label']]
train_data_JK, test_data_JK = train_test_split(JK_data, test_size=0.25, random_state=42)

# karina에 대한 데이터 추출
karina_data = [d for d in data if 'karina' in d['label']]
train_data_karina, test_data_karina = train_test_split(karina_data, test_size=0.25, random_state=42)

# rose에 대한 데이터 추출
rose_data = [d for d in data if 'rose' in d['label']]
train_data_rose, test_data_rose = train_test_split(rose_data, test_size=0.25, random_state=42)



train_data = train_data_dami + train_data_hyunbin + train_data_IU + train_data_JK + train_data_karina + train_data_rose
test_data = test_data_dami + test_data_hyunbin + test_data_IU + test_data_JK + test_data_karina + test_data_rose

# 훈련 데이터셋 및 테스트 데이터셋 생성
train_dataset = LandmarkDataset(train_data, train=True)
test_dataset = LandmarkDataset(test_data, train=False)

# DataLoader 설정
batch_size = 16  # 배치 크기 설정
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 레이블을 숫자로 매핑
labels = ['dami', 'hyunbin', 'IU', 'JK', 'karina', 'rose']

label_to_idx = {label: idx for idx, label in enumerate(labels)}
labels = torch.tensor([label_to_idx[re.sub('\d+\.png$', '', label)] for label in labels])


# 모델의 입력 차원을 랜드마크 포인트의 수로 설정
input_dim = 68 * 2  # 랜드마크 포인트가 68개이고 각각의 포인트는 x, y 좌표를 가지므로 input_dim = 68 * 2
output_dim = 6

model = LandmarkModel(input_dim=input_dim, output_dim=output_dim)



# 손실 함수 및 최적화 기법 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 학습
num_epochs = 10
for epoch in range(num_epochs):
    for batch in train_loader:
        landmarks, labels = batch['landmarks'], batch['label']

        landmarks = landmarks.view(-1, input_dim)  # 모델의 입력 차원에 맞게 형태 변환
        outputs = model(landmarks)

        labels = torch.tensor([label_to_idx[re.sub('\d+\.png$', '', label)] for label in labels])  # 레이블을 숫자로 변환
        loss = criterion(outputs, labels)

        # 역전파 및 최적화
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 에폭마다 손실 출력
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')


# 테스트
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in test_loader:
        landmarks, labels = batch['landmarks'], batch['label']

        landmarks = landmarks.view(-1, input_dim)  # 모델의 입력 차원에 맞게 형태 변환
        outputs = model(landmarks)

        labels = torch.tensor([label_to_idx[re.sub('\d+\.png$', '', label)] for label in labels])  # 레이블을 숫자로 변환
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)  # 여기에 total 설정 추가
        correct += (predicted == labels).sum().item()

        # 예측 결과 출력
        print("Ground truth:", labels)
        print("Predicted:", predicted)

accuracy = correct / total
print(f'Test Accuracy: {accuracy * 100:.2f}%')




