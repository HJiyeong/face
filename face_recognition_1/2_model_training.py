import random
import numpy as np
import json
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn


def create_episode(dataset, n_classes=2, n_support=2, n_query=2):
    """
    dataset: 전체 데이터셋. {클래스: [샘플 리스트]} 형태의 딕셔너리여야 합니다.
    n_classes: 한 에피소드에서 사용할 클래스의 수.
    n_support: 각 클래스당 학습 데이터로 사용할 샘플의 수.
    n_query: 각 클래스당 테스트에 사용할 샘플의 수.
    """
    # 클래스들을 무작위로 선택
    class_list = list(dataset.keys())
    selected_classes = random.sample(class_list, n_classes)

    support_samples = []
    query_samples = []

    for class_ in selected_classes:
        samples = dataset[class_]

        # 한 클래스 내에서 샘플들을 무작위로 선택
        selected_samples = random.sample(samples, n_support + n_query)

        support_samples.extend([(class_, sample) for sample in selected_samples[:n_support]])
        query_samples.extend([(class_, sample) for sample in selected_samples[n_support:]])

    return support_samples, query_samples


# 임베딩 네트워크 정의
class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(2, 64),  # 각 랜드마크 좌표는 2차원입니다.
            nn.ReLU(),
            nn.Linear(64, 64)
        )

    def forward(self, x):
        return self.layers(x)


def load_data(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data


def convert_data(data):
    converted_data = {}
    for item in data:
        label = item['label'].split('.')[0]  # 'dami1.png' -> 'dami1'
        label = ''.join([i for i in label if not i.isdigit()])  # 'dami1' -> 'dami'
        if label not in converted_data:
            converted_data[label] = []
        converted_data[label].append(item['landmarks'])
    return converted_data


# JSON 파일 로드 및 데이터 변환
raw_data = load_data('/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/landmark/landmark_datas')
dataset = convert_data(raw_data)

for label, samples in list(dataset.items())[:5]:  # 첫 5개 클래스만 출력
    print(f'Label: {label}, Number of samples: {len(samples)}')

# 모델, 옵티마이저, 손실 함수 초기화
model = EmbeddingNet()
optimizer = Adam(model.parameters())
criterion = CrossEntropyLoss()

n_epochs = 50  # 전체 데이터셋에 대해 학습을 반복할 횟수
n_episodes = 100  # 각 에포크마다 생성할 에피소드의 수
n_classes = 2  # 한 에피소드에서 사용할 클래스의 수
n_support = 2  # 각 클래스당 학습 데이터로 사용할 샘플의 수
n_query = 2  # 각 클래스당 테스트에 사용할 샘플의 수

# 레이블 인코딩
label_encoder = {label: i for i, label in enumerate(dataset.keys())}

# 레이블 인코딩의 역매핑 생성
label_decoder = {i: label for label, i in label_encoder.items()}

for epoch in range(n_epochs):
    epoch_loss = 0.0
    epoch_acc = 0.0  # 추가: 에포크별 정확도를 저장할 변수
    for episode in range(n_episodes):
        train_samples, test_samples = create_episode(dataset, n_classes, n_support, n_query)

        # 학습 데이터의 레이블과 특징 추출
        train_labels, train_landmarks = zip(*train_samples)
        train_labels = torch.tensor([label_encoder[label] for label in train_labels])
        train_landmarks = torch.tensor(train_landmarks, dtype=torch.float32).view(-1, len(train_landmarks[0]), 2)

        # 테스트 데이터의 레이블과 특징 추출
        test_labels, test_landmarks = zip(*test_samples)
        test_labels = torch.tensor([label_encoder[label] for label in test_labels])
        test_landmarks = torch.tensor(test_landmarks, dtype=torch.float32).view(-1, len(test_landmarks[0]), 2)

        # 모델을 학습 모드로 설정
        model.train()

        # 역전파 단계 전에, Optimizer 객체를 사용하여 갱신할 변수들에 대한 모든 변화도를 0으로 만듭니다.
        optimizer.zero_grad()

        # 순전파 단계: 모델에 x를 전달하여 f(x)의 출력을 계산합니다.
        train_embeddings = model(train_landmarks)
        train_embeddings = train_embeddings.mean(1)  # 랜드마크들의 평균을 계산

        # 손실 계산
        loss = criterion(train_embeddings, train_labels)

        # 역전파 단계: 모델의 매개변수에 대한 손실의 변화도를 계산합니다.
        loss.backward()

        # Optimizer의 step 함수를 호출하면 매개변수가 갱신됩니다.
        optimizer.step()

        epoch_loss += loss.item()

        # 모델을 평가 모드로 설정
        model.eval()

        # 순전파 단계: 모델에 x를 전달하여 f(x)의 출력을 계산합니다.
        test_embeddings = model(test_landmarks)
        test_embeddings = test_embeddings.mean(1)  # 랜드마크들의 평균을 계산

        # 소프트맥스 함수를 적용하여 확률을 얻습니다.
        test_probabilities = F.softmax(test_embeddings, dim=1)

        # 가장 확률이 높은 클래스를 예측값으로 사용합니다.
        _, test_predictions = test_probabilities.max(1)

        # 정확도를 계산합니다.
        test_accuracy = (test_predictions == test_labels).float().mean()

        epoch_acc += test_accuracy.item()

    print(f'Epoch {epoch + 1}, Loss: {epoch_loss / n_episodes}, Accuracy: {epoch_acc / n_episodes}')
