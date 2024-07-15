import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
import random
import json
import os
import torch.nn.functional as F

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print(device)



class EmbeddingNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pretrained_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        for param in self.pretrained_model.parameters():
            param.requires_grad = False

        # Replace the last fully connected layer
        num_ftrs = self.pretrained_model.fc.in_features
        self.pretrained_model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Dropout(0.2)
        )

        # Set the fully connected layer parameters to be trainable
        for param in self.pretrained_model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.pretrained_model(x)
        return x


def pairwise_euclidean_distance(a, b):
    return torch.sqrt(((a[:, None] - b[None, :]) ** 2).sum(-1))


def create_episode(dataset, n_classes=3, n_support=2, n_query=2):
    attempts = 10  # 시도 횟수 제한을 둬서 무한 루프 방지
    for attempt in range(attempts):
        valid_classes = [class_ for class_ in dataset.keys() if len(dataset[class_]) >= n_support + n_query]

        if len(valid_classes) >= n_classes:
            selected_classes = random.sample(valid_classes, n_classes)
            class_label_mapping = {class_: i for i, class_ in enumerate(selected_classes)}
            support_samples, query_samples = [], []

            for class_ in selected_classes:
                selected_samples = random.sample(dataset[class_], n_support + n_query)
                support_samples.extend(
                    [(class_label_mapping[class_], sample) for sample in selected_samples[:n_support]])
                query_samples.extend([(class_label_mapping[class_], sample) for sample in selected_samples[n_support:]])

            return support_samples, query_samples, class_label_mapping
        else:
            # 클래스 수를 줄여서 다시 시도
            n_classes -= 1
            print(f"Not enough valid classes found. Reducing n_classes to {n_classes} and retrying.")

    raise ValueError("Failed to create an episode with sufficient samples after multiple attempts.")


def load_data(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)


def convert_data(data):
    converted_data = {}
    for item in data:
        if 'person_name' not in item :
            label = ''.join([i for i in item['label'].split('.')[0] if not i.isdigit()])
        else :
            label = item['person_name']

        converted_data.setdefault(label, []).append(item['landmarks'])
    return converted_data





def estimate_original_size(dataset):
    all_landmarks = [landmark for data in dataset for landmark in data]
    all_landmarks = torch.tensor(all_landmarks)
    max_x, max_y = all_landmarks.max(dim=0)[0]
    min_x, min_y = all_landmarks.min(dim=0)[0]
    original_size = max(max_x - min_x, max_y - min_y).item()
    return original_size


def landmarks_to_image(landmarks, image_size=224, original_size=1000):
    image = torch.zeros(3, image_size, image_size)  # Initialize a blank image
    scale = image_size / original_size  # Scale factor
    for x, y in landmarks:
        x = int(x * scale)
        y = int(y * scale)
        if 0 <= x < image_size and 0 <= y < image_size:
            image[:, y, x] = 1.0  # Mark the landmark point

            # Apply Gaussian blur to the marked point to reduce density issues
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= x + dx < image_size and 0 <= y + dy < image_size:
                        image[:, y + dy, x + dx] += 0.5
    return image


class PrototypicalLoss(nn.Module):
    def __init__(self):
        super(PrototypicalLoss, self).__init__()

    def forward(self, embeddings, labels, n_classes, n_support, n_query):
        # Extract support and query samples
        support_embeddings = embeddings[:n_classes * n_support]
        query_embeddings = embeddings[n_classes * n_support:]
        support_labels = labels[:n_classes * n_support]
        query_labels = labels[n_classes * n_support:]

        # Compute class prototypes
        prototypes = torch.stack([support_embeddings[support_labels == i].mean(0) for i in range(n_classes)])

        # Compute distances from query samples to prototypes
        distances = pairwise_euclidean_distance(query_embeddings, prototypes)

        # Compute log probabilities
        log_p_y = F.log_softmax(-distances, dim=1)

        # Compute the loss
        loss = -log_p_y.gather(1, query_labels.view(-1, 1)).mean()

        # Compute accuracy
        _, y_hat = log_p_y.max(1)
        acc = (y_hat == query_labels).float().mean()

        return loss, acc


def train_model(model, optimizer, criterion, dataset, n_epochs, n_episodes, n_classes, n_support, n_query, original_size):
    model.to(device)
    for epoch in range(n_epochs):
        epoch_loss, epoch_acc = 0.0, 0.0
        for episode in range(n_episodes):
            try:
                train_samples, test_samples, class_label_mapping = create_episode(dataset, n_classes, n_support, n_query)
            except ValueError as e:
                print(f"Skipping episode {episode} due to: {e}")
                continue

            samples = train_samples + test_samples
            labels = torch.tensor([label for label, _ in samples]).to(device)
            landmarks = torch.stack(
                [landmarks_to_image(landmarks, original_size=original_size) for _, landmarks in samples]).to(device)

            model.train()
            optimizer.zero_grad()
            embeddings = model(landmarks)
            loss, acc = criterion(embeddings, labels, n_classes, n_support, n_query)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc.item()

        print(f'Epoch {epoch + 1}, Loss: {epoch_loss / n_episodes}, Accuracy: {epoch_acc / n_episodes}')


def predict(model, dataset, new_data, n_classes=3, n_support=2, n_query=2, original_size=1000):
    support_samples, _, class_label_mapping = create_episode(dataset, n_classes, n_support, n_query)
    support_labels = torch.tensor([label for label, _ in support_samples]).to(device)
    support_landmarks = torch.stack([landmarks_to_image(landmarks, original_size=original_size) for _, landmarks in support_samples]).to(device)

    model.eval()
    support_embeddings = model(support_landmarks)
    centroids = torch.stack([support_embeddings[support_labels == i].mean(0) for i in range(n_classes)])

    new_landmarks = torch.stack([landmarks_to_image(new_data, original_size=original_size)]).to(device)
    embeddings = model(new_landmarks)

    if embeddings.dim() == 1:
        embeddings = embeddings.unsqueeze(0)
    if centroids.dim() == 1:
        centroids = centroids.unsqueeze(0)

    class_scores = -torch.cdist(embeddings, centroids)
    _, predictions = class_scores.max(1)

    reverse_class_label_mapping = {i: class_ for class_, i in class_label_mapping.items()}
    predicted_class_name = reverse_class_label_mapping[predictions.item()]

    return predicted_class_name


if __name__ == "__main__":
    lpw_data_path = '/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/output_lpw_landmarks.json'
    custom_data_path = '/Users/hwangjiyeong/PycharmProjects/facial reconition/face_recognition_1/landmark/landmark_datas'

    # Load and convert LPW dataset
    lpw_raw_data = load_data(lpw_data_path)
    lpw_dataset = convert_data(lpw_raw_data)
    print(f'Total number of classes in LPW dataset: {len(lpw_dataset)}')

    # Estimate the original size of the images
    estimated_original_size = estimate_original_size([item['landmarks'] for item in lpw_raw_data])
    print(f'Estimated original size: {estimated_original_size}')

    model = EmbeddingNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = PrototypicalLoss()
    train_model(model, optimizer, criterion, lpw_dataset, n_epochs=500, n_episodes=100, n_classes=3, n_support=2,
                n_query=2, original_size=estimated_original_size)

    # Load and convert custom dataset
    custom_raw_data = load_data(custom_data_path)
    custom_dataset = convert_data(custom_raw_data)
    print(f'Total number of classes in custom dataset: {len(custom_dataset)}')

    # Fine-tune the model on custom dataset
    custom_optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for fine-tuning
    train_model(model, custom_optimizer, criterion, custom_dataset, n_epochs=100, n_episodes=50, n_classes=3,
                n_support=2, n_query=2, original_size=estimated_original_size)

    sample_data = {
        'dami': [[178, 334], [181, 376], [190, 418], [200, 459], [215, 497], [239, 527], [272, 551], [310, 569],
                 [351, 572], [392, 562], [430, 542], [463, 517], [486, 484], [498, 440], [505, 393], [507, 345],
                 [505, 299], [182, 273], [194, 248], [218, 237], [245, 236], [270, 246], [337, 235], [366, 218],
                 [399, 213], [430, 219], [456, 237], [308, 295], [309, 326], [310, 356], [311, 386], [288, 416],
                 [303, 419], [319, 421], [336, 414], [352, 408], [208, 314], [224, 300], [246, 300], [269, 316],
                 [248, 324], [224, 325], [360, 302], [377, 281], [402, 278], [426, 287], [407, 300], [382, 305],
                 [278, 483], [293, 464], [312, 451], [325, 453], [340, 447], [366, 455], [396, 466], [371, 486],
                 [348, 495], [332, 499], [318, 500], [297, 496], [288, 481], [314, 472], [328, 471], [342, 468],
                 [385, 468], [344, 469], [329, 472], [315, 473]],
        'hyunbin': [[213, 228], [218, 261], [223, 294], [232, 326], [248, 354], [273, 376], [304, 393], [337, 404],
                    [370, 406], [398, 401], [417, 382], [432, 359], [442, 333], [449, 305], [454, 279], [456, 251],
                    [453, 223], [262, 196], [280, 181], [303, 175], [327, 178], [349, 186], [379, 188], [395, 180],
                    [413, 179], [430, 182], [442, 195], [367, 211], [371, 227], [375, 244], [379, 261], [351, 283],
                    [363, 285], [374, 287], [382, 285], [390, 282], [291, 216], [304, 208], [318, 208], [329, 217],
                    [317, 219], [303, 219], [391, 218], [403, 208], [417, 208], [427, 215], [417, 219], [404, 219],
                    [322, 329], [342, 317], [360, 311], [372, 313], [382, 310], [393, 316], [402, 328], [395, 337],
                    [384, 340], [373, 340], [361, 340], [343, 337], [329, 328], [360, 325], [372, 325], [382, 324],
                    [397, 328], [383, 324], [373, 325], [361, 325]],
        'IU': [[192, 273], [186, 322], [182, 373], [187, 420], [205, 464], [233, 504], [268, 539], [307, 564],
               [346, 576], [382, 574], [416, 554], [448, 527], [475, 497], [499, 462], [517, 425], [529, 386],
               [533, 345], [265, 243], [298, 232], [334, 234], [367, 244], [399, 258], [450, 272], [476, 268],
               [501, 272], [523, 279], [536, 298], [416, 312], [413, 341], [411, 370], [409, 400], [366, 417],
               [381, 424], [395, 431], [408, 431], [420, 428], [297, 288], [322, 279], [346, 285], [358, 310],
               [339, 307], [315, 300], [449, 328], [472, 312], [494, 314], [505, 332], [492, 337], [470, 334],
               [310, 458], [344, 456], [371, 456], [385, 463], [399, 461], [414, 470], [426, 482], [407, 498],
               [390, 502], [374, 501], [359, 496], [336, 482], [319, 461], [367, 473], [382, 477], [396, 477],
               [417, 482], [395, 479], [380, 478], [365, 473]],
        'rose': [[244, 228], [247, 269], [254, 309], [263, 347], [278, 383], [300, 417], [325, 447], [354, 470],
                 [386, 477], [419, 470], [450, 448], [477, 418], [501, 384], [517, 347], [527, 308], [535, 269],
                 [539, 228], [266, 215], [288, 203], [315, 202], [341, 208], [366, 217], [412, 217], [437, 207],
                 [463, 203], [489, 203], [510, 215], [389, 248], [389, 274], [388, 301], [388, 328], [363, 348],
                 [375, 351], [388, 355], [400, 351], [413, 348], [296, 246], [313, 238], [333, 240], [348, 254],
                 [330, 257], [310, 256], [428, 253], [443, 239], [463, 238], [481, 246], [465, 256], [445, 257],
                 [341, 393], [358, 382], [375, 375], [388, 381], [400, 376], [416, 383], [432, 395], [416, 412],
                 [400, 421], [386, 423], [372, 421], [355, 412], [350, 393], [374, 393], [387, 395], [400, 394],
                 [423, 395], [399, 395], [386, 397], [373, 396]],
        'JK': [[218, 263], [223, 296], [232, 328], [243, 360], [256, 391], [275, 418], [301, 439], [333, 452],
               [371, 452], [408, 444], [442, 426], [471, 401], [491, 372], [501, 338], [505, 304], [505, 268],
               [504, 233], [228, 227], [243, 207], [268, 198], [295, 197], [319, 204], [353, 194], [380, 184],
               [409, 180], [439, 186], [463, 203], [342, 228], [344, 250], [345, 273], [346, 296], [326, 321],
               [339, 324], [352, 326], [365, 321], [377, 316], [260, 245], [273, 234], [290, 232], [307, 242],
               [292, 247], [274, 249], [383, 234], [398, 220], [416, 218], [431, 225], [418, 233], [400, 235],
               [313, 372], [329, 360], [343, 351], [354, 353], [366, 348], [385, 353], [407, 360], [390, 376],
               [374, 384], [360, 387], [347, 387], [332, 383], [321, 371], [344, 364], [356, 364], [369, 361],
               [399, 361], [370, 365], [357, 368], [345, 368]],
        'karina': [[130, 302], [135, 350], [143, 396], [159, 437], [185, 474], [218, 506], [253, 534], [290, 554],
                   [325, 559], [355, 549], [380, 523], [402, 493], [421, 461], [437, 427], [449, 390], [457, 351],
                   [458, 312], [183, 275], [210, 259], [242, 254], [276, 258], [306, 270], [367, 268], [391, 256],
                   [418, 251], [442, 255], [457, 272], [338, 313], [340, 340], [343, 368], [345, 395], [309, 417],
                   [325, 421], [340, 424], [352, 420], [363, 416], [222, 314], [246, 305], [271, 305], [287, 323],
                   [268, 327], [243, 324], [373, 321], [393, 304], [415, 301], [430, 312], [418, 323], [395, 324],
                   [275, 469], [301, 457], [324, 450], [338, 454], [351, 449], [365, 455], [379, 465], [366, 485],
                   [353, 495], [338, 498], [323, 498], [300, 489], [286, 468], [323, 464], [337, 465], [351, 463],
                   [370, 466], [352, 473], [338, 475], [324, 475]],
    }

    for name, landmarks in sample_data.items():
        prediction = predict(model, custom_dataset, landmarks, original_size=estimated_original_size)
        print(f'Predicted class (GT = {name}) class: {prediction}')
