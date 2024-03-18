import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from PIL import Image
import os
import numpy as np


class ResNetFingerprintExtractor(nn.Module):
    def __init__(self, num_devices):
        super(ResNetFingerprintExtractor, self).__init__()
        resnet = models.resnet101(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
        self.fc = nn.Linear(2048, num_devices) # detach fc from the pipeline

    def forward(self, x):
        features = self.resnet(x)
        features = features.view(features.size(0), -1)
        device_scores = self.fc(features)
        return device_scores


def downsample_image(image_path):
    img = Image.open(image_path)
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    img = transform(img)
    return img


def generate_prnu_free_images_downsampling(image_paths):
    prnu_free_images = []
    for path in image_paths:
        prnu_free_images.append(downsample_image(path))
    return torch.stack(prnu_free_images)


def generate_prnu_free_images_random(image_paths):
    prnu_free_images = []
    for path in image_paths:
        img = Image.open(path)
        transform = transforms.Compose(
            [transforms.RandomCrop((224, 224)), transforms.ToTensor()]
        )
        img = transform(img)
        prnu_free_images.append(img)
    return torch.stack(prnu_free_images)


def train_svm_classifier(features, labels):
    clf = SVC(kernel="rbf", gamma="scale")
    clf.fit(features, labels)
    return clf


def main():
    image_paths = os.listdir("../test/data/nat-jpg")
    labels = [
        "Nikon_D70_0",
        "Nikon_D70_1",
        "Nikon_D70s_0",
        "Nikon_D70s_1",
        "Nikon_D200_0",
        "Nikon_D200_1",
    ]
    prnu_free_images_downsampling = generate_prnu_free_images_downsampling(image_paths)
    prnu_free_images_random = generate_prnu_free_images_random(image_paths)

    num_devices = len(set(labels))
    device_fingerprint_extractor = ResNetFingerprintExtractor(num_devices)

    optimizer = torch.optim.Adam(device_fingerprint_extractor.parameters(), lr=0.001)
    softmax = nn.CrossEntropyLoss()
    num_epochs = 10
    batch_size = 12
    for epoch in range(num_epochs):
        for i in range(0, len(image_paths), batch_size):
            batch_images = prnu_free_images_downsampling[i : i + batch_size]
            batch_labels = torch.tensor(labels[i : i + batch_size])

            optimizer.zero_grad()
            outputs = device_fingerprint_extractor(batch_images)
            outputs = device_fingerprint_extractor.forward(outputs)
            loss = softmax(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

    features_downsampling = (
        device_fingerprint_extractor(prnu_free_images_downsampling).detach().numpy()
    )
    features_random = (
        device_fingerprint_extractor(prnu_free_images_random).detach().numpy()
    )

    svm_classifier_downsampling = train_svm_classifier(features_downsampling, labels)
    svm_classifier_random = train_svm_classifier(features_random, labels)

    predicted_labels_downsampling = svm_classifier_downsampling.predict(
        features_downsampling
    )
    predicted_labels_random = svm_classifier_random.predict(features_random)

    accuracy_downsampling = accuracy_score(labels, predicted_labels_downsampling)
    accuracy_random = accuracy_score(labels, predicted_labels_random)

    print(f"downsampling acc: {accuracy_downsampling}")
    print(f"random acc: {accuracy_random}")


if __name__ == "__main__":
    main()
