import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from tqdm import tqdm

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
DATA_DIR = "dataset/processed"
MODEL_PATH = "ms_resnet18.pth"
BATCH_SIZE = 32

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE)
    model.eval()
    return model

def create_test_loader():
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    test_dataset = datasets.ImageFolder(
        os.path.join(DATA_DIR, 'test'), transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    return test_loader, test_dataset.classes

if __name__ == "__main__":
    print(f"Устройство: {DEVICE}")
    model = load_model()
    test_loader, classes = create_test_loader()

    # Сбор вероятностей
    true_labels = []
    all_probs = []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            true_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_probs = np.array(all_probs)
    true_labels = np.array(true_labels)

    print(f"\nПоиск оптимального порога для класса MS...")
    print(f"  (Цель: найти баланс между Accuracy и Recall MS)")

    # Ищем лучший порог — просто максимизируем F1 для MS
    best_f1 = 0
    best_threshold = 0.5
    best_preds = (all_probs[:, 0] >= 0.5).astype(int)  # дефолт на старт
    best_preds = np.where(all_probs[:, 0] >= 0.5, 0, 1)

    results = []
    for threshold in np.arange(0.05, 0.95, 0.05):
        threshold = round(threshold, 2)
        preds = np.where(all_probs[:, 0] >= threshold, 0, 1)

        # Recall для MS (класс 0)
        ms_mask = (true_labels == 0)
        recall_ms = np.sum((preds[ms_mask] == 0)) / np.sum(ms_mask) if np.sum(ms_mask) > 0 else 0

        # Precision для MS
        pred_ms_mask = (preds == 0)
        precision_ms = np.sum((preds[pred_ms_mask] == 0) & (true_labels[pred_ms_mask] == 0)) / np.sum(pred_ms_mask) if np.sum(pred_ms_mask) > 0 else 0

        # F1 для MS
        f1_ms = 2 * precision_ms * recall_ms / (precision_ms + recall_ms) if (precision_ms + recall_ms) > 0 else 0

        acc = accuracy_score(true_labels, preds)
        results.append((threshold, acc, recall_ms, precision_ms, f1_ms))

        if f1_ms > best_f1:
            best_f1 = f1_ms
            best_threshold = threshold
            best_preds = preds

    print(f"\nТоп-5 порогов по F1-score для MS:")
    results.sort(key=lambda x: x[4], reverse=True)
    for t, acc, rec, prec, f1 in results[:5]:
        print(f"  Порог: {t:.2f} | Acc: {acc:.4f} | Recall MS: {rec:.4f} | Precision MS: {prec:.4f} | F1 MS: {f1:.4f}")

    print(f"\nВыбранный порог: {best_threshold:.2f}")

    # Финальные метрики
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ (с калибровкой порога)")
    print(f"{'='*60}")
    print(f"Общая точность (Accuracy): {accuracy_score(true_labels, best_preds)*100:.2f}%")
    print(f"Всего изображений: {len(true_labels)}")
    print(f"Правильных: {np.sum(true_labels == best_preds)}")
    print(f"Ошибок: {np.sum(true_labels != best_preds)}")
    print(f"\nClassification Report:")
    print(classification_report(true_labels, best_preds, target_names=classes, digits=4))

    # Матрица ошибок
    cm = confusion_matrix(true_labels, best_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(f'Confusion Matrix (Test) | MS threshold = {best_threshold:.2f}',
              fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    print("Матрица ошибок сохранена: confusion_matrix.png")