import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "ms_resnet18.pth"
DATA_DIR = "dataset/processed/test"
OUTPUT_DIR = "gradcam_results"
IMG_SIZE = 224

CLASSES = ['MS', 'Normal']

# Экспертные вероятности для дифференциального ряда
DIFFERENTIAL = {
    "MS_typical": {
        "main": ("Рассеянный склероз", 85),
        "alternatives": [
            ("Сосудистые изменения белого вещества", 10,
             "Рекомендована МР-ангиография для исключения"),
            ("Нейродегенеративные процессы", 5,
             "Рекомендовано нейропсихологическое тестирование при когнитивных жалобах")
        ]
    },
    "MS_atypical": {
        "main": ("Очаги в белом веществе (требуют дифференциации)", 50),
        "alternatives": [
            ("Рассеянный склероз", 30,
             "Картина не полностью типична, рекомендовано МРТ с контрастом"),
            ("Сосудистая энцефалопатия", 15,
             "Рекомендована МР-ангиография"),
            ("Нейродегенеративные изменения", 5,
             "Рекомендовано нейропсихологическое тестирование")
        ]
    }
}

RECOMMENDATIONS = {
    "Normal": [
        "Плановое наблюдение у невролога",
        "При неврологических жалобах — МР-ангиография",
        "При когнитивных жалобах — нейропсихологическое тестирование"
    ],
    "MS_typical": [
        "МРТ с контрастом для выявления активных очагов (приоритет: высокий)",
        "Консультация невролога в течение 2 недель (приоритет: высокий)",
        "МР-ангиография для исключения сосудистой патологии (приоритет: средний)",
        "Нейропсихологическое тестирование при когнитивных жалобах (приоритет: низкий)",
        "Повторное МРТ через 3-6 месяцев"
    ],
    "MS_atypical": [
        "МР-ангиография (приоритет: высокий)",
        "Консультация невролога (приоритет: высокий)",
        "МРТ с контрастом (приоритет: средний)",
        "Нейропсихологическое тестирование (приоритет: средний)",
        "Повторное МРТ через 1-3 месяца"
    ]
}

def load_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.fc.in_features, 2)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model = model.to(DEVICE).eval()
    return model

def analyze_lesions(grayscale_cam):
    """Анализ тепловой карты: процент поражения и категория"""
    hot_pixels = (grayscale_cam > 0.6).sum()
    total_pixels = grayscale_cam.size
    percentage = (hot_pixels / total_pixels) * 100

    if percentage < 2:
        category = "Минимальная"
        description = "Единичные мелкие очаги"
    elif percentage < 5:
        category = "Умеренная"
        description = "Множественные очаги в белом веществе"
    elif percentage < 10:
        category = "Выраженная"
        description = "Распространённое поражение белого вещества"
    else:
        category = "Тяжёлая"
        description = "Обширные зоны демиелинизации"

    return percentage, category, description

def get_atypical_flag(grayscale_cam):
    """Проверка на атипичность: очаги только по краям = возможно сосудистые"""
    h, w = grayscale_cam.shape
    center_region = grayscale_cam[h//4:3*h//4, w//4:3*w//4]
    periphery = grayscale_cam.copy()
    periphery[h//4:3*h//4, w//4:3*w//4] = 0

    if periphery.sum() > center_region.sum() * 2:
        return True
    return False

def get_slice_type(filename):
    """Определяет тип среза по имени файла"""
    if filename.startswith("sag_"):
        return "Сагиттальный"
    else:
        return "Аксиальный"

def draw_bounding_boxes(img_np, grayscale_cam, confidence=None, pred_class=None, threshold=0.4, min_area=40):
    # Бинаризация тепловой карты
    heatmap_binary = (grayscale_cam > threshold).astype(np.uint8) * 255

    # Морфологическое разделение слипшихся областей
    kernel = np.ones((3, 3), np.uint8)
    heatmap_binary = cv2.erode(heatmap_binary, kernel, iterations=1)
    heatmap_binary = cv2.dilate(heatmap_binary, kernel, iterations=1)

    # Поиск контуров
    contours, _ = cv2.findContours(heatmap_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Копия изображения для рисования
    img_with_boxes = (img_np * 255).astype(np.uint8).copy()
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_RGB2BGR)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_area:
            # Зелёный прямоугольник (без номера)
            cv2.rectangle(img_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Верхняя плашка с результатом и уверенностью (компактная)
    if pred_class is not None and confidence is not None:
        if pred_class == "MS":
            text = f"MS | {confidence:.1f}%"
            color = (0, 0, 255)  # Красный
        else:
            text = f"NORMAL | {confidence:.1f}%"
            color = (0, 255, 0)  # Зелёный

        # Тонкая плашка сверху
        overlay_box = img_with_boxes.copy()
        cv2.rectangle(overlay_box, (0, 0), (img_with_boxes.shape[1], 22), (0, 0, 0), -1)
        img_with_boxes = cv2.addWeighted(img_with_boxes, 0.75, overlay_box, 0.25, 0)

        # Мелкий шрифт
        cv2.putText(img_with_boxes, text, (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    # Обратно в RGB
    img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    return img_with_boxes

def generate_report(pred_class, confidence, percentage, category, description, is_atypical, filename, slice_type):
    """Генерация текстового заключения"""
    print(f"        ЗАКЛЮЧЕНИЕ НЕЙРОСЕТЕВОГО АНАЛИЗА МРТ")
    print(f"\nИНФОРМАЦИЯ О СНИМКЕ:")
    print(f"  ├── Файл: {filename}")
    print(f"  └── Тип среза: {slice_type}")
    print(f"\nДИАГНОСТИЧЕСКИЙ РЕЗУЛЬТАТ:")
    print(f"  ├── Основной результат: {pred_class}")
    print(f"  ├── Уверенность: {confidence:.1f}%")
    if pred_class == "MS":
        print(f"  └── Альтернатива (Normal): {100-confidence:.1f}%")

    if pred_class == "MS":
        print(f"\nАНАЛИЗ ОБЛАСТЕЙ ВНИМАНИЯ (Grad-CAM):")
        print(f"  ├── Процент активированных пикселей: {percentage:.1f}%")
        print(f"  ├── Категория: {category}")
        print(f"  ├── Описание: {description}")
        print(f"  └── Паттерн: {'Атипичный (периферический)' if is_atypical else 'Типичный для РС'}")

        diff_key = "MS_atypical" if is_atypical else "MS_typical"
        diff = DIFFERENTIAL[diff_key]

        print(f"\nДИФФЕРЕНЦИАЛЬНАЯ ОЦЕНКА:")
        print(f"  ├── {diff['main'][0]} — {diff['main'][1]}% (основной)")
        for i, (name, prob, note) in enumerate(diff['alternatives']):
            prefix = "├──" if i < len(diff['alternatives'])-1 else "└──"
            print(f"  {prefix} {name} — {prob}% ({note})")

        recs = RECOMMENDATIONS[diff_key]
    else:
        recs = RECOMMENDATIONS["Normal"]

    print(f"\nРЕКОМЕНДАЦИИ:")
    for i, rec in enumerate(recs):
        prefix = "├──" if i < len(recs)-1 else "└──"
        print(f"  {prefix} {rec}")


def visualize_gradcam(model, cam, img_path, true_class, output_dir):
    """Создание визуализации Grad-CAM + Bounding Boxes + полный анализ"""
    filename = os.path.basename(img_path)
    slice_type = get_slice_type(filename)

    img = Image.open(img_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.softmax(output, dim=1)[0].cpu().numpy()
        pred_idx = np.argmax(prob)
        pred_class = CLASSES[pred_idx]
        confidence = prob[pred_idx] * 100

    grayscale_cam = cam(input_tensor=img_tensor, targets=None)[0]
    img_resized = img.resize((IMG_SIZE, IMG_SIZE))
    img_np = np.array(img_resized) / 255.0
    overlay = show_cam_on_image(img_np, grayscale_cam, use_rgb=True, colormap=cv2.COLORMAP_JET)

    # Bounding boxes и анализ
    if pred_class == "MS":
        img_with_boxes = draw_bounding_boxes(img_np, grayscale_cam, confidence, pred_class)
        percentage, category, description = analyze_lesions(grayscale_cam)
        is_atypical = get_atypical_flag(grayscale_cam)
    else:
        img_with_boxes = None
        percentage, category, description, is_atypical = 0, "Не применимо", "", False

    # Визуализация
    if pred_class == "MS" and img_with_boxes is not None:
        # 4 панели для MS
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))

        axes[0].imshow(img_resized, cmap='gray')
        axes[0].set_title(f'Original [{slice_type}]\n{filename[:30]}...', fontsize=8, fontweight='bold')
        axes[0].axis('off')

        axes[1].imshow(grayscale_cam, cmap='jet')
        axes[1].set_title('Grad-CAM: Areas of interest', fontsize=10, fontweight='bold')
        axes[1].axis('off')

        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay ({confidence:.1f}%)', fontsize=10, fontweight='bold')
        axes[2].axis('off')

        axes[3].imshow(img_with_boxes)
        axes[3].set_title('Bounding Boxes', fontsize=10, fontweight='bold')
        axes[3].axis('off')
    else:
        # 1 панель для Normal
        img_with_text = (img_np * 255).astype(np.uint8).copy()
        img_with_text = cv2.cvtColor(img_with_text, cv2.COLOR_RGB2BGR)

        text = f"NORMAL | {confidence:.1f}%"
        overlay_box = img_with_text.copy()
        cv2.rectangle(overlay_box, (0, 0), (img_with_text.shape[1], 22), (0, 0, 0), -1)
        img_with_text = cv2.addWeighted(img_with_text, 0.75, overlay_box, 0.25, 0)
        cv2.putText(img_with_text, text, (4, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        img_with_text = cv2.cvtColor(img_with_text, cv2.COLOR_BGR2RGB)

        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        ax.imshow(img_with_text)
        ax.set_title(f'[{slice_type}] {filename[:40]}...\nResult: Normal', fontsize=8, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"gradcam_{filename}")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    generate_report(pred_class, confidence, percentage, category, description, is_atypical, filename, slice_type)

    return pred_class == true_class

if __name__ == "__main__":
    print(f"Устройство: {DEVICE}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    model = load_model()
    target_layers = [model.layer4[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)

    print("Модель загружена, Grad-CAM готов\n")

    samples_per_class = 5
    stats = {cls: {'correct': 0, 'total': 0} for cls in CLASSES}

    for class_name in CLASSES:
        class_dir = os.path.join(DATA_DIR, class_name)
        if not os.path.exists(class_dir):
            continue
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        selected = np.random.choice(images, size=min(samples_per_class, len(images)), replace=False)

        print(f"\n{'─'*60}")
        print(f"Класс: {class_name} ({len(selected)} изображений)")
        print(f"{'─'*60}")

        for img_name in selected:
            img_path = os.path.join(class_dir, img_name)
            try:
                is_correct = visualize_gradcam(model, cam, img_path, class_name, OUTPUT_DIR)
                stats[class_name]['total'] += 1
                if is_correct:
                    stats[class_name]['correct'] += 1
            except Exception as e:
                print(f"Ошибка: {img_name} — {e}")

    print(f"\n{'='*60}")
    print(f"СТАТИСТИКА GRAD-CAM")
    for cls in CLASSES:
        if stats[cls]['total'] > 0:
            acc = stats[cls]['correct'] / stats[cls]['total'] * 100
            print(f"  {cls}: {stats[cls]['correct']}/{stats[cls]['total']} ({acc:.1f}%)")
    print(f"Результаты сохранены в: {OUTPUT_DIR}/")