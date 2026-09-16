"""
Сборка объединённого датасета Multiple Sclerosis MRI
Источники:
  1. buraktaci/multiple-sclerosis (MS Axial + Control Axial)
  2. omarradi1993/mri-scans-brain-neurological-classes (MS + Normal)
  3. buraktaci (сагиттальные срезы — только для теста)
  4. trainingdatapro/multiple-sclerosis-dataset (доп. MS, только JPEG)
"""

import os
import shutil
import splitfolders

# Пути
BURAKTACI_DIR = "dataset/MS"                    # уже скачан
OMAR_DIR = "dataset/mri_scans"                  # уже скачан
MS_EXTRA_DIR = "dataset/ms_extra"               # новый: trainingdatapro MS
RAW_DIR = "dataset/raw_all"
OUTPUT_DIR = "dataset/processed"

# Очищаем и создаём папки
if os.path.exists(RAW_DIR):
    shutil.rmtree(RAW_DIR)
os.makedirs(os.path.join(RAW_DIR, "MS"), exist_ok=True)
os.makedirs(os.path.join(RAW_DIR, "Normal"), exist_ok=True)

count_ms = 0
count_normal = 0

def copy_files(src_dir, dst_dir, label):
    """Копирует файлы изображений из папки (без вложенных подпапок)"""
    global count_ms, count_normal
    if not os.path.exists(src_dir):
        print(f"  ⚠️ Папка не найдена: {src_dir}")
        return
    files = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
    for f in files:
        shutil.copy2(os.path.join(src_dir, f), os.path.join(dst_dir, f))
    if label == "MS":
        count_ms += len(files)
    else:
        count_normal += len(files)
    print(f"  ✓ {label}: {len(files)} файлов")

def copy_files_recursive(root_dir, dst_dir, label):
    """Рекурсивно ищет JPEG во всех подпапках и копирует их"""
    global count_ms, count_normal
    if not os.path.exists(root_dir):
        print(f"  ⚠️ Папка не найдена: {root_dir}")
        return
    found = 0
    for dirpath, _, filenames in os.walk(root_dir):
        for f in filenames:
            if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                # Добавляем имя родительской папки как префикс чтобы избежать конфликтов
                parent = os.path.basename(dirpath)
                new_name = f"extra_{parent}_{f}"
                shutil.copy2(os.path.join(dirpath, f), os.path.join(dst_dir, new_name))
                found += 1
                if label == "MS":
                    count_ms += 1
                else:
                    count_normal += 1
    print(f"  ✓ {label}: {found} файлов (рекурсивно)")

# ============ Источник 1: buraktaci (аксиальные) ============
print("Источник 1: buraktaci (аксиальные срезы)")
copy_files(os.path.join(BURAKTACI_DIR, "MS Axial_crop"),
           os.path.join(RAW_DIR, "MS"), "MS")
copy_files(os.path.join(BURAKTACI_DIR, "Control Axial_crop"),
           os.path.join(RAW_DIR, "Normal"), "Normal")

# ============ Источник 2: omarradi1993 ============
print("\nИсточник 2: omarradi1993 (MRI Scans Neurological Classes)")
copy_files(os.path.join(OMAR_DIR, "MS"),
           os.path.join(RAW_DIR, "MS"), "MS")
copy_files(os.path.join(OMAR_DIR, "Normal"),
           os.path.join(RAW_DIR, "Normal"), "Normal")

# ============ Источник 4: trainingdatapro (доп. MS) ============
print("\nИсточник 4: trainingdatapro MS (рекурсивный поиск JPEG)")
copy_files_recursive(MS_EXTRA_DIR, os.path.join(RAW_DIR, "MS"), "MS")

# ============ Источник 3: сагиттальные (ТОЛЬКО в тест) ============
print("\nИсточник 3: сагиттальные срезы buraktaci (будут добавлены в тест отдельно)")
sagittal_ms_dir = os.path.join(BURAKTACI_DIR, "MS Saggital_crop")
sagittal_normal_dir = os.path.join(BURAKTACI_DIR, "Control Saggital_crop")

sagittal_ms = len([f for f in os.listdir(sagittal_ms_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(sagittal_ms_dir) else 0
sagittal_normal = len([f for f in os.listdir(sagittal_normal_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]) if os.path.exists(sagittal_normal_dir) else 0
print(f"  MS Saggital: {sagittal_ms} файлов")
print(f"  Normal Saggital: {sagittal_normal} файлов")

# ============ Статистика перед разбивкой ============
print(f"\n{'='*50}")
print(f"ВСЕГО ПЕРЕД РАЗБИВКОЙ:")
print(f"  MS: {count_ms}")
print(f"  Normal: {count_normal}")
print(f"  Итого: {count_ms + count_normal}")

# ============ Разбивка train/val/test ============
print(f"\nРазбиваю на train/val/test (80/10/10)...")
splitfolders.ratio(
    RAW_DIR,
    output=OUTPUT_DIR,
    seed=42,
    ratio=(0.8, 0.1, 0.1)
)

# ============ Добавляем сагиттальные в тест ============
print(f"\nДобавляю сагиттальные срезы в тестовую выборку...")
for sag_dir, label in [(sagittal_ms_dir, "MS"), (sagittal_normal_dir, "Normal")]:
    if os.path.exists(sag_dir):
        dst = os.path.join(OUTPUT_DIR, "test", label)
        files = [f for f in os.listdir(sag_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for f in files:
            new_name = f"sag_{f}"
            shutil.copy2(os.path.join(sag_dir, f), os.path.join(dst, new_name))
        print(f"  ✓ Sagittal {label}: {len(files)} файлов → test/{label}/")

# ============ Финальная статистика ============
print(f"\n{'='*50}")
print(f"ФИНАЛЬНЫЙ ДАТАСЕТ:")
for split in ['train', 'val', 'test']:
    ms = len(os.listdir(os.path.join(OUTPUT_DIR, split, "MS")))
    normal = len(os.listdir(os.path.join(OUTPUT_DIR, split, "Normal")))
    print(f"  {split.upper()}: MS={ms}, Normal={normal}, Всего={ms+normal}")

print(f"\nГотово! Данные в {OUTPUT_DIR}")