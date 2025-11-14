import os
import datetime
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, accuracy_score
from tqdm.auto import tqdm

# === Основные настройки ===
BASE_MODEL = "DeepPavlov/rubert-base-cased"  # Базовая русскоязычная модель BERT
SAVE_ROOT = "models"                         # Папка для сохранения моделей
BATCH_SIZE = 4                               # Размер батча
EPOCHS = 5                                   # Количество эпох обучения
MAX_LEN = 100                                # Максимальная длина текста (в токенах)
LR = 2e-5                                    # Скорость обучения
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Используем GPU, если доступен
os.makedirs(SAVE_ROOT, exist_ok=True)        # Создаём папку для сохранения, если её нет

print(f"Используется устройство: {DEVICE}")

# === Загрузка данных ===
print("Загрузка данных...")
data = pd.read_csv("data/incidents_fuul.csv")   # Загружаем CSV с инцидентами
data = data.dropna(subset=["urgency"])          # Удаляем строки без метки срочности
print(f"Загружено {len(data)} записей")

# === Кодирование категориальных признаков ===
print("Кодирование меток...")
le_category = LabelEncoder()
data["category_label"] = le_category.fit_transform(data["category"])  # Категории → числа
le_urgency = LabelEncoder()
data["urgency_label"] = le_urgency.fit_transform(data["urgency"])     # Срочность → числа

# Добавляем категорию в текст (data augmentation)
data["text_aug"] = "[категория: " + data["category"] + "] " + data["text"]

# === Разделение данных на train/val/test ===
print("Разделение данных...")
train_texts, temp_texts, train_cat, temp_cat, train_urg, temp_urg = train_test_split(
    data["text_aug"], data["category_label"], data["urgency_label"], 
    test_size=0.3, random_state=42, stratify=data["urgency_label"]
)
val_texts, test_texts, val_cat, test_cat, val_urg, test_urg = train_test_split(
    temp_texts, temp_cat, temp_urg, test_size=0.6667, random_state=42, stratify=temp_urg
)
print(f"Размеры данных: Train={len(train_texts)}, Val={len(val_texts)}, Test={len(test_texts)}")

# === Загрузка токенизатора ===
print("Загрузка токенизатора...")
tokenizer = BertTokenizer.from_pretrained(BASE_MODEL)

# === Определение класса Dataset ===
class IncidentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        # Токенизация текста
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt"
        )
        # Возвращаем словарь тензоров для модели
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Функция для создания DataLoader
def create_loader(texts, labels, shuffle=True):
    ds = IncidentDataset(texts.tolist(), labels.tolist(), tokenizer, MAX_LEN)
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)

# === Создание DataLoader для обеих задач ===
print("Создание DataLoader...")
train_cat_loader = create_loader(train_texts, train_cat)
val_cat_loader = create_loader(val_texts, val_cat, shuffle=False)
test_cat_loader = create_loader(test_texts, test_cat, shuffle=False)

train_urg_loader = create_loader(train_texts, train_urg)
val_urg_loader = create_loader(val_texts, val_urg, shuffle=False)
test_urg_loader = create_loader(test_texts, test_urg, shuffle=False)

# === Количество классов ===
num_cat = len(le_category.classes_)
num_urg = len(le_urgency.classes_)
print(f"Категорий: {num_cat}, Срочность: {num_urg}")

# === Загрузка моделей BERT для каждой задачи ===
print("Загрузка моделей...")
model_cat = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=num_cat).to(DEVICE)
model_urg = BertForSequenceClassification.from_pretrained(BASE_MODEL, num_labels=num_urg).to(DEVICE)

# === Балансировка классов для срочности ===
print("Вычисление весов классов...")
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_urg),
    y=train_urg
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(DEVICE)
loss_fn_urg = torch.nn.CrossEntropyLoss(weight=class_weights)  # Взвешенная функция потерь

# === Оптимизаторы и шедулеры ===
optimizer_cat = AdamW(model_cat.parameters(), lr=LR)
optimizer_urg = AdamW(model_urg.parameters(), lr=LR)

scheduler_cat = get_linear_schedule_with_warmup(
    optimizer_cat, num_warmup_steps=0, num_training_steps=len(train_cat_loader)*EPOCHS
)
scheduler_urg = get_linear_schedule_with_warmup(
    optimizer_urg, num_warmup_steps=0, num_training_steps=len(train_urg_loader)*EPOCHS
)

# === Функция оценки модели ===
def eval_model(model, loader, desc="Evaluation"):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(batch["labels"].numpy())
    acc = accuracy_score(labels, preds)
    return acc, labels, preds

# === Функция обучения одной эпохи ===
def train_epoch(model, optimizer, scheduler, loader, loss_fn=None, desc="Training"):
    model.train()
    total_loss = 0
    progress_bar = tqdm(loader, desc=desc, leave=False)
    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)
        # Если есть кастомная функция потерь — используем её
        if loss_fn is None:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.logits, labels)
        # Шаг оптимизации
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
    return total_loss / len(loader)

# === Основной цикл обучения ===
print("Начало обучения...")
history = {'epoch': [], 'cat_loss': [], 'urg_loss': [], 'val_cat_acc': [], 'val_urg_acc': []}

for epoch in range(EPOCHS):
    print(f"\n{'='*50}")
    print(f"ЭПОХА {epoch+1}/{EPOCHS}")
    print(f"{'='*50}")
    # Обучаем модель для категорий
    avg_cat_loss = train_epoch(
        model_cat, optimizer_cat, scheduler_cat, train_cat_loader, desc=f"Категории [Эпоха {epoch+1}]"
    )
    # Обучаем модель для срочности (с весами классов)
    avg_urg_loss = train_epoch(
        model_urg, optimizer_urg, scheduler_urg, train_urg_loader, loss_fn_urg, desc=f"Срочность [Эпоха {epoch+1}]"
    )
    # Проверяем точность на валидации
    cat_acc, cat_labels, cat_preds = eval_model(model_cat, val_cat_loader, desc=f"Валидация категорий")
    urg_acc, urg_labels, urg_preds = eval_model(model_urg, val_urg_loader, desc=f"Валидация срочности")
    # Сохраняем результаты
    history['epoch'].append(epoch + 1)
    history['cat_loss'].append(avg_cat_loss)
    history['urg_loss'].append(avg_urg_loss)
    history['val_cat_acc'].append(cat_acc)
    history['val_urg_acc'].append(urg_acc)
    print(f"Категории: loss={avg_cat_loss:.4f}, val_acc={cat_acc:.3f}")
    print(f"Срочность: loss={avg_urg_loss:.4f}, val_acc={urg_acc:.3f}")

# === Финальное тестирование ===
print(f"\n{'='*50}")
print("Финальная оценка на тестовых данных")
print(f"{'='*50}")
cat_acc, cat_labels, cat_preds = eval_model(model_cat, test_cat_loader, desc="📊 Тест категорий")
urg_acc, urg_labels, urg_preds = eval_model(model_urg, test_urg_loader, desc="📊 Тест срочности")
print(f"Категории (тест): acc={cat_acc:.3f}")
print(f"Срочность (тест): acc={urg_acc:.3f}")

# === Отчёты по качеству ===
print(f"\n{'='*50}")
print("ИТОГОВЫЕ ОТЧЁТЫ")
print(f"{'='*50}")
print("\n=== Категории ===")
print(classification_report(cat_labels, cat_preds, target_names=le_category.classes_))
print("\n=== Срочность ===")
print(classification_report(urg_labels, urg_preds, target_names=le_urgency.classes_))

# === Сохранение моделей и результатов ===
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
save_dir = f"{SAVE_ROOT}/dual_rubert_{ts}"
os.makedirs(save_dir, exist_ok=True)
print(f"\nСохранение моделей в: {save_dir}")

try:
    # Сохраняем модели
    print("Сохранение модели категорий...")
    model_cat.save_pretrained(f"{save_dir}/category_model")
    print("Сохранение модели срочности...")
    model_urg.save_pretrained(f"{save_dir}/urgency_model")
    print("Сохранение токенизатора...")
    tokenizer.save_pretrained(save_dir)
    # Сохраняем кодировщики LabelEncoder
    print("Сохранение кодировщиков...")
    pd.to_pickle(le_category, f"{save_dir}/le_category.pkl")
    pd.to_pickle(le_urgency, f"{save_dir}/le_urgency.pkl")
    # Сохраняем историю обучения
    print("Сохранение истории обучения...")
    history_df = pd.DataFrame(history)
    history_df.to_csv(f"{save_dir}/training_history.csv", index=False)
    # Сохраняем метаданные об обучении
    print("Сохранение параметров обучения...")
    training_info = {
        'base_model': BASE_MODEL,
        'batch_size': BATCH_SIZE,
        'epochs': EPOCHS,
        'max_len': MAX_LEN,
        'learning_rate': LR,
        'device': str(DEVICE),
        'timestamp': ts,
        'test_accuracy_category': cat_acc,
        'test_accuracy_urgency': urg_acc,
        'num_categories': num_cat,
        'num_urgency_levels': num_urg
    }
    info_df = pd.DataFrame([training_info])
    info_df.to_csv(f"{save_dir}/training_info.csv", index=False)
    # Выводим результаты
    print(f"Все файлы успешно сохранены в: {save_dir}")
    print(f"Итоговые результаты:")
    print(f"   • Категории: {cat_acc:.3f}")
    print(f"   • Срочность: {urg_acc:.3f}")
    # Вывод содержимого папки
    print(f"\nСодержимое папки {save_dir}:")
    for file in os.listdir(save_dir):
        file_path = os.path.join(save_dir, file)
        if os.path.isdir(file_path):
            print(f"   {file}/")
        else:
            size = os.path.getsize(file_path)
            print(f"    {file} ({size} bytes)")
except Exception as e:
    print(f"Ошибка при сохранении: {e}")
