import pandas as pd
from collections import Counter
import re

# Загружаем CSV-файл
df = pd.read_csv("data/incidents_fuul.csv")

# Общая информация
print("Общая информация:")
print(df.info())
print("\nКоличество записей:", len(df))

# Первые 10 записей
print("\nПервые 10 записей:")
print(df.head(10))

# Распределение по категориям и срочности
print("\nРаспределение по категориям:")
print(df['category'].value_counts())

print("\nРаспределение по уровню срочности:")
print(df['urgency'].value_counts())

# Комбинация категории и срочности
combo_count = df.groupby(['category', 'urgency']).size().reset_index(name='Count')
print("\nКомбинация категории и срочности:")
print(combo_count)

# Длина текста
df['text_length'] = df['text'].apply(len)
print("\nСтатистика по длине текста:")
print(df['text_length'].describe())

# Топ 5 длинных инцидентов
print("\nТоп 5 длинных инцидентов:")
print(df.sort_values(by='text_length', ascending=False).head(5)[['text', 'text_length']])

# Часто встречающиеся слова
all_text = ' '.join(df['text'].astype(str).tolist())
# Простая очистка: только слова
words = re.findall(r'\b\w+\b', all_text.lower())
common_words = Counter(words).most_common(20)
print("\nТоп 20 часто встречающихся слов в описаниях:")
for word, count in common_words:
    print(f"{word}: {count}")

# Сохраняем комбинированный анализ категории и срочности
combo_count.to_csv("category_urgency_analysis.csv", index=False)

print("\nАнализ сохранён в category_urgency_analysis.csv")
