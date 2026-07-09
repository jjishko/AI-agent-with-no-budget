import json
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import numpy as np
import re

plots_dir = "../../reports/plots/EDA/"

# Загрузка данных
def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


# 1. Базовая информация о датасете
def basic_info(data):
    print("БАЗОВАЯ ИНФОРМАЦИЯ")
    print(f"Всего записей: {len(data)}")

    # Проверка наличия текста
    texts = [item.get('text', '') for item in data]
    empty_texts = sum(1 for t in texts if not t.strip())
    print(f"Пустых текстов: {empty_texts}")

    # Проверка наличия разметки
    no_labels = sum(1 for item in data if not item.get('label', []))
    print(f"Записей без разметки: {no_labels}")

    # Фильтрация: оставляем только записи с разметкой
    filtered_data = [item for item in data if item.get('label', [])]
    print(f"Записей с разметкой: {len(filtered_data)}")

    return filtered_data, texts


# 2. Анализ длины текстов
def text_length_analysis(texts):
    print("\nАНАЛИЗ ДЛИНЫ ТЕКСТОВ")

    lengths = [len(text) for text in texts if text]

    if not lengths:
        print("Нет текстов для анализа")
        return [], []

    print(f"Минимальная длина (символов): {min(lengths)}")
    print(f"Максимальная длина (символов): {max(lengths)}")
    print(f"Средняя длина (символов): {np.mean(lengths):.2f}")
    print(f"Медианная длина (символов): {np.median(lengths):.2f}")

    # Количество слов
    word_counts = [len(re.findall(r'\w+', text)) for text in texts if text]
    print(f"\nСреднее количество слов: {np.mean(word_counts):.2f}")
    print(f"Медианное количество слов: {np.median(word_counts):.2f}")

    # Гистограмма длин
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.hist(lengths, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Длина текста (символы)')
    plt.ylabel('Количество записей')
    plt.title('Распределение длины текстов')

    plt.subplot(1, 2, 2)
    plt.hist(word_counts, bins=30, edgecolor='black', alpha=0.7, color='green')
    plt.xlabel('Количество слов')
    plt.ylabel('Количество записей')
    plt.title('Распределение количества слов')

    plt.tight_layout()
    plt.savefig(plots_dir + "NER_text_analysis.png")
    plt.show()
    return lengths, word_counts


# 3. Анализ сущностей
def entity_analysis(data, all_entity_types):
    print("\nАНАЛИЗ СУЩНОСТЕЙ")

    # Сбор всех сущностей
    entity_counts = Counter()
    entities_per_text = []

    for item in data:
        text_entities = []
        for label in item.get('label', []):
            for entity in label.get('labels', []):
                entity_counts[entity] += 1
                text_entities.append(entity)
        entities_per_text.append(len(set(text_entities)))

    # Статистика по типам сущностей
    found_types = set(entity_counts.keys())
    missing_types = set(all_entity_types) - found_types

    print(f"Найдено типов сущностей: {len(found_types)} из {len(all_entity_types)}")
    print(f"Найденные типы: {sorted(found_types)}")

    if missing_types:
        print(f"Отсутствуют в данных: {sorted(missing_types)}")

    # Распределение количества сущностей
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    entity_dist = Counter(entities_per_text)
    plt.bar(entity_dist.keys(), entity_dist.values(), edgecolor='black', alpha=0.7)
    plt.xlabel('Количество уникальных сущностей в тексте')
    plt.ylabel('Количество записей')
    plt.title('Распределение сущностей по текстам')

    # График распределения частоты сущностей с подписями
    plt.subplot(1, 2, 2)
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    entities = [e[0] for e in sorted_entities]
    frequencies = [e[1] for e in sorted_entities]

    bars = plt.bar(entities, frequencies, edgecolor='black', alpha=0.7, color='orange')
    plt.xlabel('Тип сущности')
    plt.ylabel('Количество вхождений')
    plt.title('Распределение частоты сущностей')
    plt.xticks(rotation=45, ha='right')

    for bar, freq in zip(bars, frequencies):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(freq)}',
                 ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir + "entity_analysis.png")
    plt.show()

    return entity_counts, entities_per_text


# 4. Соотношение размеченных токенов к общему числу токенов
def token_coverage_analysis(data):
    print("\nСООТНОШЕНИЕ РАЗМЕЧЕННЫХ ТОКЕНОВ")

    total_tokens = 0
    labeled_tokens = 0

    for item in data:
        text = item.get('text', '')
        # Токенизация по словам
        tokens = re.findall(r'\w+', text)
        total_tokens += len(tokens)

        # Подсчет размеченных токенов
        labeled_positions = set()
        for label in item.get('label', []):
            start = label.get('start', 0)
            end = label.get('end', 0)
            # Находим позиции начала каждого слова в размеченном фрагменте
            if start < end and start < len(text):
                pos = start
                while pos < end and pos < len(text):
                    # Нашли начало слова
                    word_start = pos
                    # Ищем конец слова
                    while pos < len(text) and pos < end:
                        pos += 1
                    # Добавляем позицию начала слова как размеченный токен
                    labeled_positions.add(word_start)

        labeled_tokens += len(labeled_positions)

    print(f"Всего токенов: {total_tokens}")
    print(f"Размеченных токенов: {labeled_tokens}")
    if total_tokens > 0:
        print(f"Соотношение размеченных токенов: {labeled_tokens / total_tokens * 100:.2f}%")
    else:
        print("Нет токенов для анализа")


# 5. OOV Rate (Out-of-Vocabulary Rate)
def oov_rate_analysis(data):
    print("\nАНАЛИЗ OOV RATE")

    # Сбор всей лексики
    all_words = []
    for item in data:
        text = item.get('text', '')
        words = re.findall(r'\w+', text.lower())
        all_words.extend(words)

    # Частотный словарь
    word_freq = Counter(all_words)
    total_words = len(all_words)
    unique_words = len(word_freq)

    print(f"Всего слов в корпусе: {total_words}")
    print(f"Уникальных слов: {unique_words}")
    print(f"Type-Token Ratio: {unique_words / total_words:.4f}")

    # Анализ редких слов
    singletons = sum(1 for count in word_freq.values() if count == 1)
    print(f"Слов, встречающихся 1 раз: {singletons} ({singletons / unique_words * 100:.2f}% от уникальных)")

    rare_words = sum(1 for count in word_freq.values() if 2 <= count <= 5)
    print(f"Слов, встречающихся 2-5 раз: {rare_words} ({rare_words / unique_words * 100:.2f}% от уникальных)")

    # Топ-20 самых частотных слов (сортировка по возрастанию для графика)
    top_words = word_freq.most_common(20)
    top_words_reversed = list(reversed(top_words))
    words, counts = zip(*top_words_reversed)

    # Визуализация только топ-20
    plt.figure(figsize=(10, 6))
    plt.barh(words, counts, color='skyblue')
    plt.xlabel('Частота')
    plt.title('Топ-20 самых частотных слов')

    # Подписываем значения
    for i, (word, count) in enumerate(top_words_reversed):
        plt.text(count + 1, i, str(count), va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(plots_dir + "top_word_freq.png")
    plt.show()

    return word_freq


# 6. Совместная встречаемость сущностей
def cooccurrence_analysis(data):
    print("\nАНАЛИЗ СОВМЕСТНОЙ ВСТРЕЧАЕМОСТИ СУЩНОСТЕЙ")

    entity_pairs = defaultdict(int)

    for item in data:
        entities_in_text = set()
        for label in item.get('label', []):
            for entity in label.get('labels', []):
                entities_in_text.add(entity)

        entities_list = list(entities_in_text)
        for i in range(len(entities_list)):
            for j in range(i + 1, len(entities_list)):
                pair = tuple(sorted([entities_list[i], entities_list[j]]))
                entity_pairs[pair] += 1

    if entity_pairs:
        print("Топ-10 наиболее частых пар сущностей в одном тексте:")
        for (e1, e2), count in sorted(entity_pairs.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {e1} + {e2}: {count} раз")
    else:
        print("Нет пар сущностей для анализа")

    return entity_pairs

# 7. Анализ длины сущностей
def entity_length_analysis(data):
    entity_lengths = defaultdict(list)

    for item in data:
        for label in item.get('label', []):
            length = label.get('end', 0) - label.get('start', 0)
            for entity in label.get('labels', []):
                entity_lengths[entity].append(length)

    if not entity_lengths:
        print("Нет сущностей для анализа")
        return {}

    plt.figure(figsize=(12, 6))

    sorted_entities = sorted(entity_lengths.keys())
    data_to_plot = [entity_lengths[entity] for entity in sorted_entities]

    plt.boxplot(data_to_plot)

    plt.xticks(range(1, len(sorted_entities) + 1), sorted_entities, rotation=45, ha='right')

    plt.title('Распределение длины сущностей')
    plt.ylabel('Длина (символы)')
    plt.xlabel('Тип сущности')
    plt.tight_layout()
    plt.savefig(plots_dir + "entity_boxplot.png")
    plt.show()

    return entity_lengths

# 8. Примеры текстов с редкими сущностями
def show_examples(data, entity_counts):
    print("\nПРИМЕРЫ ТЕКСТОВ С РАЗЛИЧНЫМИ СУЩНОСТЯМИ")

    rare_th = 20
    rare_entities = [e for e, c in entity_counts.most_common() if c < rare_th]

    if rare_entities:
        print(f"Редкие сущности (менее {rare_th} примеров): {rare_entities}")

        for entity in rare_entities[:3]:
            examples = []
            for item in data:
                for label in item.get('label', []):
                    if entity in label.get('labels', []):
                        examples.append(item.get('text', ''))
                        break
                if len(examples) >= 2:
                    break

            if examples:
                print(f"\nПримеры для '{entity}':")
                for i, ex in enumerate(examples[:2], 1):
                    print(f"  {i}. {ex}")


# Основная функция
def run_eda(file_path, all_entity_types):
    # Загрузка данных
    data = load_data(file_path)

    # Проверка, что data - список
    if not isinstance(data, list):
        data = [data]

    # Фильтрация записей с разметкой
    filtered_data, texts = basic_info(data)

    # Пропускаем анализ, если нет данных с разметкой
    if not filtered_data:
        print("Нет записей с разметкой для дальнейшего анализа")
        return

    # Анализ
    text_length_analysis(texts)
    entity_counts, entities_per_text = entity_analysis(filtered_data, all_entity_types)
    token_coverage_analysis(filtered_data)
    oov_rate_analysis(filtered_data)
    cooccurrence_analysis(filtered_data)
    entity_length_analysis(filtered_data)
    show_examples(filtered_data, entity_counts)


# Список всех возможных сущностей
ALL_ENTITIES = [
    'unit', 'equipment', 'tag', 'parameter', 'symptom',
    'document_type', 'vendor', 'model', 'unit_of_measure',
    'limit', 'time_range', 'action', 'quantity'
]

# Путь к файлу
file_path = '../../data/annotated/NER_annotated.json'
run_eda(file_path, ALL_ENTITIES)