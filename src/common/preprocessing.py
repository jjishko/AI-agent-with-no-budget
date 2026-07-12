import random
from pathlib import Path
from collections import defaultdict
from src.common.normalizer import NERNormalizer
from src.common.misc import load_data, save_data

OUTPUT_DIR = "../../data/processed"

def get_entity_type_combo(item):
    types = set()
    if 'label' in item:
        for label in item['label']:
            if 'labels' in label:
                for lbl in label['labels']:
                    types.add(lbl.lower())
    return tuple(sorted(types))


def entity_stratified_split(data, train_ratio=0.8, seed=42):
    random.seed(seed)

    # Группируем по комбинации типов
    groups = defaultdict(list)
    for item in data:
        combo = get_entity_type_combo(item)
        groups[combo].append(item)

    train_data = []
    test_data = []

    for combo, items in groups.items():
        # Перемешиваем внутри группы
        shuffled = items.copy()
        random.shuffle(shuffled)

        n_total = len(items)
        n_test = max(1, int(n_total * (1 - train_ratio)))

        # Если в группе только 1 элемент, отдаем в train
        if n_total <= 1:
            train_data.extend(items)

        else:
            # Если группа маленькая (2-3 элемента), берем 1 в test
            if n_total <= 3:
                n_test = 1

            train_data.extend(shuffled[:-n_test])
            test_data.extend(shuffled[-n_test:])

    # Перемешиваем финальные выборки
    random.shuffle(train_data)
    random.shuffle(test_data)

    return train_data, test_data


def entity_comparation(train_data, test_data):
    print("\nСравнение распределения сущностей в Train vs Test:")
    train_counts = defaultdict(int)
    test_counts = defaultdict(int)

    for item in train_data:
        if 'label' in item:
            for label in item['label']:
                if 'labels' in label:
                    for lbl in label['labels']:
                        train_counts[lbl] += 1

    for item in test_data:
        if 'label' in item:
            for label in item['label']:
                if 'labels' in label:
                    for lbl in label['labels']:
                        test_counts[lbl] += 1

    print(f"{'Тип':<20} {'Train':<10} {'Test':<10} {'Всего':<10}")
    print("-" * 50)

    all_types = set(train_counts.keys()) | set(test_counts.keys())
    for entity_type in sorted(all_types):
        train_c = train_counts.get(entity_type, 0)
        test_c = test_counts.get(entity_type, 0)
        total = train_c + test_c
        print(f"{entity_type:<20} {train_c:<10} {test_c:<10} {total:<10}")


def ner_prerocessing(verbose=False):
    input_file = "../../data/annotated/NER_annotated.json"
    train_file = "NER_train.json"
    test_file = "NER_test.json"
    train_ratio = 0.8

    # 1. Загружаем данные
    data = load_data(input_file)

    # 2. Нормализуем данные
    normalizer = NERNormalizer()
    normalized_data = normalizer.normalize_dataset(data)

    # 3. Стратифицированное разделение
    train_data, test_data = entity_stratified_split(normalized_data, train_ratio)

    # 4. Сохраняем результаты
    train_path = Path(OUTPUT_DIR) / train_file
    test_path = Path(OUTPUT_DIR) / test_file

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    save_data(train_data, train_path)
    save_data(test_data, test_path)

    if verbose:
        print(f"Загружено {len(data)} элементов")
        print(f"\nРезультат разделения:")
        print(f"   Train: {len(train_data)} элементов")
        print(f"   Test: {len(test_data)} элементов")
        print(f"   Test доля: {len(test_data) / (len(train_data) + len(test_data)):.2%}")
        print(f"\nСохранено:")
        print(f"   - Train: {len(train_data)} элементов")
        print(f"   - Test: {len(test_data)} элементов")

        # 5. Сравнение распределения
        entity_comparation(train_data, test_data)


if __name__ == "__main__":
    ner_prerocessing(verbose=True)