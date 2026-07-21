from collections import defaultdict
from typing import Dict, List
from datetime import datetime
import os


def make_ner_metrics(model, test_data: List[Dict]) -> Dict[str, float]:
    """Оценивает качество модели на тестовых данных"""

    if not test_data or 'label' not in test_data[0]:
        raise ValueError("Тестовые данные должны содержать поле 'label'")

    # Инициализация счетчиков для всего датасета
    tp_by_type = defaultdict(int)
    fp_by_type = defaultdict(int)
    fn_by_type = defaultdict(int)

    # Для Exact Match
    tp_exact = 0
    fp_exact = 0
    fn_exact = 0

    # Проходим по всем примерам в датасете
    for item in test_data:
        text = item['text']
        if 'label' not in item:
            gold_entities = []
        else:
            gold_entities = item['label']

        pred_result = model.predict([item])
        pred_entities = pred_result['label']

        gold_by_type = defaultdict(set)
        gold_exact_set = set()

        for entity in gold_entities:
            entity_type = entity.get('labels', [])[0] if entity.get('labels') else None
            value = entity.get('normalized', entity.get('text', ''))
            start = entity.get('start')
            end = entity.get('end')

            if entity_type and value:
                # Для обычных метрик - по типу и нормализованному значению
                gold_by_type[entity_type].add(value)

                # Для Exact Match - полное совпадение
                if start is not None and end is not None:
                    gold_exact_set.add((start, end, entity_type, value))

        pred_by_type = defaultdict(set)
        pred_exact_set = set()

        for entity in pred_entities:
            for label in entity.get('labels', []):
                value = entity.get('normalized', entity.get('text', ''))
                start = entity.get('start')
                end = entity.get('end')

                if label and value:
                    # Для обычных метрик
                    pred_by_type[label].add(value)

                    # Для Exact Match
                    if start is not None and end is not None:
                        pred_exact_set.add((start, end, label, value))

        # Расчет TP, FP, FN
        all_types = set(gold_by_type.keys()) | set(pred_by_type.keys())

        for entity_type in all_types:
            gold_set = gold_by_type.get(entity_type, set())
            pred_set = pred_by_type.get(entity_type, set())

            tp = len(gold_set & pred_set)  # Правильно предсказанные
            fp = len(pred_set - gold_set)  # Ложные срабатывания
            fn = len(gold_set - pred_set)  # Пропущенные

            tp_by_type[entity_type] += tp
            fp_by_type[entity_type] += fp
            fn_by_type[entity_type] += fn

        tp_exact += len(gold_exact_set & pred_exact_set)
        fp_exact += len(pred_exact_set - gold_exact_set)
        fn_exact += len(gold_exact_set - pred_exact_set)

    # Метрики по типам
    precision_by_type = {}
    recall_by_type = {}
    f1_by_type = {}

    all_types_final = set(tp_by_type.keys()) | set(fp_by_type.keys()) | set(fn_by_type.keys())

    for entity_type in all_types_final:
        tp = tp_by_type[entity_type]
        fp = fp_by_type[entity_type]
        fn = fn_by_type[entity_type]

        # Precision = TP / (TP + FP)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        # Recall = TP / (TP + FN)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 = 2 * Precision * Recall / (Precision + Recall)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_by_type[entity_type] = precision
        recall_by_type[entity_type] = recall
        f1_by_type[entity_type] = f1

    # Micro-усреднение
    tp_micro = sum(tp_by_type.values())
    fp_micro = sum(fp_by_type.values())
    fn_micro = sum(fn_by_type.values())

    precision_micro = tp_micro / (tp_micro + fp_micro) if (tp_micro + fp_micro) > 0 else 0.0
    recall_micro = tp_micro / (tp_micro + fn_micro) if (tp_micro + fn_micro) > 0 else 0.0
    f1_micro = (2 * precision_micro * recall_micro /
                (precision_micro + recall_micro)) if (precision_micro + recall_micro) > 0 else 0.0

    # Macro-усреднение
    if f1_by_type:
        f1_macro = sum(f1_by_type.values()) / len(f1_by_type)
    else:
        f1_macro = 0.0

    # Exact Match метрики
    precision_exact = tp_exact / (tp_exact + fp_exact) if (tp_exact + fp_exact) > 0 else 0.0
    recall_exact = tp_exact / (tp_exact + fn_exact) if (tp_exact + fn_exact) > 0 else 0.0
    f1_exact = (2 * precision_exact * recall_exact /
                (precision_exact + recall_exact)) if (precision_exact + recall_exact) > 0 else 0.0

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'exact_match_f1': f1_exact,
        'precision_by_type': precision_by_type,
        'recall_by_type': recall_by_type,
        'f1_by_type': f1_by_type,
    }


def save_ner_metrics(metrics, model_name, output_file ="../../reports/NER_metrics.md") :
    """Сохраняет метрики в metrics.md"""

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    file_exists = os.path.exists(output_file)

    # 'a' - append, 'w' - write
    mode = 'a' if file_exists else 'w'

    with open(output_file, mode, encoding='utf-8') as f:
        # Если файл новый, добавляем заголовок
        if not file_exists:
            f.write("# Отчет по метрикам NER моделей\n\n")
            f.write("---\n\n")

        if file_exists:
            f.write("\n---\n\n")

        f.write(f"## NER модель: {model_name}\n\n")
        f.write(f"## Дата генерации: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Общие метрики
        f.write("### Общие метрики\n\n")
        f.write("| Метрика | Значение |\n")
        f.write("|---------|----------|\n")
        f.write(f"| F1-Micro | {metrics.get('f1_micro', 0):.4f} |\n")
        f.write(f"| F1-Macro | {metrics.get('f1_macro', 0):.4f} |\n")
        f.write(f"| Exact Match F1 | {metrics.get('exact_match_f1', 0):.4f} |\n")
        f.write("\n")

        # Метрики по типам сущностей
        f.write("### Метрики по типам сущностей\n\n")

        precision_by_type = metrics.get('precision_by_type', {})
        recall_by_type = metrics.get('recall_by_type', {})
        f1_by_type = metrics.get('f1_by_type', {})

        all_types = set(precision_by_type.keys()) | set(recall_by_type.keys()) | set(f1_by_type.keys())

        if all_types:
            f.write("| Тип сущности | Precision | Recall | F1-Score |\n")
            f.write("|--------------|-----------|--------|----------|\n")

            # Сортируем по убыванию F1
            sorted_types = sorted(all_types, key=lambda x: f1_by_type.get(x, 0), reverse=True)

            for entity_type in sorted_types:
                precision = precision_by_type.get(entity_type, 0)
                recall = recall_by_type.get(entity_type, 0)
                f1 = f1_by_type.get(entity_type, 0)
                f.write(f"| `{entity_type}` | {precision:.4f} | {recall:.4f} | {f1:.4f} |\n")

            # Средние значения
            avg_precision = sum(precision_by_type.values()) / len(precision_by_type) if precision_by_type else 0
            avg_recall = sum(recall_by_type.values()) / len(recall_by_type) if recall_by_type else 0
            avg_f1 = sum(f1_by_type.values()) / len(f1_by_type) if f1_by_type else 0

            f.write(
                "| **Среднее** | **{:.4f}** | **{:.4f}** | **{:.4f}** |\n".format(avg_precision, avg_recall, avg_f1))
            f.write("\n")
        else:
            f.write("*Нет данных по типам сущностей*\n\n")

