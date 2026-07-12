import random
from typing import Dict
from collections import defaultdict
from src.entity_extraction.rule_based import RuleBasedNER
from pathlib import Path
from src.common.misc import load_data


def analyze_errors(model, test_data):
    """Анализирует ошибки модели и возвращает примеры FP и FN"""

    fp_examples = []  # модель выделила неверную сущность
    fn_examples = []  # модель не нашла верную сущность

    # Статистика по типам
    fp_stats = defaultdict(int)
    fn_stats = defaultdict(int)

    for idx, item in enumerate(test_data):
        text = item['text']
        if 'label' not in item:
            gold_entities = []
        else:
            gold_entities = item['label']

        pred_result = model.extract_entities(text)
        pred_entities = pred_result['label']

        gold_set = set()
        gold_details = {}

        for entity in gold_entities:
            entity_type = entity.get('labels', [])[0] if entity.get('labels') else None
            value = entity.get('normalized', entity.get('text', ''))

            if entity_type and value:
                key = (entity_type, value)
                gold_set.add(key)
                gold_details[key] = {
                    'type': entity_type,
                    'value': value,
                    'text': entity.get('text', ''),
                    'start': entity.get('start'),
                    'end': entity.get('end'),
                    'context': text,
                    'example_idx': idx
                }

        pred_set = set()
        pred_details = {}

        for entity in pred_entities:
            for label in entity.get('labels', []):
                value = entity.get('normalized', entity.get('text', ''))

                if label and value:
                    key = (label, value)
                    pred_set.add(key)
                    pred_details[key] = {
                        'type': label,
                        'value': value,
                        'text': entity.get('text', ''),
                        'start': entity.get('start'),
                        'end': entity.get('end'),
                        'context': text,
                        'example_idx': idx
                    }

        # --- FP: предсказано, но нет в gold ---
        fp = pred_set - gold_set

        for key in fp:
            if key in pred_details:
                pred_info = pred_details[key]

                # Ищем ближайшую gold сущность для сравнения (по позиции)
                gold_for_comparison = None
                min_distance = float('inf')

                for gold_key, gold_info in gold_details.items():
                    if gold_info['start'] is not None and pred_info['start'] is not None:
                        distance = abs(gold_info['start'] - pred_info['start'])
                        if distance < min_distance:
                            min_distance = distance
                            gold_for_comparison = gold_info

                fp_examples.append({
                    'predicted': {
                        'type': pred_info['type'],
                        'value': pred_info['value'],
                        'start': pred_info['start'],
                        'end': pred_info['end'],
                        'text': pred_info['text']
                    },
                    'gold': gold_for_comparison,  # может быть None, если нет близких gold
                    'context': text,
                    'example_idx': idx
                })
                fp_stats[key[0]] += 1

        # --- FN: есть в gold, но не предсказано ---
        fn = gold_set - pred_set

        for key in fn:
            if key in gold_details:
                fn_examples.append(gold_details[key])
                fn_stats[key[0]] += 1

    random.seed(42)

    # Инициализируем переменные с пустыми списками, даже если нет ошибок
    fp_samples = []
    fn_samples = []

    if fp_examples:
        fp_samples = random.sample(fp_examples, len(fp_examples))

    if fn_examples:
        fn_samples = random.sample(fn_examples, len(fn_examples))

    result = {
        'total_fp': len(fp_examples),
        'total_fn': len(fn_examples),
        'fp_by_type': dict(fp_stats),
        'fn_by_type': dict(fn_stats),
        'fp_samples': fp_samples,
        'fn_samples': fn_samples,
        'fp_all': fp_examples,
        'fn_all': fn_examples
    }

    return result


def print_error_summary(error_analysis):
    """Печатает сводку по ошибкам"""

    print("СВОДКА ПО ОШИБКАМ")

    print(f"\nВсего ошибок:")
    print(f"  • False Positives (ложные срабатывания): {error_analysis['total_fp']}")
    print(f"  • False Negatives (пропуски): {error_analysis['total_fn']}")

    print(f"\nДля анализа взято:")
    print(f"  • FP примеров: {len(error_analysis['fp_samples'])}")
    print(f"  • FN примеров: {len(error_analysis['fn_samples'])}")

    print("\n" + "-" * 80)
    print("Распределение FP по типам:")
    fp_by_type = error_analysis.get('fp_by_type', {})
    sorted_fp = sorted(fp_by_type.items(), key=lambda x: x[1], reverse=True)
    for entity_type, count in sorted_fp[:10]:
        bar = '█' * min(count, 40)
        print(f"  • {entity_type:15} {count:4} {bar}")

    print("\nРаспределение FN по типам:")
    fn_by_type = error_analysis.get('fn_by_type', {})
    sorted_fn = sorted(fn_by_type.items(), key=lambda x: x[1], reverse=True)
    for entity_type, count in sorted_fn[:10]:
        bar = '█' * min(count, 40)
        print(f"  • {entity_type:15} {count:4} {bar}")

    print("\n" + "=" * 80)


def highlight_entity_in_context(context: str, start: int, end: int) -> str:
    """Выделяет сущность в контексте с помощью маркеров"""
    if start is None or end is None:
        return context
    return context[:start] + '**' + context[start:end] + '**' + context[end:]


def print_error_examples(error_analysis: Dict, num_examples: int = 5) -> None:
    """Печатает примеры ошибок в консоль"""

    print("ПРИМЕРЫ FALSE POSITIVES (ложные срабатывания)")
    print("=" * 80)

    fp_samples = error_analysis['fp_samples'][:num_examples]
    if fp_samples:
        for idx, error in enumerate(fp_samples, 1):
            print(f"\n{idx}.")

            # Предсказанная сущность
            pred = error['predicted']
            print(
                f"   ПРЕДСКАЗАНО: тип='{pred['type']}', значение='{pred['value']}', позиция={pred['start']}-{pred['end']}")

            # Истинная сущность (если есть)
            gold = error['gold']
            if gold:
                print(
                    f"   ИСТИНА:      тип='{gold['type']}', значение='{gold['value']}', позиция={gold['start']}-{gold['end']}")
            else:
                print(f"   ИСТИНА:      Нет близкой сущности")

            # Контекст с подсветкой предсказанной сущности
            print(f"   КОНТЕКСТ:    {highlight_entity_in_context(error['context'], pred['start'], pred['end'])}")
    else:
        print("\nНет False Positive ошибок!")

    print("\n" + "=" * 80)
    print("ПРИМЕРЫ FALSE NEGATIVES (пропуски)")
    print("=" * 80)

    fn_samples = error_analysis['fn_samples'][:num_examples]
    if fn_samples:
        for idx, error in enumerate(fn_samples, 1):
            print(f"\n{idx}.")
            print(f"   ТИП:         {error['type']}")
            print(f"   ЗНАЧЕНИЕ:    '{error['value']}'")
            print(f"   ПОЗИЦИЯ:     {error['start']} - {error['end']}")
            print(f"   КОНТЕКСТ:    {highlight_entity_in_context(error['context'], error['start'], error['end'])}")
            print(f"   ПРИЧИНА:     Модель не распознала эту сущность")
    else:
        print("\nНет False Negative ошибок!")


test_path = Path("../../data/processed/NER_test.json")

if not test_path.exists():
    from src.common.preprocessing import ner_prerocessing

    ner_prerocessing()

test = load_data(test_path)

# Инициализация модели
model = RuleBasedNER()

# Анализ ошибок
error_analysis = analyze_errors(model, test)

# Вывод результатов
print_error_summary(error_analysis)
print_error_examples(error_analysis, num_examples=15)