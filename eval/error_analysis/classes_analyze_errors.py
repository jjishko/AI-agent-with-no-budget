#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ ошибок классификации: какие классы путаются, анализ неоднозначных запросов
"""

import os
import sys
import json
from collections import Counter, defaultdict

# Добавляем src в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.intent_classification import HybridClassifier


def load_data(train_path: str, test_path: str):
    """Загружает train и test данные"""
    with open(train_path, 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test = json.load(f)
    return train, test


def prepare_data(data):
    queries = [item['query'] for item in data]
    intents = [item['intent'] for item in data]
    return queries, intents


def analyze_errors(test_data, predictions, output_dir: str = './eval/error_analysis'):
    """
    Анализирует ошибки классификации
    """
    os.makedirs(output_dir, exist_ok=True)
    
    test_queries = [item['query'] for item in test_data]
    test_intents = [item['intent'] for item in test_data]
    pred_intents = [p['intent'] for p in predictions]
    pred_confidences = [p['confidence'] for p in predictions]
    
    # Находим ошибки
    errors = []
    correct = []
    low_confidence = []
    
    for i, (true, pred, conf) in enumerate(zip(test_intents, pred_intents, pred_confidences)):
        if true != pred:
            errors.append({
                'index': i,
                'query': test_queries[i],
                'true': true,
                'pred': pred,
                'confidence': conf
            })
        else:
            correct.append({
                'index': i,
                'query': test_queries[i],
                'true': true,
                'pred': pred,
                'confidence': conf
            })
        
        if conf < 0.5:
            low_confidence.append({
                'index': i,
                'query': test_queries[i],
                'true': true,
                'pred': pred,
                'confidence': conf
            })
    
    # 1. Общая статистика
    print("\n" + "="*80)
    print("АНАЛИЗ ОШИБОК")
    print("="*80)
    print(f"\nВсего запросов: {len(test_intents)}")
    print(f"  Правильно: {len(correct)} ({len(correct)/len(test_intents):.1%})")
    print(f"  Ошибок: {len(errors)} ({len(errors)/len(test_intents):.1%})")
    print(f"  Низкая уверенность (<0.5): {len(low_confidence)} ({len(low_confidence)/len(test_intents):.1%})")
    
    # 2. Матрица ошибок (детально)
    print("\n" + "-"*80)
    print("ДЕТАЛЬНАЯ МАТРИЦА ОШИБОК")
    print("-"*80)
    
    error_matrix = defaultdict(lambda: defaultdict(int))
    for err in errors:
        error_matrix[err['true']][err['pred']] += 1
    
    print(f"\n{'True\\Pred':<20}", end='')
    all_classes = sorted(set(test_intents) | set(pred_intents))
    for cls in all_classes:
        print(f"{cls:<15}", end='')
    print()
    print("-" * (20 + 15 * len(all_classes)))
    
    for true_cls in all_classes:
        print(f"{true_cls:<20}", end='')
        for pred_cls in all_classes:
            count = error_matrix[true_cls].get(pred_cls, 0)
            print(f"{count:<15}", end='')
        print()
    
    # 3. Самые частые ошибки
    print("\n" + "-"*80)
    print("САМЫЕ ЧАСТЫЕ ОШИБКИ")
    print("-"*80)
    
    error_counts = defaultdict(int)
    for err in errors:
        key = f"{err['true']} → {err['pred']}"
        error_counts[key] += 1
    
    for key, count in sorted(error_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"  {key}: {count} раз")
    
    # 4. Примеры ошибок (первые 10)
    print("\n" + "-"*80)
    print("ПРИМЕРЫ ОШИБОК (первые 10)")
    print("-"*80)
    
    for i, err in enumerate(errors[:10]):
        print(f"\n{i+1}. Запрос: {err['query']}")
        print(f"   Истинный класс: {err['true']}")
        print(f"   Предсказанный: {err['pred']}")
        print(f"   Уверенность: {err['confidence']:.3f}")
    
    # 5. Запросы с низкой уверенностью
    print("\n" + "-"*80)
    print("ЗАПРОСЫ С НИЗКОЙ УВЕРЕННОСТЬЮ (<0.5)")
    print("-"*80)
    
    for i, item in enumerate(low_confidence[:10]):
        print(f"\n{i+1}. Запрос: {item['query']}")
        print(f"   Истинный класс: {item['true']}")
        print(f"   Предсказанный: {item['pred']}")
        print(f"   Уверенность: {item['confidence']:.3f}")
    
    # 6. Анализ по классам
    print("\n" + "-"*80)
    print("ТОЧНОСТЬ ПО КЛАССАМ")
    print("-"*80)
    
    class_stats = defaultdict(lambda: {'total': 0, 'correct': 0, 'errors': 0})
    for true, pred in zip(test_intents, pred_intents):
        class_stats[true]['total'] += 1
        if true == pred:
            class_stats[true]['correct'] += 1
        else:
            class_stats[true]['errors'] += 1
    
    print(f"\n{'Class':<25} {'Total':<10} {'Correct':<10} {'Errors':<10} {'Accuracy':<10}")
    print("-"*65)
    for cls in sorted(class_stats.keys()):
        stats = class_stats[cls]
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{cls:<25} {stats['total']:<10} {stats['correct']:<10} {stats['errors']:<10} {acc:.2%}")
    
    # 7. Сохраняем результаты
    analysis_results = {
        'total': len(test_intents),
        'correct': len(correct),
        'errors': len(errors),
        'low_confidence': len(low_confidence),
        'error_matrix': {k: dict(v) for k, v in error_matrix.items()},
        'error_counts': dict(error_counts),
        'class_stats': {k: dict(v) for k, v in class_stats.items()}
    }
    
    with open(os.path.join(output_dir, 'classes_error_analysis.json'), 'w', encoding='utf-8') as f:
        json.dump(analysis_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в {output_dir}/classes_error_analysis.json")
    
    return analysis_results


def analyze_ambiguous_queries(test_data, train_data, output_dir: str = './eval/error_analysis'):
    """
    Анализирует неоднозначные запросы (те, что могут быть отнесены к разным классам)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Собираем все запросы
    all_queries = [item['query'] for item in train_data + test_data]
    all_intents = [item['intent'] for item in train_data + test_data]
    
    # Ищем ключевые слова, характерные для разных классов
    ambiguous_patterns = {
        'построй график': ['data_analysis', 'clarification_needed'],
        'покажи значение': ['tag_lookup', 'data_analysis'],
        'почему': ['troubleshooting', 'data_analysis'],
        'рассчитай': ['calculation', 'data_analysis'],
        'сравни': ['document_compare', 'doc_qa'],
        'найди': ['doc_qa', 'equipment_lookup', 'tag_lookup'],
        'что показывает': ['tag_lookup', 'data_analysis'],
    }
    
    print("\n" + "="*80)
    print("АНАЛИЗ НЕОДНОЗНАЧНЫХ ЗАПРОСОВ")
    print("="*80)
    
    for pattern, classes in ambiguous_patterns.items():
        print(f"\n'{pattern}':")
        print(f"  Возможные классы: {', '.join(classes)}")
        
        # Находим примеры
        examples = []
        for query, intent in zip(all_queries, all_intents):
            if pattern in query.lower():
                examples.append((query, intent))
        
        if examples:
            print(f"  Примеры ({len(examples)}):")
            for q, i in examples[:3]:
                print(f"    - {q} → {i}")
        else:
            print("  Примеров не найдено")
    
    # Сохраняем анализ
    results = {
        'ambiguous_patterns': ambiguous_patterns,
        'count': len(ambiguous_patterns)
    }
    
    with open(os.path.join(output_dir, 'classes_ambiguous_queries.json'), 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\nРезультаты сохранены в {output_dir}/classes_ambiguous_queries.json")


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'data', 'processed', 'classes_train_data.json')
    test_path = os.path.join(base_dir, 'data', 'processed', 'classes_test_data.json')
    model_dir = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'eval', 'error_analysis')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Ошибка: Файлы не найдены!")
        return
    
    # Загрузка данных
    train_data, test_data = load_data(train_path, test_path)
    train_queries, train_intents = prepare_data(train_data)
    test_queries, test_intents = prepare_data(test_data)
    
    # Загрузка модели
    print("Загрузка Hybrid модели...")
    hybrid_clf = HybridClassifier(model_dir=model_dir)
    hybrid_clf.fit(train_queries, train_intents, force_retrain=False)
    predictions = hybrid_clf.predict(test_queries)
    
    # Анализ ошибок
    analyze_errors(test_data, predictions, output_dir)
    
    # Анализ неоднозначных запросов
    analyze_ambiguous_queries(test_data, train_data, output_dir)
    
    print("\n" + "="*80)
    print("ГОТОВО!")
    print("="*80)


if __name__ == "__main__":
    main()
