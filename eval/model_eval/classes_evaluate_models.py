#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обучение и оценка моделей
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)

# Добавляем src в путь
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.intent_classification import RuleBasedClassifier, TfidfLRClassifier, HybridClassifier


def load_data(train_path: str, test_path: str):
    """Загружает train и test данные"""
    with open(train_path, 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test = json.load(f)
    return train, test


def prepare_data(data):
    """Извлекает query и intent"""
    queries = [item['query'] for item in data]
    intents = [item['intent'] for item in data]
    return queries, intents


def evaluate_classifier(y_true, y_pred):
    """Вычисляет все метрики"""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    classes = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    precision, recall, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )
    
    class_metrics = {}
    for i, cls in enumerate(classes):
        if i < len(precision):
            class_metrics[cls] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1_per_class[i],
                'support': support[i] if i < len(support) else 0
            }
    
    return {
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'class_metrics': class_metrics,
        'confusion_matrix': cm,
        'classes': classes
    }


def print_metrics(metrics, name):
    """Печатает метрики"""
    print(f"\n{name}:")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1-Macro: {metrics['f1_macro']:.4f}")
    print(f"  F1-Weighted: {metrics['f1_weighted']:.4f}")
    
    print(f"\n  Precision/Recall per class:")
    for cls, m in sorted(metrics['class_metrics'].items()):
        print(f"    {cls}: precision={m['precision']:.3f}, recall={m['recall']:.3f}, f1={m['f1']:.3f}, support={m['support']}")


def plot_confusion_matrix(cm, classes, title, filename):
    """Визуализация матрицы ошибок"""
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ {filename} сохранён")


def measure_latency(classifier, queries, n_iterations: int = 5) -> float:
    """Измеряет среднюю задержку в миллисекундах"""
    _ = classifier.predict(queries[:5])
    
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = classifier.predict(queries)
        end = time.perf_counter()
        times.append((end - start) / len(queries) * 1000)
    
    return np.mean(times)


def evaluate_models(train_data, test_data, model_dir: str = './models', force_retrain: bool = False, output_dir: str = './eval/model_eval'):
    """Тестирует все подходы"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    train_queries, train_intents = prepare_data(train_data)
    test_queries, test_intents = prepare_data(test_data)
    
    print(f"\nTrain: {len(train_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    results = {}
    
    # ============================================================
    # 1. RULE-BASED
    # ============================================================
    print("\n" + "="*80)
    print("RULE-BASED CLASSIFIER (УЛУЧШЕННЫЙ)")
    print("="*80)
    
    rule_clf = RuleBasedClassifier()
    rule_results = rule_clf.predict(test_queries)
    rule_pred = [r['intent'] for r in rule_results]
    rule_conf = [r['confidence'] for r in rule_results]
    
    rule_metrics = evaluate_classifier(test_intents, rule_pred)
    print_metrics(rule_metrics, "Rule-based")
    
    rule_latency = measure_latency(rule_clf, test_queries)
    print(f"\n  Latency: {rule_latency:.2f} ms/query")
    
    plot_confusion_matrix(
        rule_metrics['confusion_matrix'], 
        rule_metrics['classes'],
        "Rule-based Confusion Matrix", 
        os.path.join(output_dir, 'classes_confusion_matrix_rules.png')
    )
    
    results['rule'] = {
        'metrics': rule_metrics,
        'predictions': rule_pred,
        'confidences': rule_conf,
        'latency': rule_latency
    }
    
    # ============================================================
    # 2. TF-IDF + LR
    # ============================================================
    print("\n" + "="*80)
    print("TF-IDF + LOGISTIC REGRESSION (УЛУЧШЕННЫЙ)")
    print("="*80)
    
    tfidf_clf = TfidfLRClassifier(model_dir=model_dir)
    tfidf_clf.fit(train_queries, train_intents, force_retrain)
    tfidf_results = tfidf_clf.predict(test_queries)
    tfidf_pred = [r['intent'] for r in tfidf_results]
    tfidf_conf = [r['confidence'] for r in tfidf_results]
    
    tfidf_metrics = evaluate_classifier(test_intents, tfidf_pred)
    print_metrics(tfidf_metrics, "TF-IDF+LR")
    
    tfidf_latency = measure_latency(tfidf_clf, test_queries)
    print(f"\n  Latency: {tfidf_latency:.2f} ms/query")
    
    # Важные признаки
    importance = tfidf_clf.get_feature_importance(top_n=10)
    if importance:
        print("\n  Важные признаки для классов:")
        for intent, features in list(importance.items())[:3]:
            top_features = [f[0] for f in features[:5]]
            print(f"    {intent}: {', '.join(top_features)}")
    
    plot_confusion_matrix(
        tfidf_metrics['confusion_matrix'], 
        tfidf_metrics['classes'],
        "TF-IDF+LR Confusion Matrix", 
        os.path.join(output_dir, 'classes_confusion_matrix_tfidf.png')
    )
    
    results['tfidf'] = {
        'metrics': tfidf_metrics,
        'predictions': tfidf_pred,
        'confidences': tfidf_conf,
        'latency': tfidf_latency
    }
    
    # ============================================================
    # 3. HYBRID
    # ============================================================
    print("\n" + "="*80)
    print("HYBRID (ИСПРАВЛЕННЫЕ ПОРОГИ: rule=0.7, model=0.65)")
    print("="*80)
    
    hybrid_clf = HybridClassifier(rule_threshold=0.7, model_threshold=0.65, model_dir=model_dir)
    hybrid_clf.fit(train_queries, train_intents, force_retrain)
    hybrid_results = hybrid_clf.predict(test_queries)
    hybrid_pred = [r['intent'] for r in hybrid_results]
    hybrid_conf = [r['confidence'] for r in hybrid_results]
    hybrid_levels = [r['level'] for r in hybrid_results]
    
    hybrid_metrics = evaluate_classifier(test_intents, hybrid_pred)
    print_metrics(hybrid_metrics, "Hybrid")
    
    # Уровни уверенности
    print(f"\n  Confidence levels:")
    level_counts = Counter(hybrid_levels)
    for level in ['high', 'medium', 'low']:
        count = level_counts.get(level, 0)
        print(f"    {level}: {count} ({count/len(hybrid_levels):.1%})")
    
    hybrid_latency = measure_latency(hybrid_clf, test_queries)
    print(f"\n  Latency: {hybrid_latency:.2f} ms/query")
    
    plot_confusion_matrix(
        hybrid_metrics['confusion_matrix'], 
        hybrid_metrics['classes'],
        "Hybrid Confusion Matrix", 
        os.path.join(output_dir, 'classes_confusion_matrix_hybrid.png')
    )
    
    results['hybrid'] = {
        'metrics': hybrid_metrics,
        'predictions': hybrid_pred,
        'confidences': hybrid_conf,
        'levels': hybrid_levels,
        'latency': hybrid_latency
    }
    
    # ============================================================
    # 5. СРАВНИТЕЛЬНАЯ ТАБЛИЦА
    # ============================================================
    print("\n" + "="*80)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*80)
    
    print(f"\n{'Approach':<25} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<15} {'Latency (ms)':<15}")
    print("-"*80)
    print(f"{'Rule-based':<25} {rule_metrics['accuracy']:.4f}     {rule_metrics['f1_macro']:.4f}     {rule_metrics['f1_weighted']:.4f}     {rule_latency:.2f}")
    print(f"{'TF-IDF+LR':<25} {tfidf_metrics['accuracy']:.4f}     {tfidf_metrics['f1_macro']:.4f}     {tfidf_metrics['f1_weighted']:.4f}     {tfidf_latency:.2f}")
    print(f"{'Hybrid':<25} {hybrid_metrics['accuracy']:.4f}     {hybrid_metrics['f1_macro']:.4f}     {hybrid_metrics['f1_weighted']:.4f}     {hybrid_latency:.2f}")
    
    # ============================================================
    # 6. COVERAGE ANALYSIS
    # ============================================================
    print("\n" + "="*80)
    print("ANALYSIS BY CONFIDENCE THRESHOLD (Hybrid)")
    print("="*80)
    
    thresholds = [0.5, 0.65, 0.7, 0.8, 0.9]
    for th in thresholds:
        confident = [i for i, c in enumerate(hybrid_conf) if c >= th]
        total = len(hybrid_conf)
        coverage = len(confident) / total if total > 0 else 0
        
        if confident:
            y_true_conf = [test_intents[i] for i in confident]
            y_pred_conf = [hybrid_pred[i] for i in confident]
            acc = accuracy_score(y_true_conf, y_pred_conf)
        else:
            acc = 0
        
        print(f"  Threshold {th}: coverage={coverage:.2%}, "
              f"acc={acc:.2%}, "
              f"non-confident={total - len(confident)}/{total}")
    
    # Сохраняем результаты
    summary = {
        'rule': {'accuracy': rule_metrics['accuracy'], 'f1_macro': rule_metrics['f1_macro'], 'f1_weighted': rule_metrics['f1_weighted']},
        'tfidf': {'accuracy': tfidf_metrics['accuracy'], 'f1_macro': tfidf_metrics['f1_macro'], 'f1_weighted': tfidf_metrics['f1_weighted']},
        'hybrid': {'accuracy': hybrid_metrics['accuracy'], 'f1_macro': hybrid_metrics['f1_macro'], 'f1_weighted': hybrid_metrics['f1_weighted']}
    }
    
    with open(os.path.join(output_dir, 'classes_results_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    return results


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'data', 'processed', 'classes_train_data.json')
    test_path = os.path.join(base_dir, 'data', 'processed', 'classes_test_data.json')
    model_dir = os.path.join(base_dir, 'models')
    output_dir = os.path.join(base_dir, 'eval', 'model_eval')
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Ошибка: Файлы не найдены!")
        print(f"  {train_path}")
        print(f"  {test_path}")
        return
    
    # Проверка наличия модели
    model_paths = {
        'vectorizer': os.path.join(model_dir, 'classes_tfidf_vectorizer.pkl'),
        'classifier': os.path.join(model_dir, 'classes_tfidf_classifier.pkl'),
        'intents': os.path.join(model_dir, 'classes_tfidf_intents.pkl'),
    }
    models_exist = all(os.path.exists(path) for path in model_paths.values())
    
    print("="*80)
    print("ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ")
    print("="*80)
    
    if models_exist:
        print("\nНайдена сохранённая модель TF-IDF+LR")
        print("  1 - Использовать сохранённую модель (быстро)")
        print("  2 - Переобучить модель заново")
        print("  3 - Выйти")
        
        choice = input("\nВаш выбор (1/2/3): ").strip()
        
        if choice == '1':
            force_retrain = False
        elif choice == '2':
            force_retrain = True
        else:
            print("Выход.")
            return
    else:
        print("\nМодель не найдена. Запуск обучения...")
        force_retrain = True
    
    train_data, test_data = load_data(train_path, test_path)
    evaluate_models(train_data, test_data, model_dir, force_retrain, output_dir)
    
    print("\n" + "="*80)
    print("ГОТОВО!")
    print("="*80)
    print(f"\nРезультаты сохранены в {output_dir}/")


if __name__ == "__main__":
    main()
