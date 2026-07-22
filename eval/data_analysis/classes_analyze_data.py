#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Анализ исходных данных: распределение классов, статистика, визуализация
"""

import json
import os
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


def load_data(train_path: str, test_path: str):
    """Загружает train и test данные"""
    with open(train_path, 'r', encoding='utf-8') as f:
        train = json.load(f)
    with open(test_path, 'r', encoding='utf-8') as f:
        test = json.load(f)
    return train, test


def analyze_data(train_data, test_data, output_dir: str = './eval/data_analysis'):
    """Анализирует и визуализирует распределение данных"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Подготовка данных
    train_intents = [item['intent'] for item in train_data]
    test_intents = [item['intent'] for item in test_data]
    all_intents = train_intents + test_intents
    
    train_counter = Counter(train_intents)
    test_counter = Counter(test_intents)
    all_counter = Counter(all_intents)
    
    # 1. Общая статистика
    print("\n" + "="*80)
    print("АНАЛИЗ ДАННЫХ")
    print("="*80)
    print(f"\nВсего запросов: {len(all_intents)}")
    print(f"  Train: {len(train_intents)}")
    print(f"  Test: {len(test_intents)}")
    
    print(f"\nКоличество классов: {len(all_counter)}")
    print("\nРаспределение по классам:")
    print(f"{'Intent':<25} {'Всего':<10} {'Train':<10} {'Test':<10}")
    print("-"*55)
    
    for intent, count in sorted(all_counter.items(), key=lambda x: -x[1]):
        train_count = train_counter.get(intent, 0)
        test_count = test_counter.get(intent, 0)
        print(f"{intent:<25} {count:<10} {train_count:<10} {test_count:<10}")
    
    # 2. Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Все данные
    colors = sns.color_palette('husl', len(all_counter))
    axes[0].bar(all_counter.keys(), all_counter.values(), color=colors)
    axes[0].set_title('Распределение всех запросов')
    axes[0].set_xlabel('Intent')
    axes[0].set_ylabel('Количество')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Train vs Test сравнение
    intents = sorted(all_counter.keys())
    train_counts = [train_counter.get(i, 0) for i in intents]
    test_counts = [test_counter.get(i, 0) for i in intents]
    
    x = range(len(intents))
    width = 0.35
    axes[1].bar(x, train_counts, width, label='Train', color='blue', alpha=0.7)
    axes[1].bar([i + width for i in x], test_counts, width, label='Test', color='orange', alpha=0.7)
    axes[1].set_title('Сравнение Train/Test')
    axes[1].set_xlabel('Intent')
    axes[1].set_ylabel('Количество')
    axes[1].set_xticks([i + width/2 for i in x])
    axes[1].set_xticklabels(intents, rotation=45)
    axes[1].legend()
    
    # Процентное соотношение Train/Test
    total_train = len(train_intents)
    total_test = len(test_intents)
    train_pct = [train_counts[i] / total_train * 100 for i in range(len(intents))]
    test_pct = [test_counts[i] / total_test * 100 for i in range(len(intents))]
    
    axes[2].bar(x, train_pct, width, label='Train %', color='blue', alpha=0.7)
    axes[2].bar([i + width for i in x], test_pct, width, label='Test %', color='orange', alpha=0.7)
    axes[2].set_title('Процентное распределение')
    axes[2].set_xlabel('Intent')
    axes[2].set_ylabel('Процент (%)')
    axes[2].set_xticks([i + width/2 for i in x])
    axes[2].set_xticklabels(intents, rotation=45)
    axes[2].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'classes_data_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nГрафик сохранён в {output_dir}/classes_data_distribution.png")
    
    return {
        'total': len(all_intents),
        'train': len(train_intents),
        'test': len(test_intents),
        'classes': len(all_counter),
        'distribution': dict(all_counter)
    }


def main():
    # Пути к данным
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    train_path = os.path.join(base_dir, 'data', 'processed', 'classes_train_data.json')
    test_path = os.path.join(base_dir, 'data', 'processed', 'classes_test_data.json')
    output_dir = os.path.join(base_dir, 'eval', 'data_analysis')
    
    # Проверка наличия файлов
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Ошибка: Файлы не найдены!")
        print(f"  {train_path}")
        print(f"  {test_path}")
        return
    
    # Загрузка и анализ
    train_data, test_data = load_data(train_path, test_path)
    analyze_data(train_data, test_data, output_dir)


if __name__ == "__main__":
    main()
