#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Классификация промышленных запросов с кэшированием моделей
Подходы: Rule-based, TF-IDF+LR, Hybrid
"""

import json
import re
import time
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix, classification_report
)


# КОНСТАНТЫ


MODEL_DIR = './models'
CACHE_DIR = './cache'
TRAIN_FILE = 'train_data.json'
TEST_FILE = 'test_data.json'
MODEL_PATHS = {
    'tfidf_vectorizer': os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'),
    'tfidf_classifier': os.path.join(MODEL_DIR, 'tfidf_classifier.pkl'),
    'tfidf_intents': os.path.join(MODEL_DIR, 'tfidf_intents.pkl'),
}

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)


# ЗАГРУЗКА ДАННЫХ


def load_data(train_path, test_path):
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


# RULE-BASED CLASSIFIER 

class RuleBasedClassifier:
    """Классификатор на основе правил с улучшенными правилами для tag_lookup"""
    
    def __init__(self):
        self.rules = self._build_rules()
    
    def _build_rules(self):
        return {
            'doc_qa': {
                'keywords': [
                    'инструкци', 'регламент', 'паспорт', 'руководств', 
                    'методик', 'документаци', 'описани', 'схем', 'чертеж',
                    'техническ', 'характеристик', 'порядок', 'процедур',
                    'правил', 'норм', 'найди', 'покажи', 'выдай'
                ],
                'weight': 1.0
            },
            'equipment_lookup': {
                'keywords': [
                    'насос', 'колонн', 'компрессор', 'теплообменник',
                    'сепаратор', 'емкост', 'печь', 'фильтр', 'задвижк',
                    'агрегат', 'установк', 'резервуар', 'реактор',
                    'абсорбер', 'десорбер', 'вентилятор', 'турбин',
                    'воздуходувк', 'электродвигател', 'клапан'
                ],
                'weight': 1.0
            },
            'tag_lookup': {
                'keywords': [
                    # Явные запросы про теги (строгие правила)
                    'где используется тег', 'найди тег', 'покажи тег',
                    'что за тег', 'какой тег', 'тег с индексом',
                    'какие параметры снимает', 'что показывает сейчас',
                    'текущее значение', 'покажи текущее', 'мгновенное значение',
                    'что показывает в данный момент', 'в реальном времени',
                    'покажи позицию', 'где задействован датчик',
                    'где смонтирован датчик', 'где закреплен датчик',
                    'где размещен датчик', 'где расположен датчик',
                    'что за датчик', 'какой датчик'
                ],
                'weight': 0.8  # Снижен вес, чтобы не перекрывать другие классы
            },
            'troubleshooting': {
                'keywords': [
                    'почему', 'причина', 'неисправн', 'авари',
                    'скачет', 'падает', 'упал', 'растет',
                    'гудит', 'греетс', 'не открываетс',
                    'ниже нормы', 'не выходит', 'не держит',
                    'снизил', 'превысил', 'зашкаливает',
                    'отказ', 'сбой', 'ошибк', 'помпаж', 'вибрирует'
                ],
                'weight': 1.0
            },
            'calculation': {
                'keywords': [
                    'рассчитай', 'вычисли', 'переведи', 'посчитай',
                    'определи', 'суммируй', 'средн', 'кпд',
                    'эффективн', 'загрузк', 'процент', 'объем',
                    'среднее значение', 'медианное значение'
                ],
                'weight': 1.0
            },
            'data_analysis': {
                'keywords': [
                    'построй', 'покажи', 'визуализируй', 'отобрази',
                    'проанализируй', 'выведи', 'сделай', 'тренд',
                    'график', 'истори', 'отклонени', 'выборк',
                    'за сутки', 'за смену', 'за час', 'вчера', 'сегодня',
                    'за последни', 'динамик', 'поведени', 'выборк'
                ],
                'weight': 0.9
            },
            'document_compare': {
                'keywords': [
                    'сравни две', 'две верси', 'отличи', 'разница между',
                    'сопостав', 'расхождени', 'две редакци',
                    '2021 и 2024', '2020 и 2023', 'старой и новой',
                    'сравни два', 'сравни описание'
                ],
                'weight': 1.0
            },
            'instruction_generation': {
                'keywords': [
                    'составь', 'создай', 'сформируй', 'разработай',
                    'сгенерируй', 'чек-лист', 'памятк', 'алгоритм',
                    'план', 'последовательность', 'инструкци',
                    'методик', 'программ'
                ],
                'weight': 1.0
            },
            'clarification_needed': {
                'keywords': [
                    'построй график', 'найди инструкцию', 'покажи оборудование',
                    'рассчитай расход', 'сравни документы', 'что за параметр',
                    'проанализируй теги', 'составь чек-лист', 'где используется тег',
                    'почему падает', 'выдай историю', 'посчитай среднее',
                    'отобрази тренд', 'сформируй отчет', 'создай инструкцию',
                    'сравни две версии', 'найди отличия', 'покажи паспорт',
                    'что показывает', 'какая модель', 'где установлен',
                    'есть ли данные', 'выполни расчет', 'построй зависимость',
                    'проверь состояние', 'уточни параметр', 'нужен график',
                    'дай текущее значение', 'сделай выборку', 'покажи изменения',
                    'найди описание', 'требуется анализ', 'сформируй план',
                    'разработай методику', 'составь последовательность',
                    'вычисли кпд', 'определи загрузку', 'сравни показатели',
                    'отобрази отклонения', 'визуализируй работу',
                    'проанализируй поведение', 'найди расхождения',
                    'сопоставь данные', 'сделай сводку', 'покажи динамику',
                    'рассчитай эффективность', 'уточни временной интервал',
                    'нет данных по объекту', 'покажи динаху'
                ],
                'weight': 0.9
            },
            'out_of_scope': {
                'keywords': [
                    'стих', 'байк', 'погода', 'рецепт', 'фильм',
                    'кофе', 'тост', 'анекдот', 'дружб', 'подарок',
                    'президент', 'евро', 'сапог', 'кот', 'песн',
                    'поздрав', 'зачем', 'сколько лет', 'голубое',
                    'юбилей', 'борщ', 'переведи на английский'
                ],
                'weight': 1.0
            }
        }
    
    def _has_tag(self, text: str) -> bool:
        """Проверяет наличие тега в тексте"""
        patterns = [
            r'[A-Z]{2,4}[\s\-]*[0-9]{2,4}',
            r'[A-Z]{2,4}[\s\-]*[0-9]{1,4}[A-Z]?'
        ]
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False
    
    def _get_tag_context(self, text: str) -> str:
        """
        Определяет контекст использования тега
        Возвращает: 'action', 'location', 'value', 'history', 'unknown'
        """
        text_lower = text.lower()
        
        # Проверяем наличие глаголов действия
        if any(w in text_lower for w in ['найди', 'покажи', 'выдай', 'открой', 'найти']):
            return 'action'
        if any(w in text_lower for w in ['где', 'позици', 'располож', 'смонтирован', 'закреплен', 'размещен']):
            return 'location'
        if any(w in text_lower for w in ['сейчас', 'текущее', 'мгновенное', 'прямо сейчас', 'в реальном времени']):
            return 'value'
        if any(w in text_lower for w in ['истори', 'за сутки', 'за смену', 'тренд', 'график', 'динамик']):
            return 'history'
        
        return 'unknown'
    
    def classify(self, query: str) -> tuple:
        text = query.lower().strip()
        scores = {}
        
        # Базовая оценка по правилам
        for intent, rule in self.rules.items():
            score = 0
            matches = 0
            total_keywords = len(rule['keywords'])
            
            for keyword in rule['keywords']:
                if keyword in text:
                    matches += 1
            
            if total_keywords > 0:
                score = matches / total_keywords * rule['weight']
            
            scores[intent] = score
        
        # Специальная обработка для tag_lookup
        if self._has_tag(text):
            tag_context = self._get_tag_context(text)
            
            # Если есть тег + контекст "история" → data_analysis
            if tag_context == 'history':
                scores['data_analysis'] = max(scores.get('data_analysis', 0), 0.85)
                scores['tag_lookup'] = min(scores.get('tag_lookup', 0), 0.5)
            # Если есть тег + контекст "значение сейчас" → tag_lookup
            elif tag_context == 'value':
                scores['tag_lookup'] = max(scores.get('tag_lookup', 0), 0.9)
            # Если есть тег + контекст "действие" → doc_qa или equipment_lookup
            elif tag_context == 'action':
                # Проверяем, есть ли признаки doc_qa
                if any(w in text for w in ['инструкци', 'паспорт', 'документаци', 'описани']):
                    scores['doc_qa'] = max(scores.get('doc_qa', 0), 0.85)
                    scores['tag_lookup'] = min(scores.get('tag_lookup', 0), 0.4)
                else:
                    scores['tag_lookup'] = max(scores.get('tag_lookup', 0), 0.7)
            # Если есть тег + контекст "расположение" → tag_lookup
            elif tag_context == 'location':
                scores['tag_lookup'] = max(scores.get('tag_lookup', 0), 0.85)
            # Если тег без контекста → снижаем уверенность tag_lookup
            else:
                # Проверяем, нет ли других сильных признаков
                if scores.get('troubleshooting', 0) > 0.3:
                    scores['tag_lookup'] = min(scores.get('tag_lookup', 0), 0.3)
                elif scores.get('calculation', 0) > 0.3:
                    scores['tag_lookup'] = min(scores.get('tag_lookup', 0), 0.3)
                elif scores.get('data_analysis', 0) > 0.3:
                    scores['tag_lookup'] = min(scores.get('tag_lookup', 0), 0.3)
                else:
                    scores['tag_lookup'] = max(scores.get('tag_lookup', 0), 0.6)
        
        # Нормализуем
        max_score = max(scores.values()) if scores else 0
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}
        
        # Выбираем лучший
        best_intent = max(scores, key=scores.get) if scores else 'clarification_needed'
        best_score = scores.get(best_intent, 0)
        route = self._intent_to_route(best_intent)
        
        return best_intent, route, best_score, scores
    
    def _intent_to_route(self, intent: str) -> str:
        mapping = {
            'doc_qa': 'rag_search',
            'equipment_lookup': 'registry_search',
            'tag_lookup': 'tag_search',
            'troubleshooting': 'diagnostic_agent',
            'calculation': 'tool_agent',
            'data_analysis': 'timeseries_agent',
            'document_compare': 'doc_compare_agent',
            'instruction_generation': 'generation_agent',
            'clarification_needed': 'clarification_flow',
            'out_of_scope': 'fallback'
        }
        return mapping.get(intent, 'fallback')
    
    def predict(self, queries: list) -> list:
        results = []
        for query in queries:
            intent, route, conf, _ = self.classify(query)
            results.append({
                'query': query,
                'intent': intent,
                'route': route,
                'confidence': conf
            })
        return results


# TF-IDF + LOGISTIC REGRESSION 


class TfidfLRClassifier:
    """Классификатор на основе TF-IDF + Logistic Regression с кэшированием"""
    
    def __init__(self):
        self.vectorizer = None
        self.classifier = None
        self.intent_list = None
        self.is_fitted = False
    
    def _is_model_cached(self) -> bool:
        """Проверяет, есть ли сохранённая модель"""
        return all(os.path.exists(path) for path in MODEL_PATHS.values())
    
    def _save_model(self):
        """Сохраняет модель на диск"""
        with open(MODEL_PATHS['tfidf_vectorizer'], 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(MODEL_PATHS['tfidf_classifier'], 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(MODEL_PATHS['tfidf_intents'], 'wb') as f:
            pickle.dump(self.intent_list, f)
        print(f"  [TF-IDF] Модель сохранена в {MODEL_DIR}")
    
    def _load_model(self):
        """Загружает модель с диска"""
        with open(MODEL_PATHS['tfidf_vectorizer'], 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(MODEL_PATHS['tfidf_classifier'], 'rb') as f:
            self.classifier = pickle.load(f)
        with open(MODEL_PATHS['tfidf_intents'], 'rb') as f:
            self.intent_list = pickle.load(f)
        self.is_fitted = True
        print(f"  [TF-IDF] Модель загружена из {MODEL_DIR}")
    
    def fit(self, queries: list, intents: list, force_retrain: bool = False):
        """Обучение модели с кэшированием"""
        
        if not force_retrain and self._is_model_cached():
            self._load_model()
            return self
        
        print("  [TF-IDF] Обучение модели...")
        print(f"  [TF-IDF] Количество запросов: {len(queries)}")
        
        # Векторизация с улучшенными параметрами
        self.vectorizer = TfidfVectorizer(
            max_features=8000,  # Увеличено с 5000
            ngram_range=(1, 3),  # Добавлены триграммы
            lowercase=True,
            min_df=2,
            analyzer='word'
        )
        X = self.vectorizer.fit_transform(queries)
        print(f"  [TF-IDF] Размерность: {X.shape}")
        
        # Обучение классификатора
        self.intent_list = sorted(set(intents))
        self.classifier = LogisticRegression(
            max_iter=1500,  # Увеличено для сходимости
            C=1.0,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'  # Добавлено для борьбы с дисбалансом
        )
        self.classifier.fit(X, intents)
        self.is_fitted = True
        
        # Сохраняем модель
        self._save_model()
        
        return self
    
    def predict(self, queries: list) -> list:
        if not self.is_fitted:
            raise ValueError("Модель не обучена! Запустите fit() сначала.")
        
        X = self.vectorizer.transform(queries)
        probas = self.classifier.predict_proba(X)
        
        results = []
        for i, proba in enumerate(probas):
            max_idx = np.argmax(proba)
            confidence = proba[max_idx]
            intent = self.intent_list[max_idx]
            route = self._intent_to_route(intent)
            
            results.append({
                'query': queries[i],
                'intent': intent,
                'route': route,
                'confidence': float(confidence),
                'proba': proba.tolist()
            })
        
        return results
    
    def get_feature_importance(self, top_n: int = 20) -> dict:
        """Возвращает наиболее важные признаки для каждого класса"""
        if not self.is_fitted:
            return {}
        
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.coef_
        
        importance = {}
        for i, intent in enumerate(self.intent_list):
            if i < len(coef):
                top_indices = np.argsort(coef[i])[-top_n:]
                importance[intent] = [
                    (feature_names[idx], coef[i][idx]) 
                    for idx in top_indices
                ]
        
        return importance
    
    def _intent_to_route(self, intent: str) -> str:
        mapping = {
            'doc_qa': 'rag_search',
            'equipment_lookup': 'registry_search',
            'tag_lookup': 'tag_search',
            'troubleshooting': 'diagnostic_agent',
            'calculation': 'tool_agent',
            'data_analysis': 'timeseries_agent',
            'document_compare': 'doc_compare_agent',
            'instruction_generation': 'generation_agent',
            'clarification_needed': 'clarification_flow',
            'out_of_scope': 'fallback'
        }
        return mapping.get(intent, 'fallback')


# HYBRID CLASSIFIER


class HybridClassifier:
    """
    Гибридный классификатор: Rules + TF-IDF+LR + Confidence
    Исправленные пороги:
    - rule_threshold: 0.7 (было 0.9) — правила чаще достигают этого порога
    - model_threshold: 0.65 (было 0.7) — больше покрытие
    """
    
    def __init__(self, rule_threshold=0.7, model_threshold=0.65):
        self.rule_clf = RuleBasedClassifier()
        self.model_clf = None
        self.rule_threshold = rule_threshold
        self.model_threshold = model_threshold
        self.is_fitted = False
    
    def fit(self, queries: list, intents: list, force_retrain: bool = False):
        """Обучение гибридного классификатора"""
        print("[Hybrid] Обучение TF-IDF+LR...")
        self.model_clf = TfidfLRClassifier()
        self.model_clf.fit(queries, intents, force_retrain)
        self.is_fitted = True
        return self
    
    def classify(self, query: str) -> tuple:
        """Классифицирует запрос, возвращает (intent, route, confidence, level)"""
        # 1. Правила
        rule_intent, rule_route, rule_conf, _ = self.rule_clf.classify(query)
        
        if rule_conf >= self.rule_threshold:
            return rule_intent, rule_route, rule_conf, 'high'
        
        # 2. Модель
        if self.is_fitted and self.model_clf:
            result = self.model_clf.predict([query])[0]
            model_conf = result['confidence']
            model_intent = result['intent']
            model_route = result['route']
            
            if model_conf >= self.model_threshold:
                return model_intent, model_route, model_conf, 'high'
            elif model_conf >= 0.45:  # Снижен порог для серой зоны
                return model_intent, model_route, model_conf, 'medium'
            else:
                return 'clarification_needed', 'clarification_flow', model_conf, 'low'
        
        # 3. Fallback
        if rule_conf >= 0.4:  # Снижен порог для fallback
            return rule_intent, rule_route, rule_conf, 'medium'
        else:
            return 'clarification_needed', 'clarification_flow', rule_conf, 'low'
    
    def predict(self, queries: list) -> list:
        results = []
        for q in queries:
            intent, route, conf, level = self.classify(q)
            results.append({
                'query': q,
                'intent': intent,
                'route': route,
                'confidence': conf,
                'level': level
            })
        return results


# МЕТРИКИ И ВИЗУАЛИЗАЦИЯ


def evaluate_classifier(y_true, y_pred, name="Model"):
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

def print_classification_report(y_true, y_pred, class_names):
    """Печатает детальный отчет"""
    print(f"\n{'-'*80}")
    print("Classification Report")
    print('-'*80)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    print(report)

def analyze_confusion(cm, classes):
    """Анализирует матрицу ошибок"""
    error_pairs = []
    for i in range(len(classes)):
        for j in range(len(classes)):
            if i != j and cm[i][j] > 0:
                error_pairs.append((classes[i], classes[j], cm[i][j]))
    
    error_pairs.sort(key=lambda x: -x[2])
    
    print("\n" + "="*80)
    print("Самые частые ошибки классификации:")
    print("="*80)
    if error_pairs:
        for true_cls, pred_cls, count in error_pairs[:15]:
            print(f"  {true_cls} → {pred_cls}: {count} раз")
    else:
        print("  Ошибок нет!")

def measure_latency(classifier, queries: list, n_iterations: int = 5) -> float:
    """Измеряет среднюю задержку в миллисекундах"""
    # Прогрев
    _ = classifier.predict(queries[:5])
    
    times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = classifier.predict(queries)
        end = time.perf_counter()
        times.append((end - start) / len(queries) * 1000)
    
    return np.mean(times)


# ОСНОВНАЯ ФУНКЦИЯ


def evaluate_approaches(train_data, test_data, force_retrain=False):
    """Тестирует все подходы"""
    
    train_queries, train_intents = prepare_data(train_data)
    test_queries, test_intents = prepare_data(test_data)
    
    print(f"\nTrain: {len(train_data)} samples")
    print(f"Test: {len(test_data)} samples")
    
    # Статистика по интентам
    print(f"\nTrain intent distribution:")
    for intent, count in sorted(Counter(train_intents).items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count}")
    

    # 1. RULE-BASED (улучшенный)

    print("\n" + "="*80)
    print("RULE-BASED CLASSIFIER (УЛУЧШЕННЫЙ)")
    print("="*80)
    
    rule_clf = RuleBasedClassifier()
    rule_results = rule_clf.predict(test_queries)
    rule_pred = [r['intent'] for r in rule_results]
    rule_conf = [r['confidence'] for r in rule_results]
    
    rule_metrics = evaluate_classifier(test_intents, rule_pred, "Rule-based")
    print_metrics(rule_metrics, "Rule-based")
    
    rule_latency = measure_latency(rule_clf, test_queries)
    print(f"\n  Latency: {rule_latency:.2f} ms/query")
    
    plot_confusion_matrix(rule_metrics['confusion_matrix'], rule_metrics['classes'],
                          "Rule-based Confusion Matrix", "confusion_matrix_rules.png")
    

    # 2. TF-IDF + LR (улучшенный)

    print("\n" + "="*80)
    print("TF-IDF + LOGISTIC REGRESSION (УЛУЧШЕННЫЙ)")
    print("="*80)
    
    tfidf_clf = TfidfLRClassifier()
    tfidf_clf.fit(train_queries, train_intents, force_retrain)
    tfidf_results = tfidf_clf.predict(test_queries)
    tfidf_pred = [r['intent'] for r in tfidf_results]
    tfidf_conf = [r['confidence'] for r in tfidf_results]
    
    tfidf_metrics = evaluate_classifier(test_intents, tfidf_pred, "TF-IDF+LR")
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
    
    plot_confusion_matrix(tfidf_metrics['confusion_matrix'], tfidf_metrics['classes'],
                          "TF-IDF+LR Confusion Matrix", "confusion_matrix_tfidf.png")
    

    # 3. HYBRID (с исправленными порогами)
 
    print("\n" + "="*80)
    print("HYBRID (ИСПРАВЛЕННЫЕ ПОРОГИ: rule=0.7, model=0.65)")
    print("="*80)
    
    hybrid_clf = HybridClassifier(rule_threshold=0.7, model_threshold=0.65)
    hybrid_clf.fit(train_queries, train_intents, force_retrain)
    hybrid_results = hybrid_clf.predict(test_queries)
    hybrid_pred = [r['intent'] for r in hybrid_results]
    hybrid_conf = [r['confidence'] for r in hybrid_results]
    hybrid_levels = [r['level'] for r in hybrid_results]
    
    hybrid_metrics = evaluate_classifier(test_intents, hybrid_pred, "Hybrid")
    print_metrics(hybrid_metrics, "Hybrid")
    
    # Уровни уверенности
    print(f"\n  Confidence levels:")
    level_counts = Counter(hybrid_levels)
    for level in ['high', 'medium', 'low']:
        count = level_counts.get(level, 0)
        print(f"    {level}: {count} ({count/len(hybrid_levels):.1%})")
    
    hybrid_latency = measure_latency(hybrid_clf, test_queries)
    print(f"\n  Latency: {hybrid_latency:.2f} ms/query")
    
    plot_confusion_matrix(hybrid_metrics['confusion_matrix'], hybrid_metrics['classes'],
                          "Hybrid Confusion Matrix", "confusion_matrix_hybrid.png")
    

    # 4. АНАЛИЗ ОШИБОК

    print("\n" + "="*80)
    print("АНАЛИЗ ОШИБОК (Hybrid)")
    print("="*80)
    analyze_confusion(hybrid_metrics['confusion_matrix'], hybrid_metrics['classes'])
    

    # 5. СРАВНИТЕЛЬНАЯ ТАБЛИЦА

    print("\n" + "="*80)
    print("ИТОГОВОЕ СРАВНЕНИЕ")
    print("="*80)
    
    print(f"\n{'Approach':<25} {'Accuracy':<12} {'F1-Macro':<12} {'F1-Weighted':<15} {'Latency (ms)':<15}")
    print("-"*80)
    print(f"{'Rule-based':<25} {rule_metrics['accuracy']:.4f}     {rule_metrics['f1_macro']:.4f}     {rule_metrics['f1_weighted']:.4f}     {rule_latency:.2f}")
    print(f"{'TF-IDF+LR':<25} {tfidf_metrics['accuracy']:.4f}     {tfidf_metrics['f1_macro']:.4f}     {tfidf_metrics['f1_weighted']:.4f}     {tfidf_latency:.2f}")
    print(f"{'Hybrid':<25} {hybrid_metrics['accuracy']:.4f}     {hybrid_metrics['f1_macro']:.4f}     {hybrid_metrics['f1_weighted']:.4f}     {hybrid_latency:.2f}")
    

    # 6. COVERAGE ANALYSIS
 
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

# MAIN


def main():
    print("="*80)
    print("КЛАССИФИКАЦИЯ ПРОМЫШЛЕННЫХ ЗАПРОСОВ")
    print("="*80)
    print("\nИСПРАВЛЕНИЯ:")
    print("  1. Пороги Hybrid: rule_threshold=0.7 (было 0.9), model_threshold=0.65 (было 0.7)")
    print("  2. Улучшенные правила для tag_lookup (более строгие, с контекстом)")
    print("  3. TF-IDF: max_features=8000 (было 5000), ngram_range=(1,3) (было (1,2))")
    print("  4. Добавлен class_weight='balanced' для борьбы с дисбалансом классов")
    print("="*80)
    
    # Проверка наличия файлов
    if not os.path.exists(TRAIN_FILE) or not os.path.exists(TEST_FILE):
        print(f"\nОшибка: Файлы {TRAIN_FILE} и {TEST_FILE} не найдены!")
        print("Сначала создайте train_data.json и test_data.json")
        return
    
    # Загрузка данных
    print("\n[1] Загрузка данных...")
    train_data, test_data = load_data(TRAIN_FILE, TEST_FILE)
    print(f"  Train: {len(train_data)} samples")
    print(f"  Test: {len(test_data)} samples")
    
    # Проверка, есть ли обученная модель
    models_exist = all(os.path.exists(path) for path in MODEL_PATHS.values())
    
    if models_exist:
        print("\n[2] Найдена сохранённая модель TF-IDF+LR")
        print("  Хотите использовать её или переобучить?")
        print("  1 - Использовать сохранённую модель (быстро)")
        print("  2 - Переобучить модель заново (рекомендуется для улучшений)")
        print("  3 - Выйти")
        
        choice = input("\nВаш выбор (1/2/3): ").strip()
        
        if choice == '1':
            print("\n[3] Использование сохранённой модели...")
            evaluate_approaches(train_data, test_data, force_retrain=False)
        elif choice == '2':
            print("\n[3] Переобучение модели с улучшениями...")
            evaluate_approaches(train_data, test_data, force_retrain=True)
        else:
            print("Выход.")
            return
    else:
        print("\n[2] Модель не найдена. Запуск обучения...")
        evaluate_approaches(train_data, test_data, force_retrain=True)
    
    print("\n" + "="*80)
    print("ГОТОВО!")
    print("="*80)
    print("\nСохранённые файлы:")
    print("  - confusion_matrix_rules.png")
    print("  - confusion_matrix_tfidf.png")
    print("  - confusion_matrix_hybrid.png")
    print(f"  - Модель TF-IDF сохранена в {MODEL_DIR}/")

if __name__ == "__main__":
    main()
