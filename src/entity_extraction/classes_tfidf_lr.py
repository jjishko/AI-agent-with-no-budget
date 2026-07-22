#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TF-IDF + Logistic Regression классификатор с кэшированием
"""

import os
import pickle
import numpy as np
from typing import List, Dict, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfidfLRClassifier:
    """Классификатор на основе TF-IDF + Logistic Regression с кэшированием"""
    
    def __init__(self, model_dir: str = './models'):
        self.model_dir = model_dir
        self.vectorizer = None
        self.classifier = None
        self.intent_list = None
        self.is_fitted = False
        
        self.model_paths = {
            'tfidf_vectorizer': os.path.join(model_dir, 'classes_tfidf_vectorizer.pkl'),
            'tfidf_classifier': os.path.join(model_dir, 'classes_tfidf_classifier.pkl'),
            'tfidf_intents': os.path.join(model_dir, 'classes_tfidf_intents.pkl'),
        }
        
        os.makedirs(model_dir, exist_ok=True)
    
    def _is_model_cached(self) -> bool:
        """Проверяет, есть ли сохранённая модель"""
        return all(os.path.exists(path) for path in self.model_paths.values())
    
    def _save_model(self):
        """Сохраняет модель на диск"""
        with open(self.model_paths['tfidf_vectorizer'], 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(self.model_paths['tfidf_classifier'], 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.model_paths['tfidf_intents'], 'wb') as f:
            pickle.dump(self.intent_list, f)
        print(f"  [TF-IDF] Модель сохранена в {self.model_dir}")
    
    def _load_model(self):
        """Загружает модель с диска"""
        with open(self.model_paths['tfidf_vectorizer'], 'rb') as f:
            self.vectorizer = pickle.load(f)
        with open(self.model_paths['tfidf_classifier'], 'rb') as f:
            self.classifier = pickle.load(f)
        with open(self.model_paths['tfidf_intents'], 'rb') as f:
            self.intent_list = pickle.load(f)
        self.is_fitted = True
        print(f"  [TF-IDF] Модель загружена из {self.model_dir}")
    
    def fit(self, queries: List[str], intents: List[str], force_retrain: bool = False):
        """Обучение модели с кэшированием"""
        
        if not force_retrain and self._is_model_cached():
            self._load_model()
            return self
        
        print("  [TF-IDF] Обучение модели...")
        print(f"  [TF-IDF] Количество запросов: {len(queries)}")
        
        # Векторизация с улучшенными параметрами
        self.vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(1, 3),
            lowercase=True,
            min_df=2,
            analyzer='word'
        )
        X = self.vectorizer.fit_transform(queries)
        print(f"  [TF-IDF] Размерность: {X.shape}")
        
        # Обучение классификатора
        self.intent_list = sorted(set(intents))
        self.classifier = LogisticRegression(
            max_iter=1500,
            C=1.0,
            random_state=42,
            solver='lbfgs',
            class_weight='balanced'
        )
        self.classifier.fit(X, intents)
        self.is_fitted = True
        
        # Сохраняем модель
        self._save_model()
        
        return self
    
    def predict(self, queries: List[str]) -> List[Dict]:
        """Предсказание для массива запросов"""
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
    
    def get_feature_importance(self, top_n: int = 20) -> Dict:
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
