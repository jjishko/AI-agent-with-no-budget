#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Гибридный классификатор: Rules + TF-IDF+LR + Confidence
"""

from typing import List, Dict, Tuple
from .classes_rule_based import RuleBasedClassifier
from .classes_tfidf_lr import TfidfLRClassifier


class HybridClassifier:
    """
    Гибридный классификатор: Rules + TF-IDF+LR + Confidence
    Исправленные пороги:
    - rule_threshold: 0.7
    - model_threshold: 0.65
    """
    
    def __init__(self, rule_threshold: float = 0.7, model_threshold: float = 0.65, model_dir: str = './models'):
        self.rule_clf = RuleBasedClassifier()
        self.model_clf = None
        self.rule_threshold = rule_threshold
        self.model_threshold = model_threshold
        self.model_dir = model_dir
        self.is_fitted = False
    
    def fit(self, queries: List[str], intents: List[str], force_retrain: bool = False):
        """Обучение гибридного классификатора"""
        print("[Hybrid] Обучение TF-IDF+LR...")
        self.model_clf = TfidfLRClassifier(model_dir=self.model_dir)
        self.model_clf.fit(queries, intents, force_retrain)
        self.is_fitted = True
        return self
    
    def classify(self, query: str) -> Tuple[str, str, float, str]:
        """
        Классифицирует запрос
        Возвращает: (intent, route, confidence, level)
        level: 'high', 'medium', 'low'
        """
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
            elif model_conf >= 0.45:
                return model_intent, model_route, model_conf, 'medium'
            else:
                return 'clarification_needed', 'clarification_flow', model_conf, 'low'
        
        # 3. Fallback
        if rule_conf >= 0.4:
            return rule_intent, rule_route, rule_conf, 'medium'
        else:
            return 'clarification_needed', 'clarification_flow', rule_conf, 'low'
    
    def predict(self, queries: List[str]) -> List[Dict]:
        """Predict для массива запросов"""
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
