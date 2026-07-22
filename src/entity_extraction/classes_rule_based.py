#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rule-based классификатор для промышленных запросов
"""

import re
from typing import Tuple, List, Dict


class RuleBasedClassifier:
    """Классификатор на основе правил с улучшенными правилами для tag_lookup"""
    
    def __init__(self):
        self.rules = self._build_rules()
    
    def _build_rules(self) -> Dict:
        """Строит правила для каждого интента"""
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
                'weight': 0.8
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
        
        if any(w in text_lower for w in ['найди', 'покажи', 'выдай', 'открой', 'найти']):
            return 'action'
        if any(w in text_lower for w in ['где', 'позици', 'располож', 'смонтирован', 'закреплен', 'размещен']):
            return 'location'
        if any(w in text_lower for w in ['сейчас', 'текущее', 'мгновенное', 'прямо сейчас', 'в реальном времени']):
            return 'value'
        if any(w in text_lower for w in ['истори', 'за сутки', 'за смену', 'тренд', 'график', 'динамик']):
            return 'history'
        
        return 'unknown'
    
    def classify(self, query: str) -> Tuple[str, str, float, Dict]:
        """
        Классифицирует запрос
        Возвращает: (intent, route, confidence, scores)
        """
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
            
            if tag_context == 'history':
                scores['data_analysis'] = max(scores.get('data_analysis', 0), 0.85)
                scores['tag_lookup'] = min(scores.get('tag_lookup', 0), 0.5)
            elif tag_context == 'value':
                scores['tag_lookup'] = max(scores.get('tag_lookup', 0), 0.9)
            elif tag_context == 'action':
                if any(w in text for w in ['инструкци', 'паспорт', 'документаци', 'описани']):
                    scores['doc_qa'] = max(scores.get('doc_qa', 0), 0.85)
                    scores['tag_lookup'] = min(scores.get('tag_lookup', 0), 0.4)
                else:
                    scores['tag_lookup'] = max(scores.get('tag_lookup', 0), 0.7)
            elif tag_context == 'location':
                scores['tag_lookup'] = max(scores.get('tag_lookup', 0), 0.85)
            else:
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
    
    def predict(self, queries: List[str]) -> List[Dict]:
        """Predict для массива запросов"""
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
