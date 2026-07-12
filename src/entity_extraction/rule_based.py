import re
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from src.common.normalizer import NERNormalizer

class RuleBasedNER:
    def __init__(self):
        # Паттерны для каждого типа сущностей
        self.patterns = {
            'tag': [
                # Стандартные теги КИП: буквенный префикс + цифры
                r'\b([A-Z]{1,4}-\d{1,4})\b',
                r'\b([A-Z]{1,4}\d{1,4})\b',  # без дефиса
                r'\b(PT|TI|FIC|PIT|TIT|FIT|LIT|PIC|LIC|FIC|HIC|PDT|TDT|FDT|LDT|PDA|TDA|FDA|LDA)-\d{1,4}\b',
                # Префиксы без номера
                r'\b(PT|TI|FIC|PIT|TIT|FIT|LIT|PIC|LIC|FIC|HIC|PDT|TDT|FDT|LDT|PDA|TDA|FDA|LDA)\b'
                r'\b(датчик[а-я]*|[а-я]+мер[а-я]*|тег[а-я]*)\b'
            ],
            'unit': [
                r'\b(установк[а-я]+|лини[а-я]+|систем[а-я]+)\b',
                r'\b(ЭЛОУ-АВТ|ГПЗ|УПН|УКПГ|УПСВ|УПНГ|ГНП|КС|НС|ДНС)(-\d*)?\b',  # Буквенно-цифровые индексы
            ],
            'equipment': [
                r'\b(насос[а-я]*|регулятор[а-я]*|компрессор[а-я]*|теплообменник[а-я]*|колонна[а-я]*|печ[а-я]+|реактор[а-я]*|сепаратор[а-я]*|фильтр[а-я]*|емкост[а-я]*|резервуар[а-я]*|трубопровод[а-я]*|арматур[а-я]+|задвижк[а-я]+|клапан[а-я]*|вентил[а-я]+|блок[а-я]*|агрегат[а-я]*|машин[а-я]+)(?:\s?[А-Яa-zA-Z]-\d{1,4})\b',
            ],
            'parameter': [
                r'\b(давлени[а-я]*|температур[а-я]*|расход[а-я]*|уров[а-я]*|плотност[а-я]*|вязкост[а-я]*|концентраци[а-я]*|содержани[а-я]*|объем[а-я]*|скорост[а-я]*|напор[а-я]*|ток[а-я]*|напряжени[а-я]*|мощност[а-я]*|частот[а-я]*|вибраци[а-я]*|эффективност[а-я]*|производительност[а-я]*|дельта\s+давлени[а-я]*|перепад[а-я]*\s+давлени[а-я]*)\b',
            ],
            'symptom': [
                r'\b(растет|падает|скачет|колеблется|не\s+открывается|не\s+закрывается|завышает|занижает|выше\s+нормы|ниже\s+нормы|отклоняется|превышает|увеличивается|уменьшается|снижается|повышается|изменяется|нестабилен|не\s+работает|отказал|сбоит|заклинил|перегрелся|переохладился|забился|засорился|протекает|шумит|вибрирует|дымит|искрит)\b',
                r'\b(неисправен|неисправность|отказ|сбой|ошибка|авария|инцидент|отклонение)\b'
            ],
            'document_type': [
                r'\b(регламент[а-я]?|инструкци[а-я]?|схем[а-я]*|паспорт[а-я]?|руководств[а-я]?|методик[а-я]?|алгоритм[а-я]?|положени[а-я]?|стандарт[а-я]?|норматив[а-я]?|процедур[а-я]?|правил[а-я]?|приказ[а-я]*|распоряжени[а-я]?|ПМИ|пми)\b'
            ],
            'vendor': [
                r'\b(Yokogawa|Emerson|Siemens|ABB|Honeywell|Schneider|Rockwell|GE|Mitsubishi|Yokogawa\s+Electric|Emerson\s+Rosemount|Endress|Hauser|Endress\s+\+?\s*Hauser|KROHNE|SICK|Pepperl|Fuchs|Balluff|Turck|IFM|Banner|Omron|Keyence|Panasonic|Fuji|Phoenix\s+Contact|Weidmüller|Harting|Cisco|HP|Dell|IBM)\b'
            ],
            'model': [
                r'\b([A-Z]{2,4}\d{3,4}[A-Z]?)\b',
                r'\b([A-Z]+\d{3,4})\b',
                r'\b(модел[а-я]*\d{3,4})\b',
                r'\b([A-Z]{2,5}\s*\d{3,4}[A-Z]?)\b',
                r'\b(FLXA21|TDLS200|XMTR|DCS|S7-300|S7-400|S7-1200|S7-1500|PCS\s+7|SIMATIC|ET200|PROFIBUS|PROFINET)\b'
            ],
            'unit_of_measure': [
                r'\b(°C|°F|K|МПа|кПа|Па|бар|psi|мм\s?рт\.?\s?ст|м³/ч|м3/ч|л/с|л/мин|м³/с|нм³/ч|т/ч|кг/ч|кг/с|м/с|м/мин|об/мин|об/с|Гц|кГц|МГц|Вт|кВт|МВт|А|мА|кВ|МВ|Ом|кОм|МОм|%|ppm|ppb|мкм|мм|см|м|дм|км|м²|м³|л|дм³|мл|г|кг|т|мг|мкс|мс|мин|ч|сут|мес|год)\b',
                r'\b(градус(ов)?\s+Цельсия|градус(ов)?\s+Фаренгейта|Кельвин(а)?|мегапаскал(ь|я|ей)|килопаскал(ь|я|ей)|паскал(ь|я|ей)|бар(а|ов)?|миллиметр(ов)?\s+ртутн(ого|ом)\s+столб(а|е)?|куб(ических)?\s+метр(а|ов)?|литр(а|ов)?|тонн(а|ы|у)?|килограмм(а|ов)?|грамм(а|ов)?|процент(а|ов)?|миллиампер(а|ов)?|ампер(а|ов)?|вольт(а|ов)?|киловольт(а|ов)?|мегаватт(а|ов)?|киловатт(а|ов)?|ватт(а|ов)?|герц(а|ов)?)\b',
                r'\b(%)\b'
            ],
            'limit': [
                r'\b(Н|HH|L|LL|H|HH|L|LL)\b',
                r'\b(уставк[а-я]+|предел[а-я]*|границ[а-я]*|порог[а-я]*|лимит[а-я]*|аварийн[а-я]*\s+останов[а-я]*)\b'
            ],
            'time_range': [
                r'\b(за\s+сутк(и|у)|за\s+(поза)*вчера|за\s+смен(у|ы)|за\s+недел(ю|и)|за\s+месяц|за\s+год|последни[ее]\s+\d{1,2}\s+(час(а|ов)?|минут(у|ы)?|сутк(и|у)?|дн(я|ей)?|недел(ю|и)?|месяц(а|ев)?)|в\s+течени[ея]\s+\d{1,2}\s+(час(а|ов)?|минут(у|ы)?|сутк(и|у)?)|в\s+период\s+с\s+\d{2}\.\d{2}\.\d{4}\s+по\s+\d{2}\.\d{2}\.\d{4}|вчера|сегодня|за\s+прошлы[ее]\s+\d{1,2}\s+(час(а|ов)?|смен(у|ы)?)|в\s+\d{1,2}:\d{2})\b'
            ],
            'action': [
                r'\b(покаж[а-я]?|найди|найти|сравн[а-я]*|рассчита[а-я]*|вычисл[а-я]*|определ[а-я]*|провер[а-я]*|открой|открыть|закрой|закрыть|включи|включить|выключи|выключить|запуст[а-я]*|останов(и|ить)|измер(ь|ить)|проанализиру[а-я]*|оцени[а-я]*|просмотр[а-я]+|выведи|вывести|отобрази|отобразить|сформиру[а-я]*|подготов(ь|ить)|настрой|настроить|установ(и|ить)|измен(и|ить)|скорректируй)\b',
                r'\b(нуж(ен|на|но)\s+(расчет|анализ|проверка|настройка|замер|определение|вычисление|оценка|просмотр|отображение|формирование|подготовка|установка|изменение|корректировка|диагностика|поиск|сравнение))\b'
            ]
        }

        # Дополнительные составные паттерны
        self.compound_patterns = {
            'time_range': [
                r'с\s+\d{1,2}\s+по\s+\d{1,2}\s+[а-я]+\s+\d{4}',
                r'с\s+\d{2}\.\d{2}\.\d{4}\s+по\s+\d{2}\.\d{2}\.\d{4}'
            ]
        }

        # Паттерны для чисел
        self.number_pattern = re.compile(r'\b(не )?(ниже|выше\s)?(\d+[.,]?\d*(-\d+[.,]?\d*)?)\b')
        self.number_words = {
            'один': 1, 'два': 2, 'три': 3, 'четыре': 4, 'пять': 5,
            'шесть': 6, 'семь': 7, 'восемь': 8, 'девять': 9, 'десять': 10,
            'одиннадцать': 11, 'двенадцать': 12, 'тринадцать': 13, 'четырнадцать': 14,
            'пятнадцать': 15, 'двадцать': 20, 'тридцать': 30, 'сорок': 40,
            'пятьдесят': 50, 'шестьдесят': 60, 'семьдесят': 70, 'восемьдесят': 80,
            'девяносто': 90, 'сто': 100, 'двести': 200, 'триста': 300,
            'четыреста': 400, 'пятьсот': 500, 'шестьсот': 600, 'семьсот': 700,
            'восемьсот': 800, 'девятьсот': 900, 'тысяча': 1000, 'миллион': 1000000
        }

        # Компилируем паттерны
        self.compiled_patterns = {}
        for entity_type, patterns in self.patterns.items():
            self.compiled_patterns[entity_type] = [re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns]

        for entity_type, patterns in self.compound_patterns.items():
            if entity_type not in self.compiled_patterns:
                self.compiled_patterns[entity_type] = []
            self.compiled_patterns[entity_type].extend([re.compile(p, re.IGNORECASE | re.UNICODE) for p in patterns])

        # Инициализируем нормализатор
        self.normalizer = NERNormalizer()

    def _use_rules(self, text: str) -> list[tuple]:
        """Извлекает сущности из текста"""
        entities = []

        # Извлекаем сущности по паттернам
        for entity_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    start = match.start()
                    end = match.end()
                    # Проверяем, не пересекается ли с уже найденными сущностями
                    if not self._is_overlapping(entities, start, end):
                        entities.append((entity_type, start, end, match.group(0)))

        # Извлекаем quantity
        for match in self.number_pattern.finditer(text):
            start = match.start()
            end = match.end()
            if not self._is_overlapping(entities, start, end):
                # Проверяем, что число не является частью тега или модели
                if not self._is_part_of_entity(entities, start, end):
                    entities.append(('quantity', start, end, match.group(0)))

        # Извлекаем числа, записанные словами
        for word, value in self.number_words.items():
            pattern = re.compile(rf'\b{word}\b', re.IGNORECASE | re.UNICODE)
            for match in pattern.finditer(text):
                start = match.start()
                end = match.end()
                if not self._is_overlapping(entities, start, end):
                    entities.append(('quantity', start, end, match.group(0)))

        # Постобработка: разрешение конфликтов и edge cases
        entities = self._resolve_conflicts(entities)
        entities = self._handle_edge_cases(entities, text)

        return entities

    def _is_overlapping(self, entities: List[Tuple], start: int, end: int) -> bool:
        """Проверяет, пересекается ли интервал с существующими сущностями"""
        for _, s, e, _ in entities:
            if not (end <= s or start >= e):
                return True
        return False

    def _is_part_of_entity(self, entities: List[Tuple], start: int, end: int) -> bool:
        """Проверяет, является ли интервал частью существующей сущности"""
        for _, s, e, _ in entities:
            if s <= start and end <= e:
                return True
        return False

    def _resolve_conflicts(self, entities: List[Tuple]) -> List[Tuple]:
        """Разрешает конфликты между сущностями (приоритеты)"""
        # Приоритеты сущностей
        priority = {
            'model': 10,
            'tag': 9,
            'unit': 8,
            'equipment': 7,
            'vendor': 6,
            'parameter': 5,
            'symptom': 4,
            'limit': 3,
            'document_type': 2,
            'time_range': 1,
            'action': 0,
            'quantity': 0,
            'unit_of_measure': 0
        }

        # Сортируем по длине (сначала длинные) и по приоритету
        entities.sort(key=lambda x: (x[2] - x[1], priority.get(x[0], 0)), reverse=True)

        resolved = []
        for entity in entities:
            if not self._is_overlapping(resolved, entity[1], entity[2]):
                resolved.append(entity)

        return resolved

    def _handle_edge_cases(self, entities: List[Tuple], text: str) -> List[Tuple]:
        """Обрабатывает особые случаи из таксономии"""
        result = []
        i = 0
        while i < len(entities):
            entity_type, start, end, value = entities[i]

            # Если это equipment без номера, проверяем контекст
            if entity_type == 'equipment' and not re.search(r'[A-Za-zА-Яа-я]-\d', value):
                # Проверяем, не является ли это частью другого оборудования
                if i + 1 < len(entities):
                    next_type, next_start, next_end, next_value = entities[i + 1]
                    if next_type == 'quantity' and start == next_start:
                        # Объединяем с числом
                        result.append((entity_type, start, next_end, f"{value} {next_value}"))
                        i += 2
                        continue

            # Если это tag без префикса, пытаемся добавить из контекста
            if entity_type == 'tag' and not re.search(r'[A-Z]-\d', value):
                # Проверяем, есть ли префикс в том же предложении
                sentence = text[:start].split('.')[-1] + text[start:end] + text[end:].split('.')[0]
                prefix_match = re.search(r'\b([A-Z]{1,4})-\d+\b', sentence)
                if prefix_match:
                    prefix = prefix_match.group(1)
                    result.append((entity_type, start, end, f"{prefix}-{value}"))
                    i += 1
                    continue

            result.append((entity_type, start, end, value))
            i += 1

        return result

    def normalize_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """Нормализует предсказания с помощью NERNormalizer"""
        # Преобразуем в формат, ожидаемый нормализатором
        data = [{'text': '', 'entities': predictions}]
        normalized = self.normalizer.normalize_dataset(data)
        return normalized[0]['entities'] if normalized else []


    def extract_entities(self, text: str) -> dict[str, str | list[Any]]:
        """Предсказывает NER разметку"""
        entities = self._use_rules(text)

        # Сортируем по позиции в тексте
        entities.sort(key=lambda x: x[1])

        # Группируем сущности по позиции
        grouped = defaultdict(list)
        for entity_type, start, end, value in entities:
            key = (start, end, value)
            grouped[key].append(entity_type)

        labels = []
        for (start, end, value), entity_types in grouped.items():
            # Получаем нормализованную форму
            normalized_data = self.normalizer.normalize_dataset([
                {'text': text, 'label': [{'labels': [entity_types[0]], 'text': value}]}
            ])
            normalized_value = value
            if normalized_data and 'label' in normalized_data[0]:
                for ent in normalized_data[0]['label']:
                    if ent.get('text') == value:
                        normalized_value = ent.get('normalized', value)
                        break

            labels.append({
                'start': start,
                'end': end,
                'value': value,
                'text': text,
                'labels': entity_types,
                'normalized': normalized_value
            })

        return {'text': text, 'label': labels}



