import re
from typing import List, Dict, Any


class NERNormalizer:

    def __init__(self):
        # Правила нормализации для разных типов сущностей
        self.rules = {
            'equipment': self._normalize_equipment,
            'tag': self._normalize_tag,
            'unit': self._normalize_unit,
            'parameter': self._normalize_parameter,
            'document_type': self._normalize_document_type,
            'action': self._normalize_action,
            'vendor': self._normalize_vendor,
            'model': self._normalize_model,
            'unit_of_measure': self._normalize_unit_of_measure,
            'limit': self._normalize_limit,
            'time_range': self._normalize_time_range,
            'symptom': self._normalize_symptom,
            'quantity': self._normalize_quantity,
        }

    def _normalize_equipment(self, text: str) -> str:
        """Нормализация оборудования"""
        equipment_types = {
            r'насос[а-я]?': 'насос',
            r'теплообменник[а-я]*': 'теплообменник',
            r'колонн[а-я]*': 'колонна',
            r'печ[а-я]*': 'печь',
            r'компрессор[а-я]*': 'компрессор',
            r'турбин[а-я]*': 'турбина',
            r'реактор[а-я]*': 'реактор',
            r'фильтр[а-я]*': 'фильтр',
            r'сепаратор[а-я]*': 'сепаратор',
            r'трубопровод[а-я]*': 'трубопровод',
            r'емкост[а-я]*': 'емкость',
            r'резервуар[а-я]*': 'резервуар',
            r'вентилятор[а-я]*': 'вентилятор',
            r'конденсатор[а-я]*': 'конденсатор',
            r'испарител[а-я]*': 'испаритель',
            r'блок[а-я]*': 'блок',
            r'двигател[а-я]*': 'двигатель',
            r'регулятор[а-я]*': 'регулятор',
            r'колл?ектор[а-я]*': 'коллектор',
            r'резервн[а-я]*': 'резервный',
            r'агрегат[а-я]*': 'агрегат',
        }

        for pattern, normalized_type in equipment_types.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, normalized_type, text, flags=re.IGNORECASE)
                # Добавляем дефис между буквами и цифрами
                text = re.sub(r'([A-Z]{1,3})\s*(\d{1,4})', r'\1-\2', text, flags=re.IGNORECASE)
                break

        return text

    def _normalize_tag(self, text: str) -> str:
        """Нормализация тегов"""
        text = text.strip()

        tag_id_pattern = r'^([A-Z]{1,4})\s*-?\s*(\d{1,4})$'
        match = re.match(tag_id_pattern, text, re.IGNORECASE)

        if match:
            return f"{match.group(1).upper()}-{match.group(2)}"

        instrument_map = {
            r'датчик[а-я]*': 'датчик',
            r'уровнемер[а-я]*': 'уровнемер',
            r'расходомер[а-я]*': 'расходомер',
            r'манометр[а-я]*': 'манометр',
            r'термометр[а-я]*': 'термометр',
            r'термопар[а-я]*': 'термопара',
            r'преобразовател[а-я]*': 'преобразователь',
            r'регулятор[а-я]*': 'регулятор',
            r'клапан[а-я]*': 'клапан',
            r'задвижк[а-я]*': 'задвижка',
            r'кип': 'КИП',
            r'кип[а-я]*': 'КИП',
            r'контроллер[а-я]*': 'контроллер',
        }

        for pattern, normalized in instrument_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
                break

        return text

    def _normalize_unit(self, text: str) -> str:
        """Нормализация установок"""
        text = text.strip()

        # - Буквы (1-4) + опционально дефис + буквы/цифры
        code_pattern = r'^[А-ЯA-Z]{1,4}(-[А-ЯA-Z0-9]{1,4})?$|^[А-ЯA-Z]{2,4}$'

        if re.match(code_pattern, text, re.IGNORECASE):
            return text.upper()

        return text

    def _normalize_parameter(self, text: str) -> str:
        """Нормализация параметров"""
        param_map = {
            r'давлени[а-я]*': 'давление',
            r'температур[а-я]*': 'температура',
            r'расход[а-я]*': 'расход',
            r'уровн[а-я]*': 'уровень',
            r'напор[а-я]*': 'напор',
            r'частот[а-я]*': 'частота',
            r'вибраци[а-я]*': 'вибрация',
            r'плотност[а-я]*': 'плотность',
            r'вязкост[а-я]*': 'вязкость',
            r'концентраци[а-я]*': 'концентрация',
            r'состав[а-я]*': 'состав',
        }

        for pattern, normalized in param_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
                break

        return text

    def _normalize_document_type(self, text: str) -> str:
        """Нормализация типов документов"""
        doc_map = {
            r'инструкци[а-я]*': 'инструкция',
            r'регламент[а-я]*': 'регламент',
            r'паспорт[а-я]*': 'паспорт',
            r'алгоритм[а-я]*': 'алгоритм',
            r'положени[а-я]*': 'положение',
            r'стандарт[а-я]*': 'стандарт',
            r'руководств[а-я]*': 'руководство',
            r'методик[а-я]*': 'методика',
            r'пми': 'ПМИ',
        }

        for pattern, normalized in doc_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
                break

        return text

    def _normalize_action(self, text: str) -> str:
        """Нормализация действий"""
        action_map = {
            r'покаж[а-я]*': 'показать',
            r'найд[а-я]*': 'найти',
            r'сравн[а-я]*': 'сравнить',
            r'рассчита[а-я]*': 'рассчитать',
            r'откр[а-я]*': 'открыть',
            r'закр[а-я]*': 'закрыть',
            r'включ[а-я]*': 'включить',
            r'выключ[а-я]*': 'выключить',
            r'настро[а-я]*': 'настроить',
            r'измен[а-я]*': 'изменить',
            r'установ[а-я]*': 'установить',
            r'посмотр[а-я]*': 'посмотреть',
            r'провер[а-я]*': 'проверить',
        }

        for pattern, normalized in action_map.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
                break

        return text

    def _normalize_vendor(self, text: str) -> str:
        """Нормализация вендоров"""
        return text.title()

    def _normalize_model(self, text: str) -> str:
        """Нормализация моделей"""
        return re.sub(r'([A-Z]{2,})(\d+)', r'\1-\2', text, flags=re.IGNORECASE)

    def _normalize_unit_of_measure(self, text: str) -> str:
        """Нормализация единиц измерения"""
        text = text.strip().lower()

        word_to_symbol = {
            r'процент[а-я]*': '%',
            r'литр[а-я]*': 'л',
            r'метр[а-я]*': 'м',
            r'килограмм[а-я]*': 'кг',
            r'тонн?[а-я]*': 'т',
            r'секунд[а-я]*': 'с',
            r'минут[а-я]*': 'мин',
            r'час[а-я]*': 'ч',
            r'сутк[а-я]*': 'сут',
            r'градус[а-я]*': '°C',
            r'паскал[а-я]*': 'Па',
            r'мегапаскал[а-я]*': 'МПа',
            r'килопаскал[а-я]*': 'кПа',
            r'бар[а-я]*': 'бар',
            r'атмосфер[а-я]*': 'атм',
            r'киловатт[а-я]*': 'кВт',
            r'ватт[а-я]*': 'Вт',
            r'джоул[а-я]*': 'Дж',
            r'калори[а-я]*': 'кал',
            r'герц[а-я]*': 'Гц',
            r'оборот[а-я]*': 'об',
        }

        for pattern, symbol in word_to_symbol.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, symbol, text, flags=re.IGNORECASE)

        # Заменяем "за" и другие предлоги на "/"
        text = re.sub(r'\s+за\s+', '/', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+в\s+', '/', text, flags=re.IGNORECASE)

        # Убираем лишние пробелы
        text = ' '.join(text.split())

        return text

    def _normalize_limit(self, text: str) -> str:
        """Нормализация ограничений"""
        text = text.strip()

        code_pattern = r'^[A-Za-z]{1,2}\.?$|^(alarm|high|low)$'

        if re.match(code_pattern, text, re.IGNORECASE):
            return text.upper()

        limit_words = {
            r'уставк[а-я]?': 'уставка',
            r'предел[а-я]?': 'предел',
            r'границ[а-я]?': 'граница',
            r'лимит[а-я]>': 'лимит',
            r'порог[а-я]?': 'порог',
        }

        for pattern, normalized in limit_words.items():
            if re.search(pattern, text, re.IGNORECASE):
                text = re.sub(pattern, normalized, text, flags=re.IGNORECASE)
                break

        return text

    def _normalize_time_range(self, text: str) -> str:
        """Нормализация временных диапазонов"""
        return text.lower()

    def _normalize_symptom(self, text: str) -> str:
        """Нормализация симптомов"""
        return text.lower()

    def _normalize_quantity(self, text: str) -> str:
        """Нормализация количества"""
        return re.sub(r'(\d)\s+(\d)', r'\1\2', text)

    def normalize_entity_text(self, text: str, entity_type: str) -> str:
        """Нормализация текста сущности в зависимости от типа"""
        if not text:
            return text

        entity_type_lower = entity_type.lower() if entity_type else ''

        # Ищем подходящее правило
        for type_key, normalize_func in self.rules.items():
            if type_key in entity_type_lower:
                return normalize_func(text)

        # Если тип не найден, возвращаем как есть
        return text

    def normalize_label(self, label: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация одной метки"""
        if not label:
            return label

        normalized_label = label.copy()

        # Нормализуем текст в зависимости от типа
        if 'labels' in normalized_label and normalized_label['labels']:
            entity_type = normalized_label['labels'][0]
            original_text = normalized_label.get('text', '')

            normalized_label['normalized'] = self.normalize_entity_text(
                original_text,
                entity_type)

        return normalized_label

    def normalize_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Нормализация целого элемента данных"""
        if not item or 'label' not in item:
            return item

        normalized_item = item.copy()

        # Нормализуем каждую метку
        if 'label' in normalized_item:
            normalized_item['label'] = [
                self.normalize_label(label)
                for label in normalized_item['label']
            ]

            # Сортируем по start
            normalized_item['label'] = sorted(
                normalized_item['label'],
                key=lambda x: x.get('start', 0)
            )

        return normalized_item

    def normalize_dataset(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Нормализация всего датасета"""
        return [self.normalize_item(item) for item in data]