import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class NERDataset(Dataset):
    """Подготовка датасета для моделей NER"""

    def __init__(self, data, model_name='DeepPavlov/rubert-base-cased', max_length=128):
        """
        Args:
            data: список словарей
            model_name: имя модели для токенизатора
            max_length: максимальная длина последовательности
        """
        self.data = data
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Словари для преобразования меток
        self.label2id = {
            'O': 0,
            'B-unit': 1,
            'I-unit': 2,
            'B-equipment': 3,
            'I-equipment': 4,
            'B-tag': 5,
            'I-tag': 6,
            'B-parameter': 7,
            'I-parameter': 8,
            'B-symptom': 9,
            'I-symptom': 10,
            'B-document_type': 11,
            'I-document_type': 12,
            'B-vendor': 13,
            'I-vendor': 14,
            'B-model': 15,
            'I-model': 16,
            'B-unit_of_measure': 17,
            'I-unit_of_measure': 18,
            'B-limit': 19,
            'I-limit': 20,
            'B-time_range': 21,
            'I-time_range': 22,
            'B-action': 23,
            'I-action': 24,
            'B-quantity': 25,
            'I-quantity': 26
        }
        self.id2label = {v: k for k, v in self.label2id.items()}

        # Подготовка данных
        self.processed_data = []
        for item in data:
            tokens, labels = self._prepare_labels(item)
            self.processed_data.append({
                'tokens': tokens,
                'labels': labels,
                'text': item['text'],
            })

    def _prepare_labels(self, item):
        """Разметка BIO"""
        text = item['text']
        labels = ['O'] * len(text)

        for ann in item.get('label', []):
            start = ann['start']
            end = ann['end']
            entity_type = ann['labels'][0]

            # Размечаем оригинальные позиции
            for i in range(start, min(end, len(text))):
                labels[i] = f'B-{entity_type}' if i == start else f'I-{entity_type}'

        return text, labels

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        text = item['text']
        labels = item['labels']

        # Токенизация с выравниванием
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt',
            return_offsets_mapping=True
        )

        # Создаем метки для каждого токена
        label_ids = []
        offset_mapping = encoding['offset_mapping'][0]

        for offset in offset_mapping:
            if offset[0] == 0 and offset[1] == 0:
                # Специальные токены
                label_ids.append(self.label2id['O'])
            else:
                # Находим метку для текущего токена
                token_label = 'O'
                for pos in range(offset[0], min(offset[1], len(labels))):
                    if labels[pos] != 'O':
                        token_label = labels[pos]
                        break
                label_ids.append(self.label2id.get(token_label, self.label2id['O']))

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label_ids[:self.max_length]),
            'offset_mapping': offset_mapping,
            'text': text,
        }

    def get_label_mappings(self):
        """Возвращает словари для преобразования меток"""
        return self.label2id, self.id2label