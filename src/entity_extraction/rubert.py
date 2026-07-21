import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from sklearn.model_selection import KFold
from seqeval.metrics import f1_score, classification_report
import torch
from src.entity_extraction.data_preparing import NERDataset
import json
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


class RuBERT:
    """Модель NER на основе RuBERT"""

    def __init__(self, model_name='DeepPavlov/rubert-base-cased', n_folds=5, **training_args):
        """
        Args:
            model_name: имя предобученной модели
            n_folds: количество фолдов для кросс-валидации
            **training_args: аргументы для TrainingArguments
        """
        self.model_name = model_name
        self.n_folds = n_folds
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Настройки обучения по умолчанию
        self.training_args = {
            'output_dir': '../../models/NER',
            'num_train_epochs': 3,
            'per_device_train_batch_size': 16,
            'per_device_eval_batch_size': 16,
            'eval_strategy': 'epoch',
            'save_strategy': 'epoch',
            'logging_steps': 100,
            'load_best_model_at_end': True,
            'metric_for_best_model': 'f1',
            'greater_is_better': True,
            'fp16': torch.cuda.is_available(),
            'report_to': 'none'
        }
        self.training_args.update(training_args)

        self.tokenizer = None
        self.model = None
        self.label2id = None
        self.id2label = None
        self.best_model = None
        self.best_params = None
        self.optuna_study = None

    def _compute_metrics(self, eval_pred):
        """Вычисление метрик для Trainer"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        true_predictions = []
        true_labels = []

        for i in range(len(predictions)):
            pred = []
            label = []
            for j in range(len(predictions[i])):
                if labels[i][j] != -100:
                    pred.append(self.id2label[predictions[i][j]])
                    label.append(self.id2label[labels[i][j]])
            true_predictions.append(pred)
            true_labels.append(label)

        return {
            'f1': f1_score(true_labels, true_predictions),
            'report': classification_report(true_labels, true_predictions)
        }

    def _create_trainer(self, train_dataset, eval_dataset=None, **extra_args):
        """Создание Trainer с заданными датасетами и параметрами"""
        # Инициализация модели
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id)
        )

        # Data collator для паддинга
        data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True
        )

        # Копируем аргументы и добавляем eval_strategy
        training_args_dict = self.training_args.copy()
        training_args_dict.update(extra_args)
        training_args_dict.pop('evaluation_strategy', None)
        training_args_dict['eval_strategy'] = 'epoch' if eval_dataset else 'no'

        # Аргументы обучения
        training_args = TrainingArguments(**training_args_dict)

        # Создание Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=self._compute_metrics if eval_dataset else None
        )

        return trainer

    def _objective(self, trial, data):
        """Целевая функция для Optuna"""
        # Определяем пространство поиска гиперпараметров
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-5, log=True),
            'num_train_epochs': trial.suggest_int('num_train_epochs', 2, 4),
            'per_device_train_batch_size': trial.suggest_categorical('per_device_train_batch_size', [16, 32]),
            'warmup_ratio': trial.suggest_float('warmup_ratio', 0.0, 0.2),
            'weight_decay': trial.suggest_float('weight_decay', 0.0, 0.1),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.5, 2.0),
        }

        print(f"\nTrial {trial.number}: {params}")

        # Кросс-валидация для оценки параметров
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42 + trial.number)
        fold_scores = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(data)):
            train_data = [data[i] for i in train_idx]
            val_data = [data[i] for i in val_idx]

            train_dataset = NERDataset(train_data, self.model_name)
            val_dataset = NERDataset(val_data, self.model_name)

            # Обновляем label2id/id2label
            if self.label2id is None:
                self.label2id = train_dataset.label2id
                self.id2label = train_dataset.id2label

            trainer = self._create_trainer(train_dataset, val_dataset, **params)
            trainer.train()

            metrics = trainer.evaluate()
            fold_score = metrics.get('eval_f1', 0)
            fold_scores.append(fold_score)

            # Сообщаем Optuna о промежуточном результате (для pruning)
            trial.report(fold_score, step=fold)

            # Если результат плохой - прерываем trial
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_f1 = np.mean(fold_scores)
        std_f1 = np.std(fold_scores)

        print(f" F1: {mean_f1:.4f} (±{std_f1:.4f})")

        return mean_f1

    def tune_hyperparameters_optuna(self, data, n_trials=5, timeout=None, study_name='ner_optimization'):
        """Поиск лучших гиперпараметров через Optuna"""
        print("RUBERT: ПОИСК ГИПЕРПАРАМЕТРОВ ЧЕРЕЗ OPTUNA")
        print(f"Количество итераций: {n_trials}")
        print(f"Количество фолдов: {self.n_folds}")

        # Создаем study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

        self.optuna_study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=study_name,
            storage=None
        )

        self.optuna_study.optimize(
            lambda trial: self._objective(trial, data),
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True
        )

        best_trial = self.optuna_study.best_trial
        self.best_params = best_trial.params
        best_score = best_trial.value

        print("RUBERT: ЛУЧШИЕ ПАРАМЕТРЫ:")
        for key, value in self.best_params.items():
            print(f"  {key}: {value}")
        print(f"\n  Лучший F1: {best_score:.4f}")

        # Сохраняем результаты
        self._save_optuna_results()

        # Обновляем training_args лучшими параметрами
        self.training_args.update(self.best_params)

        return self.best_params, best_score

    def _save_optuna_results(self):
        """Сохранение результатов оптимизации Optuna"""
        if self.optuna_study is None:
            return

        # Сохраняем все trials
        results = []
        for trial in self.optuna_study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                results.append({
                    'number': trial.number,
                    'params': trial.params,
                    'value': trial.value,
                    'datetime': trial.datetime_start.isoformat() if trial.datetime_start else None
                })

        # Сортируем по качеству
        results.sort(key=lambda x: x['value'] if x['value'] is not None else 0, reverse=True)

        with open('../../models/NER/optuna_results.json', 'w') as f:
            json.dump({
                'best_params': self.best_params,
                'best_value': self.optuna_study.best_trial.value,
                'n_trials': len(results),
                'results': results,
                'best_trial_number': self.optuna_study.best_trial.number
            }, f, indent=2, default=str)

        # Сохраняем study
        try:
            import joblib
            joblib.dump(self.optuna_study, '../../models/NER/optuna_study.pkl')
        except:
            print("Не удалось сохранить study (joblib не установлен)")

    def load_optuna_study(self, path='../../models/NER/optuna_study.pkl'):
        """Загрузка сохраненного study для продолжения поиска"""
        try:
            import joblib
            self.optuna_study = joblib.load(path)
            print(f"Загружен study: {self.optuna_study.study_name}")
            print(f" Лучший F1: {self.optuna_study.best_trial.value:.4f}")
            return True
        except:
            print(" Не удалось загрузить study")
            return False

    def continue_optuna_search(self, data, n_trials=5):
        """Продолжение поиска гиперпараметров"""
        if self.optuna_study is None:
            print(" Нет активного study. Сначала запустите tune_hyperparameters_optuna()")
            return None

        print(f"\nПродолжаем поиск... (+{n_trials} итераций)")

        self.optuna_study.optimize(
            lambda trial: self._objective(trial, data),
            n_trials=n_trials
        )

        # Обновляем лучшие параметры
        self.best_params = self.optuna_study.best_trial.params
        best_score = self.optuna_study.best_trial.value

        print(f"\nНовый лучший F1: {best_score:.4f}")
        self.training_args.update(self.best_params)

        return self.best_params, best_score

    def fit(self, data, use_optuna=True, n_trials=5, resume_from_checkpoint=None):
        """Обучение модели"""
        # Создаем датасет для получения меток
        full_dataset = NERDataset(data, self.model_name)
        self.label2id = full_dataset.label2id
        self.id2label = full_dataset.id2label

        # Поиск гиперпараметров через Optuna
        if use_optuna:
            best_params, best_score = self.tune_hyperparameters_optuna(data, n_trials=n_trials)
            print(f"\nИспользуем лучшие параметры: {best_params}")
        else:
            print("\nПоиск гиперпараметров пропущен. Используем стандартные.")
            self.best_params = None

        # Если есть чекпоинт - загружаем его
        if resume_from_checkpoint:
            print(f"\nПродолжаем обучение с чекпоинта: {resume_from_checkpoint}")
            self.load_model(resume_from_checkpoint)

        # Обучаем финальную модель на всех данных
        print("\nОбучение финальной модели на всех данных")
        self._train_final_model(data, resume_from_checkpoint)

        return self.best_params

    def _train_final_model(self, data, resume_from_checkpoint=None):
        """Обучение финальной модели на всех данных"""
        full_dataset = NERDataset(data, self.model_name)

        # Подготавливаем аргументы для финального обучения
        training_args_dict = self.training_args.copy()
        training_args_dict['load_best_model_at_end'] = False
        training_args_dict['eval_strategy'] = 'no'

        # Если продолжается обучение, сохраняем чекпоинты чаще
        if resume_from_checkpoint:
            training_args_dict['save_strategy'] = 'steps'
            training_args_dict['save_steps'] = 50

        print("ФИНАЛЬНОЕ ОБУЧЕНИЕ С ПАРАМЕТРАМИ:")
        for key in ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size']:
            if key in training_args_dict:
                print(f"  {key}: {training_args_dict[key]}")

        # Создаем Trainer на всех данных
        trainer = self._create_trainer(full_dataset, eval_dataset=None, **training_args_dict)

        # Обучение с возможностью продолжить с чекпоинта
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)

        # Сохраняем лучшую модель
        self.best_model = trainer.model
        self.best_model.eval()
        self.model = self.best_model

        # Сохраняем финальную модель
        self.save_model('../../models/NER/best_rubert')

    def predict(self, data):
        """
        Предсказание сущностей для списка словарей

        Args:
            data: список словарей с полями 'text' и 'id'
                  Пример: [{"text": "...", "id": 123}, ...]

        Returns:
            list: список словарей с полями 'text', 'id', 'label'
        """
        if not isinstance(data, list):
            raise ValueError("На вход должен подаваться список")

        if len(data) == 0:
            return []

        if not isinstance(data[0], dict):
            raise ValueError("Элементы списка должны быть словарями")

        if self.model is None:
            raise ValueError("Модель не обучена. Сначала вызовите fit()")

        self.model.eval()
        results = []

        with torch.no_grad():
            for item in data:
                text = item['text']
                item_id = item.get('id', None)

                # Токенизация
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    max_length=128,
                    return_tensors='pt',
                    return_offsets_mapping=True
                )

                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                offset_mapping = encoding['offset_mapping'][0]

                # Предсказание
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                predictions = torch.argmax(outputs.logits, dim=2)[0]

                # Извлечение сущностей
                entities = []
                current_entity = None

                for idx, (pred, offset) in enumerate(zip(predictions, offset_mapping)):
                    if attention_mask[0][idx] == 0:
                        continue

                    # Безопасное извлечение ID
                    label = self.id2label[str(pred.item())]

                    if label.startswith('B-'):
                        if current_entity:
                            entities.append(current_entity)
                        entity_type = label.split('-')[1]
                        start_pos = int(offset[0])
                        end_pos = int(offset[1])
                        current_entity = {
                            'start': start_pos,
                            'end': end_pos,
                            'text': text[start_pos:end_pos],
                            'labels': [entity_type]
                        }
                    elif label.startswith('I-') and current_entity:
                        current_entity['end'] = int(offset[1])
                        current_entity['text'] = text[current_entity['start']:current_entity['end']]
                    else:
                        if current_entity:
                            entities.append(current_entity)
                            current_entity = None

                if current_entity:
                    entities.append(current_entity)

                # Объединение перекрывающихся сущностей
                entities = self._merge_entities(entities)

                # Формируем результат
                result = {
                    'text': text,
                    'label': entities
                }
                if item_id is not None:
                    result['id'] = item_id

                results.append(result)

        # Нормализация
        from src.common.normalizer import NERNormalizer
        normalizer = NERNormalizer()
        return normalizer.normalize_dataset(results)[0]

    def _merge_entities(self, entities):
        """Объединение перекрывающихся сущностей"""
        if not entities:
            return []

        merged = []
        entities.sort(key=lambda x: (x['start'], x['end']))

        current = entities[0]
        for entity in entities[1:]:
            if entity['start'] <= current['end']:
                current['end'] = max(current['end'], entity['end'])
                current['text'] = current['text'] + entity['text'][current['end'] - entity['start']:]
                for label in entity['labels']:
                    if label not in current['labels']:
                        current['labels'].append(label)
            else:
                merged.append(current)
                current = entity

        merged.append(current)
        return merged

    def save_model(self, path):
        """Сохранение модели"""
        if self.best_model:
            self.best_model.save_pretrained(path)
        elif self.model:
            self.model.save_pretrained(path)
        else:
            raise ValueError("Нет обученной модели для сохранения")

        self.tokenizer.save_pretrained(path)

        with open(f'{path}/label_mappings.json', 'w') as f:
            json.dump({
                'label2id': self.label2id,
                'id2label': self.id2label,
                'best_params': self.best_params,
                'optuna_best_value': self.optuna_study.best_trial.value if self.optuna_study else None
            }, f)

    def load_model(self, path):
        """Загрузка модели"""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path)
        self.model.to(self.device)
        self.best_model = self.model

        with open(f'{path}/label_mappings.json', 'r') as f:
            mappings = json.load(f)
            self.label2id = mappings['label2id']
            self.id2label = mappings['id2label']
            self.best_params = mappings.get('best_params', None)