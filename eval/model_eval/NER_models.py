from src.entity_extraction.rule_based import RuleBasedNER
from src.common.metric_calc import make_ner_metrics, save_ner_metrics
from src.common.misc import load_data
from pathlib import Path
from src.entity_extraction.rubert import RuBERT


train_path = Path("../../data/processed/NER_train.json")
test_path = Path("../../data/processed/NER_test.json")

if not train_path.exists() or not test_path.exists():
    from src.common.preprocessing import ner_prerocessing
    ner_prerocessing()

train_data = load_data(train_path)
test_data = load_data(test_path)

rb_name = "Rule-based"
rb = RuleBasedNER()

bert_name = "ruBERT"
bert = RuBERT()

rb_param_grid = {
    'learning_rate': [1e-5, 2e-5, 3e-5],
    'num_train_epochs': [3, 5],
    'per_device_train_batch_size': [8, 16]
}

bert.load_model('../../models/NER/best_rubert')
#bert.fit(train_data, rb_param_grid)

#rb_stats = make_ner_metrics(model=rb, test_data=test_data)
bert_stats = make_ner_metrics(model=bert, test_data=test_data)
#save_ner_metrics(bert_stats, bert_name)