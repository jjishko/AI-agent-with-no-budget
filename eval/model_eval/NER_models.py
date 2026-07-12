from src.entity_extraction.rule_based import RuleBasedNER
from src.common.metric_calc import make_ner_metrics, save_ner_metrics
from src.common.misc import load_data
from pathlib import Path

train_path = Path("../../data/processed/NER_train.json")
test_path = Path("../../data/processed/NER_test.json")

if not train_path.exists() or not test_path.exists():
    from src.common.preprocessing import ner_prerocessing

    ner_prerocessing()

test = load_data(test_path)

rb_name = "Rule-based"
rb = RuleBasedNER()

stats = make_ner_metrics(model=rb, test_data=test)
#save_ner_metrics(stats, rb_name)