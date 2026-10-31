from .preprocessing import (
    clean_arabic_text,
    remove_length_outliers,
    augment_minority_classes_gpt4o,
    preprocess_dataset,
)
from .train_marbert import train_marbert, predict_marbert
from .train_arabert import train_arabert, predict_arabert
from .baselines import (
    evaluate_trivial_classifier,
    train_baseline_tfidf_lr,
    evaluate_rule_genre_to_persuasion,
    evaluate_rule_negative_word_to_loaded_language,
    FREQUENT_NEGATIVE_WORDS,
)

__all__ = [
    "clean_arabic_text",
    "remove_length_outliers",
    "augment_minority_classes_gpt4o",
    "preprocess_dataset",
    "train_marbert",
    "predict_marbert",
    "train_arabert",
    "predict_arabert",
    "evaluate_trivial_classifier",
    "train_baseline_tfidf_lr",
    "evaluate_rule_genre_to_persuasion",
    "evaluate_rule_negative_word_to_loaded_language",
    "FREQUENT_NEGATIVE_WORDS",
]

def main() -> None:
    print("Hello from decetction-persuation!")
