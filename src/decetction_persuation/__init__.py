from .preprocessing import (
    clean_arabic_text,
    remove_length_outliers,
    augment_minority_classes_gpt4o,
    preprocess_dataset,
)
from .train_marbert import train_marbert, predict_marbert
from .train_arabert import train_arabert, predict_arabert

__all__ = [
    "clean_arabic_text",
    "remove_length_outliers",
    "augment_minority_classes_gpt4o",
    "preprocess_dataset",
    "train_marbert",
    "predict_marbert",
    "train_arabert",
    "predict_arabert",
]

def main() -> None:
    print("Hello from decetction-persuation!")
