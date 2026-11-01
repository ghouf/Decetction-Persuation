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
from .nli_zeroshot import (
    evaluate_nli_task1a_zeroshot,
    compare_nli_formulations_task1a,
    evaluate_sentiment_transfer_b2,
    NLI_HYPOTHESES,
)
from .part_d_frozen_plm import (
    extract_bert_embeddings,
    run_part_d1,
    train_part_d2,
    visualize_attention_on_examples,
    BiLSTMAttentionHead,
)
from .visualizations import (
    plot_threshold_tradeoff,
    plot_per_class_f1,
    plot_predictions_distribution,
    inspect_classification_errors,
)
from .part_f_evaluation import (
    run_f1_prior_shift,
    run_f2_cross_genre,
    run_f3_length_and_genre,
    run_f4_charged_words_ablation,
    run_f5_significance_bootstrap_mcnemar,
    run_f6_classify_30_errors,
)
from .part_g_final_summary import (
    measure_model_efficiency,
    plot_multilabel_confusion_matrix,
    generate_part_g_summary_table,
    run_part_g,
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
    "evaluate_nli_task1a_zeroshot",
    "compare_nli_formulations_task1a",
    "evaluate_sentiment_transfer_b2",
    "NLI_HYPOTHESES",
    "extract_bert_embeddings",
    "run_part_d1",
    "train_part_d2",
    "visualize_attention_on_examples",
    "BiLSTMAttentionHead",
    "plot_threshold_tradeoff",
    "plot_per_class_f1",
    "plot_predictions_distribution",
    "inspect_classification_errors",
    "run_f1_prior_shift",
    "run_f2_cross_genre",
    "run_f3_length_and_genre",
    "run_f4_charged_words_ablation",
    "run_f5_significance_bootstrap_mcnemar",
    "run_f6_classify_30_errors",
    "measure_model_efficiency",
    "plot_multilabel_confusion_matrix",
    "generate_part_g_summary_table",
    "run_part_g",
]

def main() -> None:
    print("Hello from decetction-persuation!")
