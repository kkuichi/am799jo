Bachelor Thesis Project
Využitie transformerov v detekcii toxicity na sociálnych sieťach

Author: Artem Mykhailichenko

This archive contains Python source code and supporting files used in the experimental part of the bachelor thesis.

Project structure and file description

train_bert.py
    Fine-tunes the BERT-base-uncased model for binary toxicity detection on the Jigsaw Toxic Comment Classification Challenge dataset.

train_roberta.py
    Fine-tunes the RoBERTa-base model for binary toxicity detection on the same dataset.

train_distilbert.py
    Fine-tunes the DistilBERT-base-uncased model for binary toxicity detection.

train_albert.py
    Fine-tunes the ALBERT-base-v2 model for binary toxicity detection.

train_baselines.py
    Trains classical baseline models using TF-IDF features:
    - Logistic Regression
    - Linear SVM
    - Multinomial Naive Bayes

cross_dataset_evaluation.py
    Evaluates all trained models on multiple datasets:
    - Jigsaw validation set
    - Civil Comments
    - TweetEval Offensive

error_analysis.py
    Performs manual error analysis for the BERT model.
    The script identifies false negatives, false positives, and borderline cases.

benchmark_transformer_inference.py
    Measures inference latency and throughput for the transformer models.

plot_training_curves.py
    Generates training curves from training_history.csv files.

manual_bert_inference_test.py
    Runs manual prediction examples using the trained BERT model.

requirements.txt
    Contains the Python libraries required to run the scripts.

Dataset information

The file train.csv contains the Jigsaw Toxic Comment Classification Challenge dataset and must be placed in the root directory of the project.

The Civil Comments and TweetEval Offensive datasets are not included in the archive. They are loaded automatically through the Hugging Face Datasets library when running cross_dataset_evaluation.py.

Basic usage

1. Install Python 3.10 or newer.
2. Install the required libraries:

   pip install -r requirements.txt

3. Make sure that train.csv is located in the project root directory.
4. Run the required script, for example:

   python train_bert.py

Output files

Depending on the executed script, the project may generate:
    - trained model folders,
    - training_history.csv files,
    - baseline_results.csv,
    - cross_dataset_results.csv,
    - false_negatives.csv,
    - false_positives.csv,
    - borderline_cases.csv,
    - PNG graphs with training curves.

Note

Training transformer models is computationally demanding. A CUDA-compatible NVIDIA GPU is recommended.
