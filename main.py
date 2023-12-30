"""
Estimate genre of each musical audio file.

The evaluation for this challenge metric is Categorical Accuracy: 
TruePositive / (TruePositive + FalsePositive) for all categories (genres).

RENDU 1 :
Soumettre sur le kaggle :
- une solution aléatoire
- une classification obtenue à partir de MFCCs + kNN

RENDU 2 : 
Utilisation de librosa (et/ou essentia) pour obtenir plus de features
Utilisation de catboost / xgboost / lightgbm / réseau de neurones 
comme algo de classification
Objectif : obtenir la meilleure classification possible sur kaggle.

RENDU 3 :
Utilisation des melspectrogrammes en entrée
Utilisation de Deep Learning
Objectif : obtenir la meilleure classification possible sur kaggle.

"""

import config
from classifiers import classifier_knn, classifier_random, classifier_catboost
from data_extractors import librosa_extractor


if __name__ == '__main__':
    print("\nStarting main.py")

    # FEATURE EXTRACTION

    # DATA EXTRACTION

    # CLASSIFICATION
    classifier_random.main('data/test.csv',
                            'output/output_random.csv')