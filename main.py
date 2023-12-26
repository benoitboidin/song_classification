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
from data_extractors import concat_features, concat_features_pd, librosa_extractor


if __name__ == '__main__':

    # FEATURE EXTRACTION
    # librosa_extractor.features_to_csv(config.MUSIC_TRAIN_DIR,
    #                                 config.train_librosa_features)
    # librosa_extractor.features_to_csv(config.MUSIC_TEST_DIR,
    #                                 config.test_librosa_features)
    
    # DATA EXTRACTION
    # concat_features.main(config.train_filename, 
    #                     config.test_filename, 
    #                     config.train_mfcc_filename, 
    #                     config.test_mfcc_filename, 
    #                     config.MFCC_TRAIN_DIR, 
    #                     config.MFCC_TEST_DIR)
    # concat_features_pd.main('data/train_mfcc.csv',
    #                         'data/train_librosa_features_sorted.csv',
    #                         'data/concat_output.csv')
    
    # CLASSIFICATION
    # classifier_random.main(config.test_filename,
    #                         config.output_random_filename)
    classifier_knn.main('data/train_features.csv',
                        'data/test_features.csv',
                        config.output_knn_filename)
    # classifier_catboost.main(config.train_mfcc_filename,
    #                         config.test_mfcc_filename,
    #                         config.output_catboost_filename)
    