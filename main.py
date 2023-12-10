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
Utilisation de catboost / xgboost / lightgbm / réseau de neurones comme algo de classification
Objectif : obtenir la meilleure classification possible sur kaggle.

RENDU 3 :
Utilisation des melspectrogrammes en entrée
Utilisation de Deep Learning
Objectif : obtenir la meilleure classification possible sur kaggle.

"""

import config
from classification import classifier_knn, classifier_random
from data_extraction import join_data_mfcc, librosa_extractor


if __name__ == '__main__':

    # FEATURE EXTRACTION
    librosa_extractor.main(config.MUSIC_TRAIN_DIR)
    
    # DATA EXTRACTION
    # join_data_mfcc.main(config.train_filename, 
    #                                     config.test_filename, 
    #                                     config.train_mfcc_filename, 
    #                                     config.test_mfcc_filename, 
    #                                     config.MFCC_TRAIN_DIR, 
    #                                     config.MFCC_TEST_DIR)
    
    # CLASSIFICATION
    # classifier_random.main(config.test_filename,
    #                         config.output_random_filename)
    # classifier_knn.main(config.train_mfcc_filename,
    #                     config.test_mfcc_filename,
    #                     config.output_knn_filename)
    