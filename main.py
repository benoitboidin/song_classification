import config
from classification import classifier_knn, classifier_random
import data_extraction.join_data_mfcc


if __name__ == '__main__':
    
    # PATHS
    MUSIC_TEST_DIR = config.MUSIC_TEST_DIR
    MUSIC_TRAIN_DIR = config.MUSIC_TRAIN_DIR

    MFCC_TEST_DIR = config.MFCC_TEST_DIR
    MFCC_TRAIN_DIR = config.MFCC_TRAIN_DIR

    train_filename = config.train_filename
    test_filename = config.test_filename

    test_mfcc_filename = config.test_mfcc_filename
    train_mfcc_filename = config.train_mfcc_filename

    output_random_filename = config.output_random_filename
    output_knn_filename = config.output_knn_filename
    output_xkboost_filename = config.output_xkboost_filename
    output_dnn_filename = config.output_dnn_filename

    # DATA EXTRACTION
    data_extraction.join_data_mfcc.main(train_filename, 
                                        test_filename, 
                                        train_mfcc_filename, 
                                        test_mfcc_filename, 
                                        MFCC_TRAIN_DIR, 
                                        MFCC_TEST_DIR)