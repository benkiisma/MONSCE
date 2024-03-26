"""
This file is used to run the complete pipeline including the feature extraction, the cross-correlation analysis, the clustering analysis , the projections  as well as the classification.

Author: "Benkirane Ismail"
Email: "ibenkirane@mgb.org"
version: "1.0.0"
Date: 2023-10-19
"""
import os
import json
import socket


try:
    from utils import GENERAL, FEATURES, UTILITIES, CORRELATION, CrossCorrelation, CLASSIFIER
except ImportError as e:
    print("Error importing modules from 'utils'. Please ensure the 'utils.py' file is in the same repository as the 'feature_extraction.py' file.")
    raise e

def main():

    # Load configuration parameters
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)

    computer_name = socket.gethostname()

    path = config['paths'].get(computer_name, {}).get('data_path', '')
    repo_path = config['paths'].get(computer_name, {}).get('repo_path', '')

    include_neutral_emotion = config['include_neutral_emotion']
    subjects_to_remove = config['subjects_to_remove']
    desired_measurement = config['desired_measurement']
    nb_seconds = config['nb_seconds']
    groups = config['groups']

    print("--------------------------------------------------")
    print("------------ Start Of Feature Extraction ----------")
    print("--------------------------------------------------")

    for measurement in desired_measurement:
        print(f'Extracting features for {measurement} measurements...')

        if measurement == 'Empatica':
            increase_empatica_emotion_window = True
        else:
            increase_empatica_emotion_window = False

        general = GENERAL(path, include_neutral_emotion, [measurement], subjects_to_remove=subjects_to_remove, increase_empatica_emotion_window=increase_empatica_emotion_window)
        features = FEATURES([measurement], general.json_logging_filename)
        utilities = UTILITIES()

        data, _ = general.get_data()
        general.label_data(data)

        all_features, stand_features = features.extract_features(data)

        os.makedirs(f'computed_features/{measurement}', exist_ok=True)

        all_features.to_csv(f'computed_features/{measurement}/all_features.csv', index=False)
        stand_features.to_csv(f'computed_features/{measurement}/stand_features.csv', index=False)

        utilities.git_commit_push(repo_path, f'Computed new features for {measurement} measurements')

        if measurement in ['Empatica', 'Audio', 'GoPro', 'FaceReader']:

            print(f'Extracting discrete time features for {measurement} measurements with {nb_seconds}s windows...')

            all_features_windows, stand_features_windows = features.extract_discrete_time_window_features(data, nb_seconds=nb_seconds)

            all_features_windows.to_csv(f'computed_features/{measurement}/all_features_windows.csv', index=False)
            stand_features_windows.to_csv(f'computed_features/{measurement}/stand_features_windows.csv', index=False)

            utilities.git_commit_push(repo_path, f'Computed new features for {measurement} measurements with {nb_seconds}s windows')
        print("--------------------------------------------------")


    print('\n')
    print(f'Extracting features for all measurement...')
        
    general = GENERAL(path, include_neutral_emotion, desired_measurement, subjects_to_remove=subjects_to_remove)
    features = FEATURES(desired_measurement, general.json_logging_filename)
    utilities = UTILITIES()

    data, _ = general.get_data()
    general.label_data(data)

    all_features, stand_features = features.extract_features(data)
    
    all_features.to_csv('computed_features/all_features.csv', index=False)
    stand_features.to_csv('computed_features/stand_features.csv', index=False)

    utilities.git_commit_push(repo_path, 'Computed new features')

    print(f'Extracting discrete time features with {nb_seconds}s windows...')

    all_features_windows, stand_features_windows = features.extract_discrete_time_window_features(data, nb_seconds=nb_seconds)

    all_features_windows.to_csv(f'computed_features/all_features_windows.csv', index=False)
    stand_features_windows.to_csv(f'computed_features/stand_features_windows.csv', index=False)

    utilities.git_commit_push(repo_path, f'Computed new features for {nb_seconds}s windows')

    print("--------------------------------------------------")
    print("------------ End OF Feature Extraction ----------")
    print("--------------------------------------------------")
    print("\n")
    print("--------------------------------------------------")
    print("-------------Start Of the Analysis---------------")
    print("--------------------------------------------------")
    print('\n')

    desired_measurement = ['Empatica', 'Audio', 'FaceReader', 'GoPro']

    correlation = CORRELATION(desired_measurement)
    cross_correlation = CrossCorrelation(desired_measurement)
    classifier = CLASSIFIER()

    print("Extracting and saving correlation results...")
    correlation.save_all_correlation_results(desired_measurement=desired_measurement, subject_groups= groups, include_pairs=True, select_features=False)
    utilities.git_commit_push(repo_path,'Computed new correlation results')

    print("Extracting and saving cross-correlation results...")              
    cross_correlation.save_all_cross_correlation_results(desired_measurement=desired_measurement, subject_groups = groups, include_pairs = True, only_pairs=False)
    utilities.git_commit_push(repo_path,'Computed new cross-correlation results')

    print("Extracting the clustering results...")
    features.save_all_clustering_results(stand_features)
    utilities.git_commit_push(repo_path,'Computed new clustering results')

    print("Extracting all the projections...")
    features_names = utilities.get_feature_names(all_features, desired_measurement)
    _ = features.get_all_projections(stand_features, features_names, groups, nb_selected_features=15, save=True)
    utilities.git_commit_push(repo_path,'Computed new projections')

    print("Extracting classification results...")
    _ = classifier.get_classification_performances(stand_features, groups, augment_data=False, feature_selection=True, nb_features=15, reduce_dim=True, verbose=False, save=True)
    utilities.git_commit_push(repo_path,'Computed new classification results')

if __name__ == "__main__":
    main()

