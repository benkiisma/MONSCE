"""
This file is used to extract the features from the data.

Author: "Benkirane Ismail"
Email: "ibenkirane@mgb.org"
version: "1.0.0"
Date: 2023-10-19
"""

import os
import json
import socket


try:
    from utils import GENERAL, FEATURES, UTILITIES
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

    os.makedirs(f'computed_features', exist_ok=True)
    
    general = GENERAL(path, include_neutral_emotion, desired_measurement, subjects_to_remove=subjects_to_remove)
    features = FEATURES(desired_measurement, general.json_logging_filename)
    utilities = UTILITIES()

    print("Passed this")

    data, _ = general.get_data()
    general.label_data(data)

    all_features, stand_features = features.extract_features(data, save=True)
    
    utilities.git_commit_push(repo_path, 'Computed new features')

    print(f'Extracting discrete time features with {nb_seconds}s windows...')

    all_features_windows, stand_features_windows = features.extract_discrete_time_window_features(data, nb_seconds=nb_seconds, save=True)

    utilities.git_commit_push(repo_path, f'Computed new features for {nb_seconds}s windows')

if __name__ == "__main__":
    main()

