## Study

Recognizing the intricate relationship between emotions and behaviors, this research aims to understand, how do multimodal measures of behavioral and physiological dynamics contribute to understanding the complex interplay of emotional states, identifying unique emotional signatures, and enhancing the accuracy of emotional state prediction in individuals. The research methodology was meticulously designed in collaboration with professional from the psychiatric department of the Massachusetts General Hospital (MGH), incorporation six distinct emotion measurement modalities, aiming to establish the degree of concordance among these diverse measurement strategies, their differential sensitivity to various emotion states, and their predictive validity regarding self-reported emotions.


## How to run the code

Configure the following parameters in the file `config.json`:

* `include_neutral_emotion (Boolean)`: Wheter or not to include the Neutral Emotional State as one of the emotions in the study.
* `subjects_to_remove (list)`: Subjects to not consider in the feature extraction and analysis.
* `desired_measurement (list)`: Set of emotion measurements from which the features will be extracted.
* `nb_seconds (int)`: The duration of the discrete time windows from which the features will be extracted.
* `groups (dict)`: The clusters to consider in the analysis.
* `paths (dict)`: The path for the data as well as for the git repository for automatic synchronization.

Once the desired parameters are configured, navigate to the repository were the code is saved, and run `python complete_pipeline.py` on the terminal. This script will read the data from the files, extract the features, and analyze the results. This script will also generate 3 main folders:

* `computed_features`: This folder will contain the features extracted from every emotion measurement that was selected in the `config.json` file as well as the features extracted from all the selected emotion measurement at the same time. In every subfolder, four files will be generated (except for Transcript and sre features were extracting discrete time window features was not an option):

    * `all_features.csv` - Raw extracted feature.
    * `stand_features.csv` - Standardized extracted features.
    * `all_features_windows` - Raw extracted features per discrete time windows.
    * `stand_features_windows` - Standardized extracted features per discrete time windows.

* `logs`: This folder will contains all the log files. Every time a feature extraction will start, a file will be generated with a timestamp, and will log all the steps and errors encountered during the extraction.

* `Analysis`: This folder contains all the results of the analysis. It containes the following subfolders:
    * `Classification`: For all the subjects then for every selected cluster, confusion matrices are computed for every trained classifier. Furthermore, a file titled `classifier_performance.json` groups the accuracies of all the classifiers as well as the selected feautures used.
    * `Cluster Analysis`: This folder groups the results of the clustering analysis, including the visual representation of the intersections between emotions and clusters(In the folder `Intersections`), the PCA projections (In the folder `PCA`) and the Sammon projections in the folder `Saamon`. Additionaly, three files are generated, including the grouping of the clusters (`SubjectClustering.json`), the consistency of these groupings (`SubjectConsistency.json`) as well as the features that describe every cluster (`FeatureImportance.json`).
    * `Correlation`: This folder groups the results of the correlation analysis per emotion measurement and across emotion measurement too. In every subfolder, there is:
        * Two plots describing the distribution of features across subjects: `CountPerSubjectPerEmotion.png` and `DistributionPerEmotion.png`. 
        * All the Extracted correlation coefficients: `features_correlation.json`.
        * The consistent correlated pairs across subjects and emotions: `correlation_consistency.json`.
        * `PairCount` Folder, with the consistency of the correlations across subjects and emotions are detailed. 
    * `Cross-Correlation`: This folder groups the results of the cross-correlation analysis per emotion measurement and across emotion measurement too. In every subfolder, there is:
        * Plot of the significantly shifted feature pairs (`Shifts_Across_Emotions.png`) as well as a file that groups them (`consistent_shifted_pairs.json`)
        * List of lags per feature pairs and per subject: `Lags.json`
        * List of the feature pairs that are not significantly shifted: `centered_feature_pairs.json`.
        * Lists of the shifts depending on the sign of the lagss: `no_shifts.json`, `positive_shifts`, `negative_shifts`. 
    * `Projections`: This folder groups all the projections, for every clsuter and for every trained classifier. Three projections are computed, PCA (`PCA.png`), Sammon Mapping (`Sammon.png`), t-SNE (`TSNE.png`).



  