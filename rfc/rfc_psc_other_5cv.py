import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
import sys

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('csv_path')
    return parser.parse_args()

def preprocess_data(df):
    df["binary_label"] = np.where(df["patient_group"] == "PSC", 1, 0)  # 1 for PSC, 0 for others
    features = df.drop(columns=["patient_group", "ID", "StricOrDilatDuctPC", "binary_label"])
    labels = df["binary_label"]
    return features, labels

def calculate_class_weights(labels):
    # Calculate class weights for binary classification
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
    return weight_dict

def evaluate_metrics(features, labels, model):
    predictions = model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)
    specificity, sensitivity, conf_dict = calculate_specificity_sensitivity(conf_matrix)
    return accuracy, conf_matrix, specificity, sensitivity, conf_dict

def calculate_specificity_sensitivity(conf_matrix):
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 0
    return specificity, sensitivity, {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}

def cross_validate(df, n_splits=5):
    features, labels = preprocess_data(df)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    results = {
        'Fold': [],
        'Metric': [],
        'Accuracy': [],
        'Specificity': [],
        'Sensitivity': [],
        'TP': [],
        'TN': [],
        'FP': [],
        'FN': []
    }
    
    fold = 1
    for train_index, test_index in kf.split(features):
        train_features, test_features = features.iloc[train_index], features.iloc[test_index]
        train_labels, test_labels = labels.iloc[train_index], labels.iloc[test_index]
        
        # Compute class weights
        weights = calculate_class_weights(train_labels)
        clf = RandomForestClassifier(max_depth=4, random_state=0, class_weight=weights)
        clf.fit(train_features, train_labels)
        
        # Evaluate on test data
        accuracy_test, conf_matrix_test, specificity_test, sensitivity_test, conf_dict_test = evaluate_metrics(test_features, test_labels, clf)
        
        # Evaluate on training data
        accuracy_train, conf_matrix_train, specificity_train, sensitivity_train, conf_dict_train = evaluate_metrics(train_features, train_labels, clf)
        
        # Append results for this fold
        results['Fold'].extend([fold, fold])
        results['Metric'].extend(['Training', 'Testing'])
        results['Accuracy'].extend([accuracy_train, accuracy_test])
        results['Specificity'].extend([specificity_train, specificity_test])
        results['Sensitivity'].extend([sensitivity_train, sensitivity_test])
        results['TP'].extend([conf_dict_train['TP'], conf_dict_test['TP']])
        results['TN'].extend([conf_dict_train['TN'], conf_dict_test['TN']])
        results['FP'].extend([conf_dict_train['FP'], conf_dict_test['FP']])
        results['FN'].extend([conf_dict_train['FN'], conf_dict_test['FN']])
        
        fold += 1
    
    results_df = pd.DataFrame(results)
    results_df.to_csv('cross_validation_metrics_results_psc.csv', index=False)
    print("Results saved to 'cross_validation_metrics_results.csv'")
    
    return results_df

def main():
    args = get_args()
    df = pd.read_csv(args.csv_path)
    metrics_df = cross_validate(df)
    print(metrics_df)

if __name__ == '__main__':
    sys.exit(main())