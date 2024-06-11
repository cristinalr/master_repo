import argparse
import os
import sys

import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_validate
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('csv_path')
    parser.add_argument('additional_test_data_csv_path')

    args = parser.parse_args()
    return args

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')

def main():

    args = get_args()
    csv_path = args.csv_path
    test_data_csv_path = args.additional_test_data_csv_path

    df = pd.read_csv(csv_path)
    df_extra_test = pd.read_csv(test_data_csv_path)

    sorted_df = df[df["patient_group"].isin(["PSC", "HC"])]

    # NaNs  to 0s
    sorted_df  = sorted_df.fillna(0)

    hc_df = sorted_df[sorted_df["patient_group"].isin(["HC"])]
    psc_df = sorted_df[sorted_df["patient_group"].isin(["PSC"])]

    frames = [hc_df, psc_df]
    save_csv_binary = pd.concat(frames)

    save_csv_binary.to_csv('binary_cases_used.csv', encoding='utf-8', index=False)

    # Separate training/testing data for HC and PSC separately and then join
    # So they will be equally represented
    train_hc, test_hc = train_test_split(hc_df, test_size=0.2)
    train_psc, test_psc = train_test_split(psc_df, test_size=0.2)

    full_csv_file = pd.concat((hc_df, psc_df), axis=0)
    labels_full = full_csv_file["patient_group"]
    full_csv_file = full_csv_file.drop(["patient_group"], axis=1)
    full_csv_file = full_csv_file.drop(["ID"], axis=1)
    full_csv_file = full_csv_file.drop(["StricOrDilatDuctPC"], axis=1)
    labels_full = labels_full.replace(['HC'], 0)
    labels_full = labels_full.replace(['PSC'], 1)

    train_df = pd.concat((train_hc,train_psc), axis=0)
    test_df = pd.concat((test_hc, test_psc), axis=0)

    # ------------------------------------------------------------------------------

    labels_train = train_df["patient_group"]
    labels_train = labels_train.replace(['HC'], 0)
    labels_train = labels_train.replace(['PSC'], 1)

    train_df = train_df.drop(["patient_group"], axis=1)
    train_df = train_df.drop(["ID"], axis=1)
    train_df = train_df.drop(["StricOrDilatDuctPC"], axis=1)
    # train_df = train_df.drop(["X"], axis=1)

    clf = RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced')

    # print(labels_train)

    clf.fit(train_df, labels_train)

    predictions_train = clf.predict(train_df)
    
    accuracy_score_training = accuracy_score(labels_train, predictions_train)

    tn, fp, fn, tp = confusion_matrix(labels_train, predictions_train).ravel()

    # print('RF features:', clf.feature_names_in_)

    # print(tn, fp, fn, tp)

    sensitivity_training = tp / (tp + fn)
    specificity_training = tn / (tn + fp)

    # print('Training Accuracy', accuracy_score_training)
    # print('Training Sensitivity',sensitivity_training)
    # print('Training Specificity',specificity_training)
    # print('-------------------------')

    # ------------------------------------------------------------------------------

    labels_test = test_df["patient_group"]
    labels_test = labels_test.replace(['HC'], 0)
    labels_test = labels_test.replace(['PSC'], 1)

    test_df = test_df.drop(["patient_group"], axis=1)
    test_df = test_df.drop(["ID"], axis=1)
    test_df = test_df.drop(["StricOrDilatDuctPC"], axis=1)
    # test_df = test_df.drop(["X"], axis=1)

    test_predictions = clf.predict(test_df)

    accuracy_score_test = accuracy_score(labels_test, test_predictions)

    tn, fp, fn, tp = confusion_matrix(labels_test, test_predictions).ravel()

    sensitivity_test = tp / (tp + fn)
    specificity_test = tn / (tn + fp)
    # print('Original test data')
    # print('Test Accuracy', accuracy_score_test)
    # print('Test Sensitivity', sensitivity_test)
    # print('Test Specificity', specificity_test)
    # print(tn, fp, fn, tp)

    # ------------------------------------------------------------------------------

    sorted_extra_df = df_extra_test[df_extra_test["patient_group"].isin(["PSC", "HC"])]

    sorted_extra_df  = sorted_extra_df.fillna(0)

    extra_test_df = sorted_extra_df[sorted_extra_df["patient_group"].isin(["HC", "PSC"])]
    # psc_df = sorted_df[sorted_df["patient_group"].isin(["PSC"])]

    labels_test_extra = extra_test_df["patient_group"]
    labels_test_extra = labels_test_extra.replace(['HC'], 0)
    labels_test_extra = labels_test_extra.replace(['PSC'], 1)

    extra_test_df = extra_test_df.drop(["patient_group"], axis=1)
    extra_test_df = extra_test_df.drop(["ID"], axis=1)  
    extra_test_df = extra_test_df.drop(["StricOrDilatDuctPC"], axis=1)

    test_predictions_2 = clf.predict(extra_test_df)

    accuracy_score_test_2 = accuracy_score(labels_test_extra, test_predictions_2)

    tn, fp, fn, tp = confusion_matrix(labels_test_extra, test_predictions_2).ravel()

    sensitivity_test = tp / (tp + fn)
    specificity_test = tn / (tn + fp)
    # print('-----------------------')
    # print('Additional testing data')
    # print('Test Accuracy', accuracy_score_test_2)
    # print('Test Sensitivity', sensitivity_test)
    # print('Test Specificity', specificity_test)
    # print(tn, fp, fn, tp,'\n')  

    # ------------------------------------------------------------------------------
    # CROSS VALIDATION

    scores = cross_val_score(clf, full_csv_file, labels_full, cv=5)
    # print(scores)
    # print("%0.2f accuracy with a standard deviation of %0.2f\n\n" % (scores.mean(), scores.std()))
    # print('-------')

    # ------------------------------------------------------------------------------
    # CROSS VALIDATE - 5 FOLDS

    cv_results = cross_validate(clf, full_csv_file, labels_full, cv=5, return_indices=True, return_estimator=True)
    # cv_results = cross_validate(clf, full_csv_file, labels_full, cv=5)
    list_folds = []
    for it in ((cv_results['indices']['test'])):
        list_folds.append(it)

    for fold in list_folds:
        fold_idx_list = list(fold)
        fold_df = full_csv_file.iloc[fold_idx_list]
        fold_labels = labels_full.iloc[fold_idx_list]

        permutation_this_fold = permutation_importance(clf, fold_df, fold_labels)
        max_importance_val = np.max(permutation_this_fold.importances_mean)
        max_importance_idx = np.where(permutation_this_fold.importances_mean == max_importance_val)
        
        # print(max_importance_idx)
        # print(max_importance_val)

        

    # ------------------------------------------------------------------------------
    # PERMUTATION on full file (training + testing)

    result_all = permutation_importance(clf, full_csv_file, labels_full)
    # print((result_all.importances_mean))
    # print(result_all)
    max_importance_val = np.max(result_all.importances_mean)
    max_importance_idx = np.where(result_all.importances_mean == max_importance_val)

    # print((result_all.importances_mean))
    # print(np.sort(result_all.importances_mean))
    # print(max_importance_idx)
    # print()

    # --------------------------------------------------------------------------
    # PCA

    scaler = StandardScaler()
    scaler.fit(test_df)
    scaled_data = scaler.transform(test_df)
    pca = PCA()
    # pca = PCA(whiten=True)
    # pca = PCA(n_components=3)
    pca.fit_transform(scaled_data)
    X = pca.transform(scaled_data)
    # print('explained variance ratio: ')
    # print(pca.explained_variance_ratio_)

    comps  = pca.components_

    # ------------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------------

    L = np.arange(len(pca.explained_variance_ratio_))

    # plt.bar(L, (pca.explained_variance_ratio_), bottom=None, align='center')
    # plt.show()

    h2 = np.cumsum(pca.explained_variance_ratio_)
    idx = np.asarray([i for i in range(len(comps))])

    # plt.bar(L, h2, bottom=None, align='center')
    # plt.axhline(y = 0.95, color = 'r', linestyle = '-')
    # # plt.axhline(y = 0.99, color = 'k', linestyle = '-')
    # plt.title("Cumulative sum of explained variance ratio for PCA")
    # plt.xlabel("Components")
    # plt.ylabel("Variance")
    # plt.xticks(idx)
    # plt.show()

    fig, axs = plt.subplots(2)
    
    h3 = comps[0]
    L3 = np.arange(len(h3))
    h4 = comps[1]
    h5 = comps[2]
    h6 = comps[3]
    h7 = comps[4]
    h8 = comps[5]
    h9 = comps[6]

    c = ['tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 
        'tomato', 'palegreen', 'palegreen', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 'tomato', 
        'cyan', 'cyan', 'cyan', 'cyan']

    b1 = axs[0].bar(L3, h3, bottom=None, align='center', color=c)
    # b2 = axs[1].bar(L3, h4, bottom=None, align='center', color=c)
    # b3 = axs[2].bar(L3, h5, bottom=None, align='center', color=c)
    # b4 = axs[3].bar(L3, h6, bottom=None, align='center', color=c)
    # b5 = axs[4].bar(L3, h7, bottom=None, align='center', color=c)
    # b6 = axs[5].bar(L3, h8, bottom=None, align='center', color=c)
    b7 = axs[1].bar(L3, h9, bottom=None, align='center', color=c)


    labels_legend = ['Dilatations and strictures', 'Duct length measurements', 'Diameter range']

    # axs[0].set_title('Component 1')
    # axs[0].set_xlabel('MRCP+ metric')
    axs[0].set_ylabel('Variance')
    axs[0].set_ylim(-0.425, 0.44)

    # # axs[1].set_title('Component 2')
    # # axs[1].set_xlabel('MRCP+ metric')
    # axs[1].set_ylabel('Variance')
    # axs[1].set_ylim(-0.425, 0.44)

    # # axs[2].set_title('Component 3')
    # # axs[2].set_xlabel('MRCP+ metric')
    # axs[2].set_ylabel('Variance')
    # axs[2].set_ylim(-0.425, 0.44)

    # # axs[3].set_title('Component 4')
    # # axs[1].set_xlabel('MRCP+ metric')
    # axs[3].set_ylabel('Variance')
    # axs[3].set_ylim(-0.425, 0.44)

    # # axs[4].set_title('Component 5')
    # # axs[4].set_xlabel('MRCP+ metric')
    # axs[4].set_ylabel('Variance')
    # axs[4].set_ylim(-0.425, 0.44)

    # # axs[5].set_title('Component 6')
    # # axs[0].set_xlabel('MRCP+ metric')
    # axs[5].set_ylabel('Variance')
    # axs[5].set_ylim(-0.425, 0.44)

    # axs[6].set_title('Component 7')
    axs[1].set_xlabel('MRCP+ metric')
    axs[1].set_ylabel('Variance')
    axs[1].set_ylim(-0.425, 0.44)

    plt.show()

    # plt.bar(L3, h3, bottom=None, align='center', color=c)
    # plt.show()



    # -----------------------------------------------
    # Re-train Random Forest with 7 most important components (from PCA analysis)
    # -----------------------------------------------

    keep_comps = comps[0:7]
    

    rf_2 = RandomForestClassifier(max_depth=2, random_state=0, class_weight='balanced')
    pca_2 = PCA(n_components=7)

    pca_2.fit(train_df)
    X = pca_2.transform(train_df)

    # print(pca.components_)
    
    rf_2.fit(X, labels_train)

    pca_2.fit(extra_test_df)
    X_2 = pca_2.transform(extra_test_df)

    # print('------\nAfter PCA\n------')

    predictions_train = rf_2.predict(X)
    accuracy_score_training = accuracy_score(labels_train, predictions_train)
    tn, fp, fn, tp = confusion_matrix(labels_train, predictions_train).ravel()
    # print(accuracy_score_training)
    # print(tn, fp, fn, tp)

    sensitivity_training = tp / (tp + fn)
    specificity_training = tn / (tn + fp)
    # print('Training Accuracy', accuracy_score_training)
    # print('Training Sensitivity',sensitivity_training)
    # print('Training Specificity',specificity_training)

    # -------------------------------------------
    # Find most important components
    # -------------------------------------------

    result_all = permutation_importance(rf_2, X, labels_train)
    max_importance_val = np.max(result_all.importances_mean)
    max_importance_idx = np.where(result_all.importances_mean == max_importance_val)


    importances = result_all.importances_mean
    len_comps = np.arange(len(importances))
    plt.bar(len_comps, importances, bottom=None, align='center', color=c)
    plt.show()

    # ------------------------------------------------------------------
    # CROSS-VALIDATION // PERMUTATION IMPORTANCE ON PCA
    # ------------------------------------------------------------------

    cv_results = cross_validate(rf_2, X, labels_train, cv=5, return_indices=True, return_estimator=True)
    # cv_results = cross_validate(clf, full_csv_file, labels_full, cv=5)

    list_folds = []
    for it in ((cv_results['indices']['test'])):
        list_folds.append(it)


    cc = 0
    for fold in list_folds:
        
        fold_idx_list = list(fold)
        fold_df = X[fold_idx_list]
        fold_df = X[tuple(fold), :]
        fold_labels = labels_train.iloc[fold_idx_list]
        # fold_labels = labels_train[tuple(fold_idx_list), :]

        permutation_this_fold = permutation_importance(rf_2, fold_df, fold_labels)
        max_importance_val = np.max(permutation_this_fold.importances_mean)
        max_importance_idx = np.where(permutation_this_fold.importances_mean == max_importance_val)
        print("Fold ", cc, ":\n")
        print(max_importance_val)
        print(max_importance_idx, '\n')
        cc += 1



    
    

    return 0

if __name__=='__main__':
    sys.exit(main())