import argparse
import os
import sys

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('csv_path')

    args = parser.parse_args()
    return args

def main():

    args = get_args()
    csv_path = args.csv_path

    df = pd.read_csv(csv_path)

    df_mrcpp = df.iloc[:, 3:61]
    patient_group = df["Patient.Group"]
    df_mrcpp["Patient_group"] = patient_group

    # NaN to 0s
    df_mrcpp = df_mrcpp.fillna(0)

    hc_df = df_mrcpp[df_mrcpp["Patient_group"].isin(["HC"])]
    psc_df = df_mrcpp[df_mrcpp["Patient_group"].isin(["PSC"])]
    pbc_df = df_mrcpp[df_mrcpp["Patient_group"].isin(["PBC"])]
    aih_df = df_mrcpp[df_mrcpp["Patient_group"].isin(["AIH"])]

    # Separate training/testing data for 4 categories separately and then join
    # So they will be equally represented
    train_hc, test_hc = train_test_split(hc_df, test_size=0.2)
    train_psc, test_psc = train_test_split(psc_df, test_size=0.2)
    train_pbc, test_pbc = train_test_split(pbc_df, test_size=0.2)
    train_aih, test_aih = train_test_split(aih_df, test_size=0.2)

    train_df = pd.concat((train_hc,train_psc,train_pbc,train_aih), axis=0)
    test_df = pd.concat((test_hc, test_psc, test_pbc, test_aih), axis=0)

    # ------------------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------------------

    labels_train = train_df["Patient_group"]
    labels_train = labels_train.replace(['HC'], 0)
    labels_train = labels_train.replace(['PSC'], 1)
    labels_train = labels_train.replace(['PBC'], 2)
    labels_train = labels_train.replace(['AIH'], 3)

    train_df = train_df.drop(["Patient_group"], axis=1)

    clf = RandomForestClassifier(max_depth=2, random_state=0,  class_weight='balanced')

    clf.fit(train_df, labels_train)

    predictions_train = clf.predict(train_df)
    
    accuracy_score_training = accuracy_score(labels_train, predictions_train)
    # print('Accuracy', accuracy_score_training)

    # tn, fp, fn, tp = confusion_matrix(labels_train, predictions_train)

    # sensitivity_training = tp / (tp + fn)
    # specificity_training = tn / (tn + fp)


    print('Training Accuracy: ', accuracy_score_training)

    # ------------------------------------------------------------------------------
    # Testing
    # ------------------------------------------------------------------------------

    labels_test = test_df["Patient_group"]
    labels_test = labels_test.replace(['HC'], 0)
    labels_test = labels_test.replace(['PSC'], 1)
    labels_test = labels_test.replace(['PBC'], 2)
    labels_test = labels_test.replace(['AIH'], 3)

    test_df = test_df.drop(["Patient_group"], axis=1)

    test_predictions = clf.predict(test_df)

    accuracy_score_test = accuracy_score(labels_test, test_predictions)
    print('Testing Accuracy: ', accuracy_score_test)

    return 0

if __name__=='__main__':
    sys.exit(main())