# library doc string
'''
Predict Customer Churn Modules
Author: Linh Vu
Date: 14th April 2023
'''
# import libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, classification_report
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib
os.environ['QT_QPA_PLATFORM'] = 'offscreen'


keep_cols = ['Customer_Age', 'Dependent_count', 'Months_on_book',
             'Total_Relationship_Count', 'Months_Inactive_12_mon',
             'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
             'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
             'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
             'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
             'Income_Category_Churn', 'Card_Category_Churn']

cat_cols = [
    'Gender',
    'Education_Level',
    'Marital_Status',
    'Income_Category',
    'Card_Category']


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    return pd.read_csv(pth)


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''

    # Churn distribution
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(20, 10))
    df['Churn'].hist()
    plt.savefig(fname='./images/eda/churn_distribution.png')

    # Customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.savefig(fname='./images/eda/customer_age_distribution.png')

    # Marital status distribution
    plt.figure(figsize=(20, 10))
    df.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(fname='./images/eda/marital_status_distribution.png')

    # Total trans distribution
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(fname='./images/eda/total_trans_distribution.png')

    # Heatmap
    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(fname='./images/eda/heatmap.png')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name 
                [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for cat in category_lst:
        col_lst = []
        col_groups = df.groupby(cat).mean()['Churn']

        for val in df[cat]:
            col_lst.append(col_groups.loc[val])

        if response is not None:
            new_cat = cat + "_" + response
        else:
            new_cat = cat
        df[new_cat] = col_lst
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name 
                [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    df = encoder_helper(df=df, category_lst=cat_cols, response=response)
    y = df['Churn']
    X = pd.DataFrame()
    X[keep_cols] = df[keep_cols]

    # train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''

    # Save testing classification report of random forest
    test_clf_report_rf = classification_report(
        y_test, y_test_preds_rf, output_dict=True)
    plt.figure(figsize=(20, 10))
    sns.heatmap(pd.DataFrame(test_clf_report_rf).iloc[:-1, :].T, annot=True)
    plt.figtext(0.5, 0.01, "Random Forest Test", horizontalalignment='center')
    plt.savefig(fname='./images/results/rf_test_report.png')

    # Save training classification report of random forest
    train_clf_report_rf = classification_report(
        y_train, y_train_preds_rf, output_dict=True)
    plt.figure(figsize=(20, 10))
    sns.heatmap(pd.DataFrame(train_clf_report_rf).iloc[:-1, :].T, annot=True)
    plt.figtext(0.5, 0.01, "Random Forest Train", horizontalalignment='center')
    plt.savefig(fname='./images/results/rf_train_report.png')

    # Save testing classification report of logistic regression
    test_clf_report_lr = classification_report(
        y_test, y_test_preds_lr, output_dict=True)
    plt.figure(figsize=(20, 10))
    sns.heatmap(pd.DataFrame(test_clf_report_lr).iloc[:-1, :].T, annot=True)
    plt.figtext(
        0.5,
        0.01,
        "'Logistic Regression Test",
        horizontalalignment='center')
    plt.savefig(fname='./images/results/lr_test_report.png')

    # Save traing classification report of logistic regression
    train_clf_report_lr = classification_report(
        y_train, y_train_preds_lr, output_dict=True)
    plt.figure(figsize=(20, 10))
    sns.heatmap(pd.DataFrame(train_clf_report_lr).iloc[:-1, :].T, annot=True)
    plt.figtext(
        0.5,
        0.01,
        "'Logistic Regression Train",
        horizontalalignment='center')
    plt.savefig(fname='./images/results/lr_train_report.png')


def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    # Save the image
    plt.savefig(fname=output_pth)


def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    # RandomForestClassifier and LogisticRegression
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    # Grid search
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)

    lrc.fit(X_train, y_train)

    # Compute train, test predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # Plot ROC curve
    plt.figure(figsize=(15, 8))
    axis = plt.gca()
    plot_roc_curve(lrc, X_test, y_test, ax=axis, alpha=0.8)
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=axis, alpha=0.8)
    plt.savefig(fname='./images/results/roc_curve_result.png')

    classification_report_image(y_train, y_test,
                                y_train_preds_lr, y_train_preds_rf,
                                y_test_preds_lr, y_test_preds_rf)

    feature_importance_plot(
        model=cv_rfc,
        X_data=X_test,
        output_pth='./images/results/feature_importances.png')

    # save best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')


if __name__ == '__main__':
    # Read dataframe
    data = import_data(pth='./data/bank_data.csv')

    # Perform EDA
    perform_eda(df=data)

    # Feature engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df=data, response='Churn')

    # Model training, prediction and evaluation
    train_models(X_train=X_train,
                 X_test=X_test,
                 y_train=y_train,
                 y_test=y_test)
