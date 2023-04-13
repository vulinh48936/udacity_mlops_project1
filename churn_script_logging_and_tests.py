# library doc string
'''
Predict Customer Churn Modules testing and logging
Author: Linh Vu
Date: 14th April 2023
'''
# import libraries
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
# Load dataframe
    df = cls.import_data("./data/bank_data.csv")
    try:
        perform_eda(df=df)
        logging.info("Testing perform_eda: SUCCESS")
    except KeyError as e:
        logging.error('Column "%s" not found', e.args[0])

    # Check existing image churn_distribution.png
    try:
        assert os.path.isfile("./images/eda/churn_distribution.png") is True
        logging.info('Saving image %s: SUCCESS', 'churn_distribution.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'churn_distribution.png')
        raise e

    # Check existing image customer_age_distribution.png
    try:
        assert os.path.isfile(
            "./images/eda/customer_age_distribution.png") is True
        logging.info(
            'Saving image %s: SUCCESS',
            'customer_age_distribution.png')
    except AssertionError as e:
        logging.error(
            'Saving image %s: ERROR',
            'customer_age_distribution.png')
        raise e

# Check existing image marital_status_distribution.png
    try:
        assert os.path.isfile(
            "./images/eda/marital_status_distribution.png") is True
        logging.info(
            'Saving image %s: SUCCESS',
            'marital_status_distribution.png')
    except AssertionError as e:
        logging.error(
            'Saving image %s: ERROR',
            'marital_status_distribution.png')
        raise e

# Check existing image total_trans_distribution.png
    try:
        assert os.path.isfile(
            "./images/eda/total_trans_distribution.png") is True
        logging.info(
            'Saving image %s: SUCCESS',
            'total_trans_distribution.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'total_trans_distribution.png')
        raise e

# Check existing image heatmap.png
    try:
        assert os.path.isfile("./images/eda/heatmap.png") is True
        logging.info('Saving image %s: SUCCESS', 'heatmap.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'heatmap.png')
        raise e


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    # Load dataframe
    df = cls.import_data("./data/bank_data.csv")

    # Create Churn feature
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    cols_count = len(df.columns)
    try:
        df = encoder_helper(df=df,
                            category_lst=cls.cat_cols,
                            response='Churn')

        # Check len new columns
        assert len(df.columns) == len(cls.cat_cols) + cols_count
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as e:
        logging.error("Testing encoder_helper: ERROR")
        raise e


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    Test perform_feature_engineering
    '''
    # Load dataframe
    df = cls.import_data("./data/bank_data.csv")

    # Create Churn feature
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            df=df, response='Churn')

        # Check shape of X_train
        assert X_train.shape[1] == len(cls.keep_cols)
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as e:
        logging.error("Testing encoder_helper: ERROR")
        raise e


def test_train_models(train_models):
    '''
    Test train_models
    '''
    # Load dataframe
    df = cls.import_data("./data/bank_data.csv")

    # Create Churn feature
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # Perform feature engineering
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        df=df, response='Churn')

    # Check existing rf_train_report.png
    try:
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile('./images/results/rf_train_report.png') is True
        logging.info('Saving image %s: SUCCESS', 'rf_train_report.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'rf_train_report.png')
        raise e

    # Check existing rf_test_report.png
    try:
        assert os.path.isfile('./images/results/rf_test_report.png') is True
        logging.info('Saving image %s: SUCCESS', 'rf_test_report.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'rf_test_report.png')
        raise e

    # Check existing lr_train_report.png
    try:
        assert os.path.isfile('./images/results/lr_train_report.png') is True
        logging.info('Saving image %s: SUCCESS', 'lr_train_report.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'lr_train_report.png')
        raise e

    # Check existing lr_test_report.png
    try:
        assert os.path.isfile('./images/results/lr_test_report.png') is True
        logging.info('Saving image %s: SUCCESS', 'lr_test_report.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'lr_test_report.png')
        raise e

    # Check existing roc_curve_result.png
    try:
        assert os.path.isfile('./images/results/roc_curve_result.png') is True
        logging.info('Saving image %s: SUCCESS', 'roc_curve_result.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'roc_curve_result.png')
        raise e

    # Check existing feature_importances.png
    try:
        assert os.path.isfile(
            './images/results/feature_importances.png') is True
        logging.info('Saving image %s: SUCCESS', 'feature_importances.png')
    except AssertionError as e:
        logging.error('Saving image %s: ERROR', 'feature_importances.png')
        raise e

    # Check existing logistic_model.pkl
    try:
        assert os.path.isfile("./models/logistic_model.pkl") is True
        logging.info('Saving %s: SUCCESS', 'logistic_model.pkl')
    except AssertionError as e:
        logging.error('Saving %s: ERROR', 'logistic_model.pkl')
        raise e

    # Check existing rfc_model.pkl
    try:
        assert os.path.isfile("./models/rfc_model.pkl") is True
        logging.info('Saving %s: SUCCESS', 'rfc_model.pkl')
    except AssertionError as e:
        logging.error('Saving %s: ERROR', 'rfc_model.pkl')
        raise e


if __name__ == "__main__":
    test_import(cls.import_data)
    test_eda(cls.perform_eda)
    test_encoder_helper(cls.encoder_helper)
    test_perform_feature_engineering(cls.perform_feature_engineering)
    test_train_models(cls.train_models)
