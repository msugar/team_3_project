# Feature Selection using eXtreme Gradient Boost (XGBoost)

Gradient Boosting is a good method to obtain feature importance scores from this dataset because it can handle binary features well and provide insights into which symptoms are most predictive.

## Setup

To install the XGBoost Python package using Miniconda, you can follow these steps:

1. Open your Miniconda prompt or terminal.
1. Ensure your Miniconda environment is activated. If you want to install XGBoost in a specific environment, activate it first:
    ```
    conda activate your_environment_name
    ```
1. Install XGBoost using the conda-forge channel by running the following command:
    ```
    conda install -c conda-forge py-xgboost
    ```