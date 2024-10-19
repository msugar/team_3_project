# Symptom Clustering with K-Means

## Overview

This project loads symptom occurrence data from a disease dataset, computes the total occurrence of each symptom across all records, and applies K-means clustering to group similar symptoms based on their occurrence counts. The data is then visualized with a scatter plot, where each symptom is colored based on its cluster. The aim is to provide insights into symptom patterns and clustering tendencies based on their frequency.

## Files

- **Training.csv**: CSV file containing training data with symptom columns and a `prognosis` target column.
- **Testing.csv**: CSV file containing testing data with similar structure as the training file.

## Setup
Adjust num_kmeans to match the number of unique prognoses in your dataset for improved accuracy.

## Requirements

Before running the script, ensure you have the following Python libraries installed:

```bash
pip install pandas matplotlib scikit-learn
