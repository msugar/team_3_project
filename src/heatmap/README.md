# Symptom Heatmap Visualization

## Overview

This project loads symptom occurrence data from a disease dataset and creates a heatmap to visualize the correlation between symptoms. The heatmap provides a graphical representation of how symptoms relate to one another based on their occurrence counts, helping to identify potential patterns and associations among symptoms.

## Files

- **Training.csv**: CSV file containing training data with symptom columns and a `prognosis` target column.
- **Testing.csv**: CSV file containing testing data with a similar structure as the training file.

## Setup

Make sure to have the necessary libraries installed for generating the heatmap.

## Requirements

Before running the script, ensure you have the following Python libraries installed:

```bash
pip install pandas matplotlib seaborn
