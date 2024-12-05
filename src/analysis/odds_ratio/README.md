# Odds Ratio Visualization

## Overview

This project analyzes the relationship between various symptoms and their corresponding prognosis using odds ratios. The odds ratio provides a measure of association between a symptom and the likelihood of a specific prognosis, enabling the identification of significant symptoms that contribute to different health outcomes. This analysis helps in understanding how individual symptoms correlate with prognosis classes.

It takes into account the following factors
- Symptom present & prognosis present
- Symptom present & prognosis absent
- Symptom absent & prognosis present
- Symptom absent & prognosis absent

## Files

- **Training.csv**: CSV file containing training data with symptom columns and a `prognosis` target column.
- **Testing.csv**: CSV file containing testing data with a similar structure as the training file.

## Setup

Make sure to have the necessary libraries installed for calculating and visualizing odds ratios.

## Requirements

Before running the script, ensure you have the following Python libraries installed:

```bash
pip install pandas matplotlib
```

## Reference

For more information on odds ratios, you can visit Wikipedia - Odds Ratio.
https://en.wikipedia.org/wiki/Odds_ratio
