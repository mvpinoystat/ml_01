# Machine Learning: Kaggle ICR Competition
This repo contains the script of the machine learning model submitted as a solution to the Kaggle Competition that you can find here: https://www.kaggle.com/competitions/icr-identify-age-related-conditions

The objective of the competition is to classify if the person has an age-related condition (1) or none (0). This is therefore a binary classification task.

The pre-preprocessing steps include a SimpleImputer, PowerTransformer, and RatioFeature (feature engineering). The final model is a Support Vector Machine classifier with optimized parameters using RandomizedSearchCV. The script is a part of the iteration process wherein different models are tested.  
