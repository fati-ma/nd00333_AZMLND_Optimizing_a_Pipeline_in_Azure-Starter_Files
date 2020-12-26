# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

This dataset contains data about personal information for clients of a bank. We seek to predict how likely the client will subscribe for a deposit, and the problem is a classifiction problem.
The model with best performance was VotingEnsemble with accuracy of 0.9168.


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The data was downloaded in the CSV format, and loaded from TabularDatasetFactory, then the data was cleaned using **clean_data** function where preprocessing steps were performed.
After that, the dataset was split into train and test datasets, and then built a Logistic Regression model.
Then the hyperparameters were tuned with **Hyperdrive** which are **Inverse of Regularization Strength: C** and the **Maximum Number of Iterations: max_iter**.
The **BanditPolicy** was used as the termintion policy based on *slack factor* and *evaluation interval*.
After that an **SKLearn** estimator was created, and then it was passed it along with the policy and hyperparameters to **HyperDriveConfig**.
At last, the best model was selected and saved.


**What are the benefits of the parameter sampler you chose?**

**Random Parameter Sampling** was the one used. It searches through the search space (but not the entire seacrh space) and returns random values on that space.
In this problem the seacrh space consisted of the hyperparameters C and max_iter.


**What are the benefits of the early stopping policy you chose?**

**BanditPolicy** was the one used. It terminates based on *slack factor* and *evaluation interval* which will terminate any run that doesn't fall within the specified *slack factor* .

## AutoML

or the automl model, **AutoMLConfig** class was used, and various parameters were passed to it (**automl_config**): experiment_timeout_minutes=30, task= 'classification', primary_metric= 'accuracy', training_data= x, label_column_name= 'y', and n_cross_validations= 4.
The great thing about AutoML is that it generates various models, in this experiment 31 models were generated and the models are: LightGBM, XGBoostClassifier, RandomForest, GradientBoosting, LogisticRegression, ExtremeRandomTrees, StackEnsemble and VotingEnsemble.
And the model with the best accuracy was VotingEnsemble with score of 0.9168.

## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

In the first model which was **Logistic Regression** classifier, I had only one model unlike **AutoML** model it generted many models.
**AutoML** considering it had generated 31 models, I would say it was faster than the **HyperDrive**.
The best accuracy in *HyperDrive* was 0.9130857004127215 and in *AutoML* was 0.9168.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

I would like to use different metrics to test the performance of the model to see how the result will differ.
I also would like to use deep learning frameworks as NN were not used in this project.
And I would try to work on data preprocessing more as the dataset is imbalanced and compare the results between the two models.

## Proof of cluster clean up

I did it at the last line in the jupyter notebook.



