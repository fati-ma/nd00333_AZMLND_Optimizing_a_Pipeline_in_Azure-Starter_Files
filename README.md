# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

- This dataset contains data about personal information for clients of a bank. We seek to predict *how likely the client will subscribe for a deposit*, and the problem is a *classifiction* problem.
- The model with best performance was **VotingEnsemble** with accuracy of **0.9167**.


## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**

The data was downloaded in the CSV format, and loaded from TabularDatasetFactory, then the data was cleaned using **clean_data** function where preprocessing steps were performed.
After that, the dataset was split into train and test datasets, and then built a **Logistic Regression** model.
Then the hyperparameters were tuned with **Hyperdrive** which are **Inverse of Regularization Strength: C** and the **Maximum Number of Iterations: max_iter**.
The **BanditPolicy** was used as the termintion policy based on *slack factor* and *evaluation interval*.
After that an **SKLearn** estimator was created, and then it was passed it along with the policy and hyperparameters to **HyperDriveConfig**.
At last, the best model was selected and saved.
The best accuracy was **0.9158**.


**What are the benefits of the parameter sampler you chose?**

**Random Parameter Sampling** was the one used. Hyperparameter values are randomly selected from the search space, where it chooses the values from a set of discrete values or a distribution over a continous range besides easy execution with minimal resources. For this problem, the hyperparameters that were given in search space are C (continuous) and max_iter(discrete).
The hypermarameters:
 - A uniform distribution of values between 0.1 and 1 for Inverse of Regularization Strength: C
 - The Maximum Number of Iterations: max_iter between a range of 100 and 200


**What are the benefits of the early stopping policy you chose?**

**BanditPolicy** was the one used. It terminates based on *slack factor* and *evaluation interval* which will terminate any run that doesn't fall within the specified *slack factor* .



## AutoML

**AutoMLConfig** class was used, and various parameters were passed to it (**automl_config**): experiment_timeout_minutes=30, task= 'classification', primary_metric= 'accuracy', training_data= train_dataset, label_column_name= 'y', and n_cross_validations= 4.

The great thing about AutoML is that it generates various models, in this experiment 42 models were generated and the models are: LightGBM, XGBoostClassifier, RandomForest, GradientBoosting, StackEnsemble and VotingEnsemble.

And the model with the best accuracy was **VotingEnsemble** with score of 0.9167. **VotingEnsemble** predicts based on the weighted average of predicted class probabilities.

Best metrics for best run:

![](https://github.com/fati-ma/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/automl-completed5.PNG?raw=true)

Best model output:

![](https://github.com/fati-ma/nd00333_AZMLND_Optimizing_a_Pipeline_in_Azure-Starter_Files/blob/master/best-ml-model.PNG?raw=true)



## Pipeline comparison
**Compare the two models and their performance. What are the differences in accuracy? In architecture? If there was a difference, why do you think there was one?**

In the first model which was **Logistic Regression** classifier, it had only one model unlike in **AutoML** it generted many models.
**AutoML** considering it had generated 31 models, I would say it was faster than the **HyperDrive**.
The best accuracy in *HyperDrive* was 0.9158 and in *AutoML* was 0.9167.

## Future work
**What are some areas of improvement for future experiments? Why might these improvements help the model?**

I would like to use different metrics to test the performance of the model to see how the result will differ.
I also would like to use deep learning frameworks as NN were not used in this project.
And I would try to work on data preprocessing more as the dataset is imbalanced and compare the results between the two models.

## Proof of cluster clean up

I did it at the last line in the jupyter notebook.



