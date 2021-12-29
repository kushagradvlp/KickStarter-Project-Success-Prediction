# KickStarter-Project-Success-Prediction
Predicting success of a kickstarter project on its launch date

Kickstarter is a crowdfunding platform for gathering money from public. Project creators choose a deadline and minimum funding goal. If the backers back the project and the project reach the specified goal then the project is classified as successful otherwise its classified as failure. Goal of the model is to predict on the project launch date whether a project will be successful or not.


Data Cleaning
• Filtered the data for taking only the states that has successful and failed in them.
• Dropped state change days column as it had more than 70% missing values
• Dropped name_len and blurb_len column and used clean columns for them
• Number of backers, pledged amount and pledged amount in USD are removed as they will come in only after the launch of the project but we have to predict the    success when the project is launched.
• Removed spotlight as it is highly correlated with state.
• Dropped disable communication column has only one value so it won't make any impact on the model.
• Removed timestamp columns as they are already segregated into day, date and hour
• Dummified categorical columns like country, currency, staff_pick, category and removed original columns
• Removed the project_id and name columns as they are unique for most of the rows and wouldn't make any effect while modelling
• Removed anomalies using Isolationforest with contamination rate of 0.04


To select important features, I did the feature selection with random forest feature importance library. I took the features where importance score was greater than 0.01 and trained all the models on those features.

Tried the data on different classification and clustering algorithms. On the base model without any hyperparameters, the gradient boosted trees performed much better than others. Captured accuracy, precision, f1 score, specificity and auc score to compare different models. I used F1 score as the base measure for comparison as it takes both recall and precision into account.
