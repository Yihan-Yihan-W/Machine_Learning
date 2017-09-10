# Machine_Learning
Practices for Python package SciKit Learn

	supervised learning: algorithms that create a model of the world by looking at labeled examples
	linear, logistic regression, SVM, decision trees, random forest, bayesian methods, text analysis
	
	unsupervised learning: algorithms that create a model of the world using examples without labels
	k-means clusting
	clustering: unsupervised learning used to automatically find groups in datasets

	Bayesian analysis: based on Bayes Theorem, make inferences about the world by combining domain knowldege or assumptions and observed evidence

## Supervised Machine Learning 
	1 KNN_digit_image: compare accuracy of image classification by n neighbors on training & test dataset.
	
	2 supervised learning knn iris : plot scatter matrix of y and xs , fit knn and calculate score
	
	3 SVM_party : deal with missing data, use imputer, pipeline , fit svm to classify party, print classification report
	
	4 Regression_gapminder: heatmap of correlation among variables; 
	fit linear regression & cross validation, computer Rsquare and MSE; 
	Fit lasso, ridge ElasticNet, tunning parameter with GridSearchCV
	Explore categorical variable, create dummy variable, CV ridge;
	Pipeline(imputer,scaler, elasticnet) + GridSearchCV
	
	5 KNN diabetes: fit KNN, confusion matrix, classification report
	fit logistic regression, plot ROC, compute AUC,
	Parameter tunning with GridSearchCV for logregression
	tunning with RandomizedSearchCV for decision trees
	
	6 KNN SVC wine quality: scale, fit KNN, compare accuracy; Pipeline, scale, SVC, GridSearch
