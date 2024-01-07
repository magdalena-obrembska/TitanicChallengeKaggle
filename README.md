## Titanic Challenge Kaggle
 
This is my first challenge and first begginer data science project. I am very grateful for the opportunity to practice on the Kaggle platform and for the lot of knowledge from many users and creators of other solutions. I used many notebooks, but also many internet sources and books to create project and understand basic topics. I am a beginner in Python and data science, so I am open to any comments to improve my solution. I would also like to thank other users for sharing their solutions, notebooks and ideas on the discussion threads, which also helped me a lot in understanding many issues. My solution summary:

EDA analysis - 1st Problem - I had problem with missing data in Age column I must repeat fill missing data. 2nd Problem - I detect outliers but I can't delete them, because at the end length of my data doesn't fit to submission. I need to read up on this problem. I think that dealing with outliers could be helpful with better model accuracy and reduce large kurtosis in few columns.

Feature engineering - I create some new features and assess by correlation which will be good for the next steps.

Standard Scaler to prepare data to modeling.

Modeling and hyperparameters tuning (I know I have a messy code but I tried a lot of models. I tried to asses the best parameters by create a function or I'm trying manually change values of hyperparameters.)

My summary of models with accuracy based on train data score (Decision Tree, Random Forest, SVC, Logistic Regression, KNN, Extra Trees Classifier, Ada Boost Classifier and Gradient Boost Classifier.) I created table with selected scores(I rejected the worse ones).

I choose two models Random Forest Classifier model and Extra Trees Classifier model and I asses them with Learning Curve. I also asses in Validation Curve impact of max depth in both models and I choose Random Forest Classifier (which I improved changing max depth value to 3, because there was some problems : I had higher train accuracy than test accuracy (the difference between these two measures was to high)).

I tried the improve model on data and make a prediction.

Inside the repo you can find solution in pure Python, and in Jupyter notebook file.

Link to my notebook in Kaggle, if you want to see how it works:
[https://www.kaggle.com/code/magdalenaobrembska/my-first-challenge-titanic?kernelSessionId=155931192](https://www.kaggle.com/code/magdalenaobrembska/my-first-challenge-titanic?kernelSessionId=155931192)