# Financial-Sentiment-Analysis

This study aims to perform sentiment analysis on the financial textual data available at Kaggle (https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) and to construct classification models by the following machine learning algorithms - Logistic Regression, Support Vector Machine, Decision Tree, Random Forest and  Multinomial Naive Bayes. To apply those machine learning models, I converted the textual data into a numerical representation using Bag of Words and TF-IDF, and compared the performance. To deal with the imbalanced classifications, I used various over-sampling techniques along with the cost-sensitive learning.  

Support Vector Machine worked the best followed by Logistic Regression. RandomOverSampler was the best technique rather than SMOTE or combination of SMOTE and RandomUnderSampler to deal with my imbalance data.  TF-IDF generally worked better than Bag of Words as Bag of Words worked well only when randomly over-sampled.  

Overall, the models performed better when the text was not cleaned by removing the stop words and punctuation. It appears to work better not to remove the stop words to preserve the entire meaning of the context as when performing the sentiment analysis, it could cause a misclassification of the sentiment. 

Support Vector Machine with text as is (stopwords and punctuation not removed), TF-IDF and RandomOverSampler when  resulted in the best F1 score of 0.85.

RandomOverSampler significantly improved the model performance on the imbalanced data, but the predictions were still biased towards the majority class. 
