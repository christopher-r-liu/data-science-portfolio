# data-science-portfolio
# Instagram Use and Stress Level
This project investigated the relationship between Instagram usage patterns and perceived stress using a dataset of 90,000 synthetic user profiles sourced from Kaggle. After cleaning and preparing the data, I conducted exploratory analysis and hypothesis testing (Pearson correlation, ANOVA, and t-tests) to establish that heavier Instagram usage is significantly associated with higher stress. Building on these findings, the team applied supervised machine learning (Logistic Regression, KNN, Random Forest, and Gradient Boosting) to classify users into low, medium, and high stress categories based on 17 behavioral features, achieving roughly 70% accuracy against a 34.5% baseline. The analysis also compared predictive patterns across age groups and used K-Means clustering with PCA visualization to identify four distinct behavioral user segments, ranging from casual to power users, whose stress and happiness levels formed a clear gradient tied to usage intensity. A key insight was that passive consumption behaviors (scrolling, viewing, watching) predicted stress more strongly than active engagement (posting, commenting).

#Classifying Real and Fake News
I built an end-to-end pipeline to classify news articles as real or fake, working with two labeled datasets of 5,000 articles each (10,000 total, real vs. fake). The workflow covered the full data science lifecycle: exploratory data analysis, data cleaning (handling missing values, stripping special characters, normalizing case), and feature engineering, before training and comparing multiple text classification approaches.

I implemented and benchmarked two distinct NLP pipelines on article text: a TF-IDF vectorizer paired with a Linear SVM, and a SentenceTransformer embedding model (384-dimensional sentence embeddings) paired with Logistic Regression. I also evaluated both using accuracy and full classification reports, achieving 98% test accuracy with TF-IDF + SVM versus 96% with sentence embeddings + Logistic Regression, and delivered a data-driven recommendation on which approach to deploy.

# Vegas Restaurant Exploration
I nalyzed the Yelp dataset for Las Vegas eateries to identify top-rated, budget-friendly dining options by predicting restaurant preference ratings and applying unsupervised clustering techniques.

Key Responsibilities & Highlights

Data Preprocessing & Cleaning: Loaded and cleaned a dataset of over 21,000 Yelp entries, handled missing values, and transformed business metrics into relevant features for analysis.
Supervised Machine Learning: Evaluated multiple classification models (Logistic Regression, K-Nearest Neighbors, Decision Trees, and Random Forests) using 5-fold Stratified Cross-Validation to predict high-preference restaurants.
Model Performance Evaluation: Achieved a peak model accuracy of 97.7% using a Decision Tree Classifier, outperforming other baseline models in overall precision and recall for target preferences.
Feature Engineering & Importance: Extracted top predictive features using Decision Tree feature importances, identifying review_count, business_id, reviewer_stars, and business_stars as the key drivers of restaurant ratings.
Unsupervised Clustering: Standardized high-dimensional feature sets, performed Principal Component Analysis (PCA) for dimensionality reduction, and applied K-Means clustering (using the Elbow Method with custom distortion calculations) to segment dining patterns.
