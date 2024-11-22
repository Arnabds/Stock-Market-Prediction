# Stock-Market-Prediction
A foundational step in predicting the stock market of a company by leveraging stock market prices of other companies and related commodities.

# Asset Monitoring and Predictive Maintenance
### Sponsored by FPoliSolutions, LLC

## Table of Contents

<li><a href="#Project-details">Project Details</a></li>
<li><a href="#Pre-analysis">Pre-analysis</a></li>
<ul>
    <li><a href="#Time-Series">Time Series</a></li>
    <li><a href="#Financial-Factors">Financial Factors</a></li>
<ul>
<li><a href="#Data">Data</a></li>
<li><a href="#Computer-experiments-to-study-patterns">Computer Experiments to Study Patterns</a></li>
<li><a href="#Project-instructions">Project Instructions</a></li>
<ul>
  <li>
    <a href="#Steps">Steps</a>
    <ul>
        <li><a href="#EDA-and-Preprocessing">EDA and Preprocessing</a></li>
        <li><a href="#Cluster-Analysis">Cluster Analysis</a></li>
        <li><a href="#Models">Models</a></li>
        <li><a href="#Prediction">Prediction</a></li>
        <li><a href="#Performance">Performance</a></li>
        <li><a href="#Extra">Extra</a>
            <ul>
              <li><a href="#SVC">SVC</a></li>
              <li><a href="#Neural-Net">Neural Net</a></li>
            </ul>
        </li>
    </ul>
  </li>
</ul>
<li><a href="#Summary">Summary</a></li>
<li><a href="#Things-to-answer-and-to-be-updated-next">Things to Answer and Update Next</a></li>
<li><a href="#References">References</a></li>

---

<h2 id="Project-details">Project Details</h2>
This project aims to predict stock market prices, an inherently volatile and complex task. A primary challenge was the scarcity of readily available data since financial data is often expensive and lacks predictor variables. Through extensive pre-analysis, we developed and refined our dataset.

<h3 id="Pre-analysis">Pre-analysis</h3>
We began by studying the time series of Amazon stock prices and gold commodity prices. Using data imported from Yahoo Finance, we performed time series analyses. The available data included `Volume`, `High`, `Low`, `Open`, and `Close` prices, but we focused solely on `Close` prices for analysis.

We fitted regression models to estimate trends and removed these trends to analyze residuals for stationarity. Residual analyses involved the Ljung-Box test and the ADF Fuller test, revealing non-stationarity. Differencing was applied to achieve stationarity, paving the way for ARIMA modeling. The best-fitting models were ARIMA(1,1,2) for gold and ARIMA(1,2,0) for Amazon. Although we did not explore GARCH models for volatility prediction at this stage, this is a next step. This preliminary analysis helped us understand the challenges of stock market prediction.

<h3 id="Financial-Factors">Financial Factors</h3>
Given the daily granularity of Yahoo Finance data, the predictors derived from the raw data were insufficiently meaningful due to multicollinearity. To address this, we engineered the following financial factors:
- **Gain**
  - Average Gain
- **Loss**
  - Average Loss
- **RSI (Relative Strength Index)**
- **Moving Averages**
  - Simple Moving Average
  - Exponential Moving Average
- **Rate of Change**
- **Price Volume Trend**

<h2 id="Data">Data</h2>
To enhance the predictive power, we created a custom dataset focusing on the automotive sector. After thorough market research, we identified related stocks linked to Ford and Toyota Motors. 

**Ford Suppliers:**
- Major suppliers: Autoliv (airbags), Warn Industries (axle assemblies), and Flex-N-Gate Seeburn (door hinges and arms).
- Indirect suppliers: FedEx, Union Pacific, and Roush.

**Toyota Suppliers:**
- Denso, Aisin Seiki Co., Microchip Technology, Johnson Controls, Takata Corporation, Dunlop Goodyear Tires, TRW Automotive Japan Co., TPR Co., T. RAD Co., Topre Corporation, Toyo Quality One Corporation, Topy Industries Limited.

We also included currency exchange rates in the dataset.

To refine our feature selection, we:
1. Drew a correlation matrix of closing prices.
2. Selected stocks with an absolute correlation > 0.4.
3. Applied Granger Causality to further narrow the list.

The final set of features included: `'CADUSD=X'`, `'GM'`, `'JCI'`, `'TM'`, `'TRYUSD=X'`, `'^IXIC'`, and `'F'`.

<h3 id="Data Cleaning and Preparation">Data Cleaning and Preparation</h3>
Financial factors introduced missing observations (e.g., from rolling calculations like moving averages). We removed these rows to ensure a clean dataset. This prepared dataset was used for regression, logistic regression, and advanced machine learning models, including random forests, gradient boosting, SVMs, and neural networks. The cleaned data was saved for further modeling.

---

<h2 id="Computer-experiments-to-study-patterns">Computer experiments to study patterns</h2>
> Correaltion plot
>
> Granger Causality




<h2 id="Project-instructions">Project instructions</h2>
- This project has 3 primary goals:
  - Train a model that accurately predicts the closing price value.
  - Train a model that accurately classify whether the stock closing price will go up or down the next day.
  - Invest fake money in investopedia to check our model's prediction performance.
- We must use an appropriate validation scheme to select the best model!


<h2 id="Steps">Steps</h2>
We have divided our project into 6 parts: ***EDA and Preprocessing, Cluster Analysis, Models, Performance, Prediction, and Testing***. We divided our work in groups. You will get these files with codes in jupyter notebook and HTML folders in Github. Introduction has been given so far. Let us start with the EDA.

|EDA and Preprocessing|Cluster Analysis|Models|Performance|Prediction|Bonus|
|--------|--------|--------|--------|--------|--------|
|Plotting necessary data, Standardization, Removing skewness, PCA|KMeans, Hierarchical clustering|7 logistic regression models and accuracy on training data|testing on manually created data|Gridsearch, lasso, ridge, elastic net|SVC, Neural net|
|Python|Python|Python|Python|Python|Python|



<h3 id="EDA-and-Preprocessing">EDA and Preprocessing</h3>

![Vibrational Data](/images/kde.png)
Hence we needed to use log transformation to remove skewness as we will use KMeans later on. We have also observed that the input features are correlated. E.g., successive V inputs are positively correlated which prepares a good stage for PCA.
![Vibrational Data](/images/corr.png)


<h3 id="Cluster-Analysis">Cluster Analysis</h3>
We have also observed that the input features are correlated. Hence when we applied PCA the correlation got removed. We chose the first 11 PCAs to be a useful one. Then we fitted KMeans and chose 2 clusters by knee bend plot. We have also used hierarchical clustering and went with 2 clusters.
![Vibrational Data](/images/hclust.png)

<h3 id="Models">Models</h3>
We have fitted 7 models from linear additive to interaction. Calculated its coefficients and showed statistical significance. We decided on the good models over the number of coefficients, threshold, Accuracy, Sensitivity, Specificity, FPR, and ROC_AUC. This was based on a test dataset. 
```
    formula_linear = 'Y ~ ' + ' + '.join(df_standardized_transformed.drop(columns= 'Y').columns)
    mod_03 = smf.ols(formula=formula_linear, data=df_standardized_transformed).fit()
    mod_03.params
```
```
    # Apply PCA to the transformed inputs and create all pairwise interactions between the PCs.
    df_pca_transformed_int = df_pca_transformed.iloc[:, :11].copy()
    df_pca_transformed_int['Y'] = df_transformed.Y
    formula_int = 'Y ~ ' + ' ( '  + ' + '.join(df_pca_transformed_int.drop(columns= 'Y').columns) + ' ) ** 2'
    mod_07 = smf.ols(formula=formula_int, data=df_pca_transformed_int).fit() 
    mod_07.params
```
We chose model 3 and model 7 from there which are all linear additive features from the original data set and model 7 is the interaction features with PCAs. They have 64 and 67 coefficients respectively.

<h3 id="Prediction">Prediction</h3>
Recall we had only training data with us. Hence, we chose model 3 and model 7 to check them on the test dataset that was created by us manually in the input grids X01, Z01, and Z04. Then we had prediction in model 3 we drew some line prediction plots for model 3 with x='X01', y='pred_probability_03', hue='Z01', and col='Z04'.
![Vibrational Data](/images/pred1.png)

Next, we had prediction in model 7 we drew some line prediction plots for model 7 with x='pc01', y='pred_probability_07', hue='pc04', col='pc11'.
![Vibrational Data](/images/pred2.png)

<h3 id="Performance">Performance</h3>
Next, we evaluated performance with Pipelines fitting logistic regression along with regularization lasso, ridge, and elastic net. We did not restrict ourselves strictly to lasso or ridge, rather went for an elastic net. From the l1 ratio, we observed that it is leaning towards lasso. Hence, we calculated performance with lasso and we got the highest score as 84%.

```
  # model 7-Apply PCA to the transformed inputs and create all pairwise interactions between the PCs
  pc_interact_lasso_search_grid.best_score_

0.8387878787878786
```

At the end, we forced to do elestic net a grid search with l1 ratio 0.5. Here also we got 84% accurate with 31 features coefficient zero. Hence we call it the best.

 ```
  enet_to_fit = LogisticRegression(penalty='elasticnet', solver='saga',
                            random_state=202, max_iter=25001, fit_intercept=True)
  pc_interact_enet_wflow = Pipeline( steps=[('std_inputs', StandardScaler() ), 
                                           ('pca', PCA() ), 
                                           ('make_pairs', make_pairs), 
                                           ('enet', enet_to_fit )] )
  enet_grid = {'pca__n_components': [3, 5, 7, 9, 11, 13, 15, 17],
             'enet__C': np.exp( np.linspace(-10, 10, num=17)),
             'enet__l1_ratio': np.linspace(0, 1, num=3)}
  pc_df_enet_search = GridSearchCV(pc_interact_enet_wflow, param_grid=enet_grid, cv=kf)
  pc_df__enet_search_results = pc_df_enet_search.fit( x_train_transformed, y_train_transformed )
  #The optimal value for C and no. of pca components is 
  pc_df__enet_search_results.best_params_
  pc_df__enet_search_results.best_score_
0.8387878787878786
```
```
0.8387878787878786
```
```
  coef = pc_df__enet_search_results.best_estimator_.named_steps['enet'].coef_
  empty_elements = coef[coef == 0]
  empty_elements.size
```
```
31
```

<h3 id="Extra">Extra</h3>
We have also fitted SVC and Neural net. In neural net we got 91% to 100% accuracy over cross validation and in SVC we get 100% accuracy all the time.

<h4 id="SVC">SVC</h4>

```
  svm_model = SVC()

  svm_param_grid = {
      'C': [0.1, 1, 10, 100],
      'kernel': ['linear', 'rbf', 'poly'],
      'gamma': ['scale', 'auto']
  }

  svm_result=svm_grid_search.fit(x_train_transformed, y_train_transformed)
  svm_result.best_params_

  svm_result.best_score_
  svm_cross_val_scores = cross_val_score(svm_grid_search.best_estimator_, x_train_transformed, y_train_transformed, cv=5, scoring='accuracy')
  print("SVM Cross-Validation Scores:", svm_cross_val_scores)
  print("SVM Mean Cross-Validation Score:", svm_cross_val_scores.mean())
```

```
SVM Cross-Validation Scores: [1. 1. 1. 1. 1.]
SVM Mean Cross-Validation Score: 1.0
```

<h4 id="Neural-Net">Neural Net</h4>

```
  # Appropriate model based on our task (regression/classification) is 
  # RandomForestClassifier for classification(RandomForestRegressor for regression )
  model = RandomForestClassifier()

  # Define the parameter grid for tuning
  param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
  }

  # Create the GridSearchCV object
  grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')  # Use appropriate scoring for your task

  # Fit the grid search to your data
  grid_search.fit(x_train_transformed, y_train_transformed)

  # Get the best parameters
  best_params = grid_search.best_params_
  print("Best Parameters:", best_params)

  # Assess performance using cross-validation
  cross_val_scores = cross_val_score(grid_search.best_estimator_, x_train_transformed, y_train_transformed, cv=5, scoring='accuracy')  # Use     
  appropriate scoring
  print("Cross-Validation Scores:", cross_val_scores)
  print("Mean Cross-Validation Score:", cross_val_scores.mean())
```

```
Cross-Validation Scores: [1.         1.         0.95555556 0.90909091 1.        ]
Mean Cross-Validation Score: 0.972929292929293
```

<h2 id="Summary">Summary</h2>

> In EDA we saw the inputs are highly correlated and that's why they are not very good at separating Y=0,1. The KMeans k2=0,1 worked well and it was not only giving us a better hue in the scatter plot but also matched well with Y=0,1.
>
> We can see that V07, V15, X10 are statistically significant features. It seems the 3rd approach to extracting patterns from the signals is more useful.
>
>Because they are correlated to each other we need the help of PCA to evaluate effective feature variables and at the end, we saw that there are 11 to 13 such PCA features that separate the data well and, hence, effective.
>
>The best logistic regression model turns out to be the elastic net with even mixed with ridge and lasso with 31 zero coefficients. We are getting 83-84% accuracy here. The best model in training turns out to be the best in prediction as well. In the end we saw if we use SVC then we are in fact getting 100% accuracy. I have also included Neural Net in the supporting document where we get 97% accuracy.


<h2 id="Things-to-answer-and-to-be-updated-next">Things to answer and to be updated next</h2>

This was my second project and more things are yet to be learned and improved. Data Science/machine learning is a journey like life
1. Removing skew with data-independent approach
2. How to choose optimal no PCA
3. More advanced methods after logistic regression.


<h2 id="References">References</h2>

1. University of Pittsburgh course CMPINF 2100
2. VSCode, Python

💻



  
 
