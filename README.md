# Stock-Market-Prediction
A foundational step in predicting the stock market of a company by leveraging stock market prices of other companies and related commodities.

# Asset Monitoring and Predictive Maintenance
### Sponsored by FPoliSolutions, LLC

## Table of Contents

<h2 id="Table-of-Contents">Table of Contents</h2>

<ul>
    <li><a href="#Project-details">Project Details</a></li>
    <li><a href="#Pre-analysis">Pre-analysis</a>
        <ul>
            <li><a href="#Time-Series">Time Series</a></li>
            <li><a href="#Financial-Factors">Financial Factors</a></li>
        </ul>
    </li>
    <li><a href="#Data">Data</a>
        <ul>
            <li><a href="#Data-Cleaning-and-Preparation">Data Cleaning and Preparation</a></li>
        </ul>
    </li>
    <li><a href="#Computer-experiments-to-study-patterns">Computer Experiments to Study Patterns</a></li>
    <li><a href="#Project-instructions">Project Instructions</a>
        <ul>
            <li><a href="#Steps">Steps</a>
                <ul>
                    <li><a href="#EDA-and-Preprocessing">EDA and Preprocessing</a></li>
                    <li><a href="#Models">Models</a></li>
                    <li><a href="#Prediction">Prediction</a></li>
                    <li><a href="#Performance">Performance</a></li>
                </ul>
            </li>
        </ul>
    </li>
    <li><a href="#Summary">Summary</a></li>
    <li><a href="#Things-to-answer-and-to-be-updated-next">Things to Answer and Update Next</a></li>
    <li><a href="#References">References</a></li>
</ul>

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



<h2 id="EDA-and-Preprocessing">EDA and Preprocessing</h2>

![Overall model with bollinger band](/images/Overall_model_with_bollinger_band.png)


This is the correlation plot with all the finally selected features.
![Correlation plot](/images/corr.png)


<h2 id="Models">Models</h2>
We have fitted 4 models from linear regression, linear regression with PCA, linear logistic regression, linear logistic regression with PCA all with regularization, XGBoost, XGBoost PCA.
```
    
    # Step 3: Split data - keep the last 1 week as an unseen test set
    split_index = len(X_transformed) - 7
    X_train_val, X_unseen_test = X_transformed[:split_index], X_transformed[split_index:]
    y_train_val, y_unseen_test = y[:split_index], y[split_index:]
    
    # Step 4: Set up parameter grids for hyperparameter tuning
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    lasso_params = {'alpha': [0.01, 0.1, 1.0, 10.0]}
    elastic_net_params = {'alpha': [0.01, 0.1, 1.0, 10.0], 'l1_ratio': [0.2, 0.5, 0.8]}
    
    # Initialize models
    ridge = Ridge()
    lasso = Lasso()
    elastic_net = ElasticNet()
    
    # TimeSeriesSplit for cross-validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # GridSearchCV with TimeSeriesSplit
    ridge_cv = GridSearchCV(ridge, ridge_params, scoring='neg_mean_squared_error', cv=tscv)
    lasso_cv = GridSearchCV(lasso, lasso_params, scoring='neg_mean_squared_error', cv=tscv)
    elastic_net_cv = GridSearchCV(elastic_net, elastic_net_params, scoring='neg_mean_squared_error', cv=tscv)
    
    # Fit models on training and validation set
    ridge_cv.fit(X_train_val, y_train_val)
    lasso_cv.fit(X_train_val, y_train_val)
    elastic_net_cv.fit(X_train_val, y_train_val)

    # Step 5: Collect MSE and best parameters for each model in a table
    results = pd.DataFrame({
        'Model': ['Ridge', 'Lasso', 'ElasticNet'],
        'Best Parameters': [ridge_cv.best_params_, lasso_cv.best_params_, elastic_net_cv.best_params_],
        'Best MSE': [
            -ridge_cv.best_score_,  # Convert negative MSE back to positive
            -lasso_cv.best_score_,
            -elastic_net_cv.best_score_
        ]
    })
        Grid Search Results with MSE for Each Model:
            Model                   Best Parameters  Best MSE
    0       Ridge                   {'alpha': 10.0}  0.023315
    1       Lasso                   {'alpha': 0.01}  0.020247
    2  ElasticNet  {'alpha': 0.01, 'l1_ratio': 0.2}  0.018912
    Best model: ElasticNet(alpha=0.01, l1_ratio=0.2), MSE: 0.01891209280316799
```
```
Similarly for logistic regression and other regression models with PCA. 

```
    # For logistic regression
    Unseen Test Accuracy (3 days): 0.6666666666666666

    # for regression PCA
        Grid Search Results with MSE for Each Model:
            Model                  Best Parameters  Best MSE
    0       Ridge                  {'alpha': 10.0}  0.019840
    1       Lasso                   {'alpha': 0.1}  0.034708
    2  ElasticNet  {'alpha': 0.1, 'l1_ratio': 0.2}  0.030487
    Best model: Ridge(alpha=10.0), MSE: 0.019839773849628435

    # for logistic regression PCA

    Grid Search Results with Accuracy for Logistic Regression:
                    Model                                    Best Parameters  \
    0  LogisticRegression  {'C': 0.1, 'penalty': 'l1', 'solver': 'libline...   
    
       Best Accuracy  
    0       0.533333  
    Best model: LogisticRegression(C=0.1, penalty='l1', solver='liblinear'), Accuracy: 0.5333333333333333
    Unseen Test Accuracy: 0.6666666666666666
    
    Classification Report:
                  precision    recall  f1-score   support
    
               0       0.67      1.00      0.80         2
               1       0.00      0.00      0.00         1
    
        accuracy                           0.67         3
       macro avg       0.33      0.50      0.40         3
    weighted avg       0.44      0.67      0.53         3
```
For XGBoost we have
```
    # Step 3: Split data - keep the last 1 week as an unseen test set
split_index = len(X_pca) - 7
X_train_val, X_unseen_test = X_pca[:split_index], X_pca[split_index:]
y_train_val, y_unseen_test = y[:split_index], y[split_index:]

# Step 4: Set up parameter grid for XGBoost hyperparameter tuning
xgb_params = {
    'n_estimators': [200, 300, 400],
    'learning_rate': [0.01, 0.005],
    'max_depth': [3, 5, 7],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2],
    'min_child_weight': [1, 5, 10]
}

# Initialize the XGBoost Regressor (without early stopping here)
xgb_reg = XGBRegressor(objective='reg:squarederror', random_state=42)

# TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=5)

# GridSearchCV with TimeSeriesSplit
xgb_cv = GridSearchCV(xgb_reg, xgb_params, scoring='neg_mean_squared_error', cv=tscv)
xgb_cv.fit(X_train_val, y_train_val)

# Step 5: Collect MSE and best parameters for XGBoost in a table
results = pd.DataFrame({
    'Model': ['XGBoost with PCA'],
    'Best Parameters': [xgb_cv.best_params_],
    'Best MSE': [-xgb_cv.best_score_]  # Convert negative MSE back to positive
})

# For XGBoost prediction
Unseen Test MSE (XGBoost): 0.023049184263072675

# For XGBoost PCA prediction
Unseen Test MSE: 0.037972127935972814
```

<h2 id="Prediction">Prediction</h2>
In all the model we got good results. Recall that it is a stock market data which tends to be whimsical. So it will be difficult to get the accurate prediction but we can observe the classification with some nice accuracy. Next, we evaluated performance with Pipelines fitting logistic regression along with regularization lasso, ridge, and elastic net. We varied the norm ratio as well and recorded our prediction accuracy. We have also checked our model with backtesting and forward validation. Here is the picture of some of our predictions. You will see that we can make simple regression stronger enough to make good predictions.


![XGBoost regression](/images/XGboost_reg.png)

![XGBoost regression pca](/images/XGBoost_reg_pca.png)

![Regression](/images/Reg.png)

<h2 id="Performance">Performance-Backtesting</h2>

Here's the updated table with an **Average Accuracy** column: for continuous prediction I have calculated accuracy based on next 7 days forecasting and for binary classification I forecasted for 3 days.

---

| **Models**               | **Subparts**                                                   | **Accuracy**        | **Average Accuracy** |
|--------------------------|----------------------------------------------------------------|---------------------|-----------------------|
| Regression               | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 100%      | **100%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 42.9%     | **82.9%**                        |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 100%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 71.4%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 100%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 100%     |                       |
| Regression PCA           | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 100%      | **100%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 71.4%     | **82.8%**                      |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 85.7%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 71.4%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 100%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 85.7%     |                       |
| XGBoost                  | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 100%      | **100%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 42.9%     | **82.9%**                      |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 100%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 85.7%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 85.7%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 100%     |                       |
| XGBoost PCA              | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 100%      | **100%**            |
|                          | - Backtesting 1(Ridge(alpha=0.1))                              | Accuracy: 86.9%     | **86.6%**                      |
|                          | - Backtesting 2(Ridge(alpha=0.1))                              | Accuracy: 89.8%     |                       |
|                          | - Backtesting 3(Ridge(alpha=0.1))                             | Accuracy: 85.5%     |                       |
|                          | - Backtesting 4(Ridge(alpha=0.1))                             | Accuracy: 85.5%     |                       |
|                          | - Backtesting 5(Elastic Net(alpha=0.01, l1_ratio=0.2))        | Accuracy: 85.5%     |                       |
| Logistic Regression      | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 66.7%      | **66.7%**            |
| Logistic Regression PCA  | - Elastic Net(alpha=0.01, l1_ratio=0.2)                        | Accuracy: 100%      | **100%**            |
---


<h2 id="Summary">Summary</h2>

> We need to explore more on time series like GARCH.
>
> We can see some statistically significant features. It seems that financial factor and using other stock market prices to extracting patterns from the random walk is more useful.
>
>Because the financial factors are correlated to each other we need the help of PCA to evaluate effective feature variables and at the end, we saw that there are 11 to 13 such PCA features that separate the data well and, hence, effective.
>
>The best regression and logistic regression model turns out to be the elastic net with even mixed with ridge and lasso with 31 zero coefficients. We are getting 80-100% accuracy here. The best model in training turns out to be the best in prediction as well. In the end we saw if we use "     " then we are in fact getting the best accuracy. I have used XGBoost, Neural Net, LSTM here.


<h2 id="Things-to-answer-and-to-be-updated-next">Things to answer and to be updated next</h2>

This was my second project and more things are yet to be learned and improved. Data Science/machine learning is a journey like life
1. Algorithmic Trading
2. Deep learning
3. finding out more financial features
4. Congressional trading


<h2 id="References">References</h2>

1. Erdos
2. VSCode, Python

💻



