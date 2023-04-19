import pandas as pd
import re
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
import plotly.offline as py

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import  make_scorer,r2_score

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import  DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

import pickle

#load dataset
datafile = pd.read_csv("C:/Users/Nandini/project2(restaurant rating)/data/zomato.csv")
df=datafile
# Datafile Exploration
df.head(3)
df.info()
df.describe()
#eda
df.columns
# Dropping of unnecessary columns
df = df.drop(['url', 'address', 'phone','reviews_list' ], axis = 1)
# Renaming columns
df = df.rename(columns={'approx_cost(for two people)':'estimatedcost','listed_in(type)':'type',
                                  'listed_in(city)':'city'})
features_with_na=[features for features in df.columns if df[features].isnull().sum()>=1]
df['rate'].unique()
df = df.query("rate != 'NEW'")
df['rate'].unique()
df['rate'].isna().sum()
df1 = df.copy()
df1['rate'] = df1['rate'].apply(lambda x: x.replace('/5', '') if isinstance(x, str) else x)
df1['rate']=df1['rate'].str.replace(" ","")
counts = df1['rate'].value_counts()
print(counts)
import seaborn as sns
import matplotlib.pyplot as plt

sns.histplot(df1['rate'], kde=True)
plt.title("Distribution of 'rate' column")
plt.show()
#this is to understand if i want to use mean/median/mode to replace the missing values. if the distribution is skewed we will use median, if its normal we can use mean and for categorical variables we will use mode.
# Replace '-' with NaN in 'rate' column
df1['rate'] = df1['rate'].replace('-', np.nan)

# Convert 'rate' column to numeric type
df1['rate'] = pd.to_numeric(df1['rate'], errors='coerce')
#The above code will replace all '-' characters in the rate column with NaN values using the replace method, and then convert the column to a numeric type using the to_numeric method with the errors='coerce' parameter. This will automatically convert all non-numeric values (including the NaN values) to NaN, which can then be handled appropriately using pandas methods like fillna, dropna, etc
# Replace missing values in 'rate' column with the median value
median_rate = df1['rate'].median()
df1['rate'].fillna(median_rate, inplace=True)
features_with_na=[features for features in df.columns if df1[features].isnull().sum()>=1]
(df1.isna().sum()/df.shape[0])*100
#modifying estimatedcost column
df1['estimatedcost'].unique()
df1['estimatedcost']=df1['estimatedcost'].str.replace(",","")
# Replace missing values in 'estimated' column with the median value
cost = df1['estimatedcost'].median()
df1['estimatedcost'].fillna(cost, inplace=True)
del df1['location']
df1.head()
df1.dropna(subset=['dish_liked', 'cuisines', 'rest_type'], inplace=True)
#drops all the rows containing null values of each of these columns

#data visualisation
# Famous Resturants on the basis of occurance In Ascending order
plt.figure(figsize=(15,10))
chains=df1['name'].value_counts()[:30]
sns.barplot(x=chains,y=chains.index)
plt.title("Most famous restaurants chains in Bengaluru")
plt.xlabel("Number of outlets")
plt.show()

#top restaurants based on ratings
top_restaurants = df1.sort_values('rate', ascending=False).head(20)
print(top_restaurants[['name', 'rate']])

top_rated = df1.nlargest(20, 'rate') # select top 10 rated restaurants
plt.figure(figsize=(8,6))
sns.barplot(x='rate', y='name', data=top_rated, palette='rocket')
plt.xticks(rotation=45, ha='right') # rotate and align y-axis labels
plt.xlabel('Rating')
plt.ylabel('Restaurant Name', fontsize=12)
plt.title('Top Rated Restaurants in Bangalore')
plt.show()

#resturants which take online order
plt.figure(figsize=(6,4))
sns.countplot(df1['online_order'])
plt.title("Restaurants with online order",fontsize=18,color='black')
plt.ylabel("Count",fontsize=15)
plt.xlabel("Online order",fontsize=15)
plt.show()

#restaurants that have table booking option
plt.figure(figsize=(6,4))

sns.countplot(df['book_table'])
plt.title("Restaurants with table booking option",fontsize=18,color='black')
plt.ylabel("Count",fontsize=15)
plt.xlabel("Book Table",fontsize=15)
plt.show()

# Famous Resturant types
plt.figure(figsize=(15,7))
rest=df1['rest_type'].value_counts()[:20]
sns.barplot(rest,rest.index)
plt.title("Restaurant types")
plt.xlabel("count")

# top 10 most liked dishes
# Create a list of all the dishes
all_dishes = []
for dishes in df1['dish_liked']:
    all_dishes += dishes.split(", ")
# Create a frequency table for the dishes
dish_freq = pd.Series(all_dishes).value_counts()
# Print the top 10 most popular dishes
print("Top 10 most liked dishes:")
print(dish_freq.head(15))

# Create a list of all the dishes
all_dishes = []
for dishes in df1['dish_liked']:
    all_dishes += dishes.split(", ")
# Create a frequency table for the dishes
dish_freq = pd.Series(all_dishes).value_counts()
# Get the top 10 most popular dishes
top_10_dishes = dish_freq.head(15)
# Create a bar plot to show the top 10 most liked dishes
plt.figure(figsize=(10, 6))
plt.bar(x=top_10_dishes.index, height=top_10_dishes.values)
plt.xticks(rotation=90)
plt.xlabel('Dish')
plt.ylabel('Frequency')
plt.title('Top 10 most liked dishes in Bangalore restaurants')
plt.show()

#number of restaurants area wise
plt.figure(figsize=(16,16))
ax = df1.city.value_counts()
ax.plot(kind='pie',fontsize=20, autopct='%1.1f%%')
plt.title('Number of restaurants area wise',fontsize=20,color='black')
# Add labels for each slice of the pie chart
labels = df1.city.value_counts().index.tolist()
plt.legend(labels, title="City", loc="center right", bbox_to_anchor=(1, 0, 0.5, 1))
# Add text for total number of restaurants
total_restaurants = df1.shape[0]
plt.text(0, -1.1, f'Total Restaurants: {total_restaurants}', fontsize=16, ha='center')
plt.show()

df1['estimatedcost'] = df1['estimatedcost'].astype(int)

corr_matrix = df1[['estimatedcost', 'rate']].corr()
print(corr_matrix)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Create scatter plot with linear regression line
sns.lmplot(x='estimatedcost', y='rate', data=df1)

# Set plot labels and title
plt.xlabel('Cost of dishes')
plt.ylabel('Restaurant rating')
plt.title('Relationship between cost and rating of Bangalore restaurants')

# Show plot
plt.show()
#the linear regression line slopes upward, it suggests a positive correlation, which means that as the cost of dishes increases, the rating of the restaurant also tends to increase

# Create a pivot table that shows the average estimated cost for each combination of rest_type and city
pivot_table = pd.pivot_table(df1, values='estimatedcost', index='rest_type', columns='city', aggfunc='mean')

# Print the pivot table
print(pivot_table)

#average rating for each city in the dataset.
df_city_rate = df1[['city', 'rate']]
mean_ratings = df_city_rate.groupby('city')['rate'].mean()
plt.figure(figsize=(10,8))
mean_ratings.plot(kind='barh')
plt.xlabel('Rating')
plt.ylabel('City')
plt.title('Mean Ratings by City')
plt.show()

#model preparation
#changing the categorical variables into the numerical ones
df1.loc[df1.online_order == 'Yes', 'online_order'] = 1
df1.loc[df1.online_order == 'No', 'online_order'] = 0

df1.loc[df1.book_table == 'Yes', 'book_table'] = 1
df1.loc[df1.book_table == 'No', 'book_table'] = 0

le = LabelEncoder()
df1.city = le.fit_transform(df1.city)
df1.rest_type = le.fit_transform(df1.rest_type)
df1.cuisines = le.fit_transform(df1.cuisines)
df1.menu_item = le.fit_transform(df1.menu_item)

df1 = df1.drop(["name", "dish_liked", "type"], axis = 1)
# Dependent and independent variable division
y = df1['rate']
x =df1.drop(columns = ['rate'])
# Dataset Train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=101)

## 1. LINEAR REGRESSION
model_lr = LinearRegression()
model_lr.fit(x_train, y_train)

y_pred = model_lr.predict(x_test)
print(r2_score(y_test, y_pred))

print(f"The model prediction on train dataset: {round(model_lr.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_lr.score(x_test, y_test),2)}")

# create Ridge regression model with alpha=1
model_ridge = Ridge(alpha=1)

# train the model
model_ridge.fit(x_train, y_train)

# predict on test set
y_pred = model_ridge.predict(x_test)
print(r2_score(y_test, y_pred))
# calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: ', mse)

# create Ridge regression model with alpha values to test
model = RidgeCV(alphas=[0.1, 1.0, 10.0])

# perform 5-fold cross-validation to find best alpha
scores = cross_val_score(model, x, y, cv=5)

# fit the model with best alpha
model.fit(x, y)

# print the best alpha value and corresponding score
print("Best alpha: ", model.alpha_)
print("Cross-validation scores: ", scores)
print("Mean score: ", np.mean(scores))

#2.lasso regression
# create Lasso regression model with alpha=1
model = Lasso(alpha=1)

# train the model
model.fit(x_train, y_train)

# predict on test set
y_pred = model.predict(x_test)

# calculate mean squared error
mse = mean_squared_error(y_test, y_pred)
print('Mean squared error: ', mse)
print(r2_score(y_test, y_pred))

## 3.SUPPORT VECTOR MACHINE
model_svr = SVR()
model_svr.fit(x_train , y_train)
y_pred = model_svr.predict(x_test)
print(r2_score(y_test, y_pred))
print(f"The model prediction on train dataset: {round(model_svr.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_svr.score(x_test, y_test),2)}")

# 4.DECISION TREE
model_dt = DecisionTreeRegressor()
model_dt.fit(x_train, y_train)
y_pred = model_dt.predict(x_test)
print(r2_score(y_test, y_pred))
print(f"The model prediction on train dataset: {round(model_dt.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_dt.score(x_test, y_test),2)}")

## 5.RANDOM FOREST REGRESSOR
model_rf = RandomForestRegressor()
model_rf.fit(x_train, y_train)
y_pred = model_rf.predict(x_test)
print(r2_score(y_test, y_pred))
print(f"The model prediction on train dataset: {round(model_rf.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_rf.score(x_test, y_test),2)}")

## 5. RANDOM FOREST REGRESSOR
model_rf = RandomForestRegressor(n_estimators=100)
scores = cross_val_score(model_rf, x, y, cv=5)
model_rf.fit(x_train, y_train)
y_pred = model_rf.predict(x_test)
print("Cross-validation scores: ", scores)
print("Mean score: ", np.mean(scores))
print(f"The model prediction on train dataset: {round(model_rf.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_rf.score(x_test, y_test),2)}")

## 6.ADAPTIVE BOOSTING
model_ada = AdaBoostRegressor(base_estimator = model_dt)
model_ada.fit(x_train, y_train)
y_pred = model_ada.predict(x_test)
print(r2_score(y_test, y_pred))
print(f"The model prediction on train dataset: {round(model_ada.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_ada.score(x_test, y_test),2)}")

## 7.GRADIENT BOOSTING
model_gb = GradientBoostingRegressor()
model_gb.fit(x_train, y_train)
y_pred = model_gb.predict(x_test)
print(r2_score(y_test, y_pred))
print(f"The model prediction on train dataset: {round(model_gb.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_gb.score(x_test, y_test),2)}")

# 8.EXTRA TREE REGRESSOR
model_et = ExtraTreesRegressor()
model_et.fit(x_train, y_train)
y_pred = model_et.predict(x_test)
print(r2_score(y_test, y_pred))
print(f"The model prediction on train dataset: {round(model_et.score(x_train, y_train),2)}")
print(f"The model prediction on test dataset: {round(model_et.score(x_test, y_test),2)}")

df1['online_order'] = df1['online_order'].astype(int)
df1['book_table'] = df1['book_table'].astype(int)

#combining multiple models to create an ensemble model that can provide better performance than any individual model.
# Train AdaBoost model
ada_model = AdaBoostRegressor(n_estimators=100)
ada_model.fit(x_train, y_train)
# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(x_train, y_train)
# Combine predictions using simple averaging
y_pred_ensemble = (ada_model.predict(x_test) + rf_model.predict(x_test)) / 2
# Evaluate performance of ensemble model
ensemble_score = r2_score(y_test, y_pred_ensemble)
print("Ensemble score:", ensemble_score)

#since adaboost is the model giving highest accuracy so we can do some hyperparameter tuning
# Create an Adaboost model object
model_ada = AdaBoostRegressor()
# Define the hyperparameters to tune
parameters = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.01, 0.1, 1],
              'loss': ['linear', 'square', 'exponential']}
# Define the scoring metric
scoring = make_scorer(r2_score)
# Create a GridSearchCV object
grid_search = GridSearchCV(model_ada, param_grid=parameters, scoring=scoring, cv=5)
# Fit the GridSearchCV object to the training data
grid_search.fit(x_train, y_train)
# Print the best hyperparameters and score
print("Best parameters: ", grid_search.best_params_)
print("Best score: ", grid_search.best_score_)

# create a new AdaBoost model with the best hyperparameters
model_ada_tuned = AdaBoostRegressor(base_estimator=model_dt, learning_rate=0.01, loss='exponential', n_estimators=200)
# fit the model on the training data
model_ada_tuned.fit(x_train, y_train)
# make predictions on the test data
y_pred = model_ada_tuned.predict(x_test)
# evaluate the model performance on the test data using cross-validation
scores = cross_val_score(model_ada_tuned, x_train, y_train, cv=5)
mean_score = scores.mean()
std_score = scores.std()
print(f"The mean score: {mean_score}")
print(f"The standard deviation: {std_score}")
print(f"The model prediction on train dataset: {mean_score}")
print(f"The model prediction on test dataset: {model_ada_tuned.score(x_test, y_test)}")

model_ada_tuned.predict([[1,0,15,73,6900,750,27,5]])
## BEST MODEL SAVE
pickle.dump(model_ada_tuned, open('model2.pkl','wb'))
model=pickle.load(open('model2.pkl','rb'))




