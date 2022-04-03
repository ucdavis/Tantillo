import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.linear_model import Lasso

#define the predictor variables.  changed commented line to evaluate another level of theory
features = ['gfn2_C1_f+', 'gfn2_C1_f-', 'gfn2_C1_f0', 'gfn2_C1_OWf+', 'gfn2_C1_OWf-', 'gfn2_C1_OWf0', 'gfn2_C1_Mullikenq', 'gfn2_C1_Hirshfeldq', 'gfn2_C1_FOD']

#load and define the data set
df=pd.read_csv('data.csv')
y=df['Barrier']
X=df[features]

#define the scaling for descriptors
scaler = MinMaxScaler()
X[features]=scaler.fit_transform(X[features])

#define the model and grid search
model = Lasso()
cv = RepeatedKFold(n_splits=6, n_repeats=10)
grid = dict()
grid['alpha'] = np.arange(0, 1, 0.01)
search = GridSearchCV(model, grid, scoring='neg_mean_absolute_error', cv=cv, refit=True, n_jobs=-1)
results = search.fit(X, y)
MAE=-1*float(results.best_score_)
print('Single CV MAE: %.3f' % MAE)

#fit the model to the whole data set
best_model=results.best_estimator_
print(best_model)
print(best_model.coef_)
print(best_model.intercept_)

#get the importance of each feature
importance=np.abs(best_model.coef_)

#select only those features with non-zero importance
features=np.array(features)[importance > 0].tolist()
print(features)
