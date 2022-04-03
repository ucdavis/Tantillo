import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_validate

#define the scaling for descriptors
scaler = MinMaxScaler()

#define the predictor variables.  change the commented line to evaluate different descriptors/levels of theory
features = ['gfn2_C1_OWf+','gfn2_C1_FOD']
#features = ['gfn2//gfnff_C1_OWf+','gfn2//gfnff_C1_FOD']
#features = ['gfn1_C1_OWf+','gfn1_C1_FOD']
#features = ['gfn1//gfnff_C1_OWf+','gfn1//gfnff_C1_FOD']
#features = ['gfn2_C1_OWf+']
#features = ['gfn2//gfnff_C1_OWf+']
#features = ['gfn1_C1_OWf+']
#features = ['gfn1//gfnff_C1_OWf+']
#features = ['gfn2_C1_FOD']
#features = ['gfn2//gfnff_C1_FOD']
#features = ['gfn1_C1_FOD']
#features = ['gfn1//gfnff_C1_FOD']

#load and define the data set
df=pd.read_csv('data.csv')
df_train=df[df.Set=='Train']
df_test=df[df.Set=='Test']

#define the response variables
y_train=df_train['Barrier']
y_test=df_test['Barrier']

#define the predictor variables
X_train=df_train[features]
X_test=df_test[features]

#scale the predictor variables according to the fit on the training set
X_train[features] = scaler.fit_transform(X_train[features])
X_test[features] = scaler.transform(X_test[features])

#add a constant of 1 to every record for the regression in sm
X_train_lm = sm.add_constant(X_train, has_constant='add')
X_test_lm = sm.add_constant(X_test, has_constant='add')

#define and fit the model
lr_1 = sm.OLS(y_train, X_train_lm).fit()

#compute the training set MAE
y_pred=lr_1.predict(X_train_lm)
train_MAE=mean_absolute_error(y_train, y_pred)
del y_pred

#compute the test set MAE and R-squared
y_pred=lr_1.predict(X_test_lm)
test_MAE=mean_absolute_error(y_test, y_pred)
test_rsquared=r2_score(y_test, y_pred)

#compute all the predicted outcomes to record them
frames=[X_train_lm, X_test_lm]
saved_pred=lr_1.predict(sm.add_constant(pd.concat(frames)))

#print summary data
print("\n")
print(lr_1.summary())
print("\n")
print("Training Set R-squared: " +str(lr_1.rsquared_adj))
print("Training Set MAE: " +str(train_MAE))
print("Test Set R-squared: " +str(test_rsquared))
print("Test Set MAE: " +str(test_MAE))
print("\n")
print(saved_pred.sort_index())

#don't perform variance inflation factor analysis with only one descriptor
if (len(features) > 1):
    vif = pd.DataFrame()
    vif['Features'] = X_train.columns
    vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
    vif['VIF'] = round(vif['VIF'], 2)
    vif = vif.sort_values(by = "VIF", ascending = False)
    print(vif)
    print("\n")

#redefine the data set to test the model on validation set
X=df[features]
X[features]= scaler.fit_transform(X[features])
y=df['Barrier']

#define and fit the model
model=LinearRegression()
model.fit(X,y)

#define the validation set
X_val=df[df.Set == 'Validation'][features]
X_val[features]=scaler.transform(X_val)
y_val=df[df.Set == 'Validation']['Barrier']
y_pred=model.predict(X_val)
print(mean_absolute_error(y_pred,y_val))
print(r2_score(y_val.values,y_pred,))

print(y_val.values.reshape(-1))
print(np.around(y_pred.reshape(-1),decimals=1))
print(np.around(np.abs(y_val.values.reshape(-1)-y_pred.reshape(-1)),decimals=1))
