import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/DatasetFinal.csv')
    x = dt_heart.drop(['%Toxicos'], axis=1)
    y = dt_heart['%Toxicos']
    
    # Imputar los valores faltantes en X
    imputer = SimpleImputer(strategy='mean')
    x_imputed = imputer.fit_transform(x)
    
    # Imputar los valores faltantes en y
    y_imputed = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(x_imputed, y_imputed, test_size=0.35, random_state=1)

    estimators = range(2, 300, 2)
    total_mse = []
    best_result = {'mse': float('inf'), 'n_estimator': 1}

    for i in estimators:
        boost = GradientBoostingRegressor(n_estimators=i).fit(X_train, y_train)
        boost_pred = boost.predict(X_test)
        new_mse = mean_squared_error(boost_pred, y_test)
        total_mse.append(new_mse)
        
        if new_mse < best_result['mse']: 
            best_result['mse'] = new_mse
            best_result['n_estimator'] = i
    
    print(best_result)

