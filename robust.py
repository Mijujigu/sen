import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, HuberRegressor
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    dataset = pd.read_csv('./data/DatasetFinal.csv')
    print(dataset.head(5))
    
    X = dataset.drop(['security_review_rating', '%Toxicos'], axis=1)
    y = dataset['%Toxicos']
    
    # Imputar los valores faltantes en X con la media
    imputer_X = SimpleImputer(strategy='mean')
    X = imputer_X.fit_transform(X)
    
    # Imputar los valores faltantes en y con la media
    imputer_y = SimpleImputer(strategy='mean')
    y = imputer_y.fit_transform(y.values.reshape(-1, 1)).ravel()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    estimators = {
        'SVR': SVR(gamma='auto', C=1.0, epsilon=0.1),
        'RANSAC': RANSACRegressor(),
        'HUBER': HuberRegressor(epsilon=1.35)
    }
    
    warnings.simplefilter("ignore")
    
    for name, estimator in estimators.items():
        # Entrenamiento
        estimator.fit(X_train, y_train)
        
        # Predicciones del conjunto de prueba
        predictions = estimator.predict(X_test)
        
        print("=" * 64)
        print(name)
        
        # Medimos el error entre los datos de prueba y las predicciones
        mse = mean_squared_error(y_test, predictions)
        print("MSE: %.10f" % mse)
        
        # Visualización de los resultados
        plt.figure()
        plt.scatter(y_test, predictions, alpha=0.5)
        plt.plot(predictions, predictions, 'r--')
        plt.xlabel('Real Score')
        plt.ylabel('Predicted Score')
        plt.title('Predicted vs Real')
        plt.show()
