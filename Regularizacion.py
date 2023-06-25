# Importamos las bibliotecas
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # Importamos el dataset del 2017
    dataset = pd.read_csv('./data/DatasetFinal.csv')

    # Seleccionamos los features que vamos a utilizar
    X = dataset[['0-10', '20-oct', '20-30', '30-40', '40-50', '50-60', '60-70', '70-80', '80-90', '90-100',
                 'Toxics commits avarage', 'Number of authors with toxic commits',
                 'Number of authors with toxic commits > 50%', 'Total commits', 'Total commits per day',
                 'Accumulated commits', 'Committers', 'suma', 'Committers Weight', 'classes', 'ncloc', 'functions',
                 'duplicated_lines', 'test_errors', 'skipped_tests', 'coverage', 'complexity', 'comment_lines',
                 'comment_lines_density', 'duplicated_lines_density', 'files', 'directories', 'file_complexity',
                 'violations', 'duplicated_blocks', 'duplicated_files', 'lines', 'public_api', 'statements',
                 'blocker_violations', 'critical_violations', 'major_violations', 'minor_violations', 'info_violations',
                 'lines_to_cover', 'line_coverage', 'conditions_to_cover', 'branch_coverage', 'sqale_index',
                 'sqale_rating', 'false_positive_issues', 'open_issues', 'reopened_issues', 'confirmed_issues',
                 'sqale_debt_ratio', 'new_sqale_debt_ratio', 'code_smells', 'new_code_smells', 'bugs',
                 'effort_to_reach_maintainability_rating_a', 'reliability_remediation_effort', 'reliability_rating',
                 'security_remediation_effort', 'security_rating', 'cognitive_complexity', 'new_development_cost',
                 'security_hotspots', 'security_review_rating']]

    # Definimos nuestro objetivo
    y = dataset['%Toxicos']

    # Imputamos los valores faltantes utilizando la media
    imputer = SimpleImputer(strategy='mean')
    X = imputer.fit_transform(X)
    y = imputer.fit_transform(y.values.reshape(-1, 1)).ravel()  # Imputar valores faltantes en y

    # Dividimos los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # Entrenamos los modelos
    modelLinear = LinearRegression().fit(X_train, y_train)
    modelLasso = Lasso().fit(X_train, y_train)
    modelRidge = Ridge().fit(X_train, y_train)
    modelElasticNet = ElasticNet().fit(X_train, y_train)

    # Realizamos las predicciones
    y_predict_linear = modelLinear.predict(X_test)
    y_predict_lasso = modelLasso.predict(X_test)
    y_predict_ridge = modelRidge.predict(X_test)
    y_pred_elastic = modelElasticNet.predict(X_test)

    # Calculamos las pérdidas (errores cuadráticos medios)
    linear_loss = mean_squared_error(y_test, y_predict_linear)
    lasso_loss = mean_squared_error(y_test, y_predict_lasso)
    ridge_loss = mean_squared_error(y_test, y_predict_ridge)
    elastic_loss = mean_squared_error(y_test, y_pred_elastic)

    # Imprimimos los resultados
    print("Linear Loss:", linear_loss)
    print("Lasso Loss:", lasso_loss)
    print("Ridge Loss:", ridge_loss)
    print("ElasticNet Loss:", elastic_loss)

    # Imprimimos los coeficientes de cada modelo
    print("Coeficientes lineal:")
    print(modelLinear.coef_)
    print("Coeficientes Lasso:")
    print(modelLasso.coef_)
    print("Coeficientes Ridge:")
    print(modelRidge.coef_)
    print("Coeficientes ElasticNet:")
    print(modelElasticNet.coef_)

    # Calculamos la precisión (score) de cada modelo
    print("Score Lineal:", modelLinear.score(X_test, y_test))
    print("Score Lasso:", modelLasso.score(X_test, y_test))
    print("Score Ridge:", modelRidge.score(X_test, y_test))
    print("Score ElasticNet:", modelElasticNet.score(X_test, y_test))

    # Creamos un DataFrame para visualizar los resultados
    results = pd.DataFrame({'Modelo': ['Linear', 'Lasso', 'Ridge', 'ElasticNet'],
                            'Loss': [linear_loss, lasso_loss, ridge_loss, elastic_loss],
                            'Coeficientes': [modelLinear.coef_, modelLasso.coef_, modelRidge.coef_,
                                             modelElasticNet.coef_],
                            'Score': [modelLinear.score(X_test, y_test), modelLasso.score(X_test, y_test),
                                      modelRidge.score(X_test, y_test), modelElasticNet.score(X_test, y_test)]})

    # Imprimimos los resultados en forma tabular
    print(results)


