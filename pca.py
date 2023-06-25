import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

if __name__ == "__main__":
    # Cargar los datos del dataframe de pandas
    dt_heart = pd.read_csv('./data/DatasetFinal.csv')
    
    # Imprimir un encabezado con los primeros 5 registros
    print(dt_heart.head(5))
    
    # Guardar el dataset sin la columna de target
    dt_features = dt_heart.drop(['%Toxicos'], axis=1)
    
    # Este será nuestro dataset, pero sin la columna
    dt_target = dt_heart['%Toxicos']
    
    # Imputar los valores faltantes en los datos
    imputer = SimpleImputer(strategy='mean')
    dt_features_imputed = imputer.fit_transform(dt_features)
    
    # Codificar las etiquetas de destino
    label_encoder = LabelEncoder()
    dt_target_encoded = label_encoder.fit_transform(dt_target)
    
    # Normalizar los datos
    scaler = StandardScaler()
    dt_features_scaled = scaler.fit_transform(dt_features_imputed)
    
    # Partir el conjunto de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dt_features_scaled, dt_target_encoded, test_size=0.3, random_state=42)
    
    # Aplicar PCA
    pca = PCA(n_components=3)
    dt_train_pca = pca.fit_transform(X_train)
    dt_test_pca = pca.transform(X_test)
    
    # Aplicar la regresión logística a los datos de PCA
    logistic_pca = LogisticRegression(solver='lbfgs', max_iter=1000)
    logistic_pca.fit(dt_train_pca, y_train)
    score_pca = logistic_pca.score(dt_test_pca, y_test)
    
    # Imprimir el resultado de PCA
    print("SCORE PCA: ", score_pca)

