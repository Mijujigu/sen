import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
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

    # Preprocesamiento: Escalado de características
    scaler = StandardScaler()
    dt_features_scaled = scaler.fit_transform(dt_features)

    # Preprocesamiento: Imputación de valores faltantes
    imputer = SimpleImputer(strategy='mean')
    dt_features_imputed = imputer.fit_transform(dt_features_scaled)

    # División de los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(dt_features_imputed, dt_target, test_size=0.2, random_state=42)

    # Aplicación de KPCA
    kpca = KernelPCA(n_components=2, kernel='rbf')
    X_train_kpca = kpca.fit_transform(X_train)

    # Eigenvalues
    eigvals = kpca.eigenvalues_

    # Total variance
    total_variance = np.sum(eigvals)

    # Variance ratio
    explained_variance_ratio = eigvals / total_variance

    print("Explained variance ratio (KPCA):", explained_variance_ratio)

    # Visualización de los datos transformados por KPCA
    plt.scatter(X_train_kpca[:, 0], X_train_kpca[:, 1], c=y_train)
    plt.xlabel("Componente Principal 1 (KPCA)")
    plt.ylabel("Componente Principal 2 (KPCA)")
    plt.title("KPCA - Datos Transformados")
    plt.show()



