import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

if __name__ == "__main__":
    dt_heart = pd.read_csv('./data/DatasetFinal.csv')
    print(dt_heart.head(5))

    dt_features = dt_heart.drop(['%Toxicos'], axis=1)
    dt_target = dt_heart['%Toxicos']

    # Convertir variable objetivo en categórica binaria
    threshold = 0.5  # Umbral para clasificar como tóxico o no tóxico
    dt_target = dt_target.apply(lambda x: 1 if x > threshold else 0)

    # Datos originales
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.3, random_state=42)

    # Imputar valores faltantes en los datos originales
    imputer = SimpleImputer(strategy='mean')
    X_train_orig = imputer.fit_transform(X_train_orig)
    X_test_orig = imputer.transform(X_test_orig)

    kernel = ['linear', 'poly', 'rbf']
    for k in kernel:
        print("Kernel:", k)

        # Datos originales
        kpca_orig = KernelPCA(n_components=4, kernel=k)
        kpca_orig.fit(X_train_orig)
        dt_train_orig = kpca_orig.transform(X_train_orig)
        dt_test_orig = kpca_orig.transform(X_test_orig)

        logistic_orig = LogisticRegression(solver='lbfgs')
        logistic_orig.fit(dt_train_orig, y_train)
        y_pred_orig = logistic_orig.predict(dt_test_orig)
        acc_orig = accuracy_score(y_test, y_pred_orig)
        print("Accuracy (Datos originales):", acc_orig)

        # Normalización de datos
        scaler = StandardScaler()
        X_train_norm = scaler.fit_transform(X_train_orig)
        X_test_norm = scaler.transform(X_test_orig)

        # Aplicar KernelPCA a los datos normalizados
        kpca_norm = KernelPCA(n_components=4, kernel=k)
        kpca_norm.fit(X_train_norm)
        dt_train_norm = kpca_norm.transform(X_train_norm)
        dt_test_norm = kpca_norm.transform(X_test_norm)

        logistic_norm = LogisticRegression(solver='lbfgs')
        logistic_norm.fit(dt_train_norm, y_train)
        y_pred_norm = logistic_norm.predict(dt_test_norm)
        acc_norm = accuracy_score(y_test, y_pred_norm)
        print("Accuracy (Datos normalizados):", acc_norm)

        # Discretización de datos
        discretizer = KBinsDiscretizer(n_bins=3, encode='ordinal', strategy='uniform')
        X_train_discrete = discretizer.fit_transform(X_train_orig)
        X_test_discrete = discretizer.transform(X_test_orig)

        # Aplicar KernelPCA a los datos discretizados
        kpca_discrete = KernelPCA(n_components=4, kernel=k)
        kpca_discrete.fit(X_train_discrete)
        dt_train_discrete = kpca_discrete.transform(X_train_discrete)
        dt_test_discrete = kpca_discrete.transform(X_test_discrete)

        logistic_discrete = LogisticRegression(solver='lbfgs')
        logistic_discrete.fit(dt_train_discrete, y_train)
        y_pred_discrete = logistic_discrete.predict(dt_test_discrete)
        acc_discrete = accuracy_score(y_test, y_pred_discrete)
        print("Accuracy (Datos discretizados):", acc_discrete)

        print()

