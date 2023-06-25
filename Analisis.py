import pandas as pd
from sklearn.decomposition import PCA, KernelPCA, IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
import warnings
warnings.filterwarnings("ignore")

# Cargar el dataset
dt_heart = pd.read_csv('./data/DatasetFinal.csv')

# Eliminar filas con valores no finitos en el dataset objetivo
dt_heart = dt_heart.dropna(subset=['%Toxicos'])

# Guardar el dataset sin la columna de target
dt_features = dt_heart.drop(['%Toxicos'], axis=1)

# Este será nuestro dataset, pero sin la columna
dt_target = dt_heart['%Toxicos']

# Convertir la columna objetivo a enteros
dt_target = dt_target.astype(int)

# Dividir el conjunto de datos en entrenamiento y prueba
X_train_orig, X_test_orig, y_train, y_test = train_test_split(dt_features, dt_target, test_size=0.2, random_state=42)

# Escalar los datos de entrenamiento y prueba
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_orig)
X_test_scaled = scaler.transform(X_test_orig)

# Aplicar PCA (originales)
pca_orig = PCA(n_components=10)
dt_train_pca_orig = pca_orig.fit_transform(X_train_orig)
dt_test_pca_orig = pca_orig.transform(X_test_orig)

# Aplicar PCA (normalizados)
pca_scaled = PCA(n_components=10)
dt_train_pca_scaled = pca_scaled.fit_transform(X_train_scaled)
dt_test_pca_scaled = pca_scaled.transform(X_test_scaled)

# Aplicar PCA (discretizados)
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_train_disc = discretizer.fit_transform(X_train_orig)
X_test_disc = discretizer.transform(X_test_orig)

pca_disc = PCA(n_components=10)
dt_train_pca_disc = pca_disc.fit_transform(X_train_disc)
dt_test_pca_disc = pca_disc.transform(X_test_disc)

# Aplicar Kernel PCA (originales)
kernel_pca_orig = KernelPCA(n_components=10, kernel='rbf')
dt_train_kernel_pca_orig = kernel_pca_orig.fit_transform(X_train_orig)
dt_test_kernel_pca_orig = kernel_pca_orig.transform(X_test_orig)

# Aplicar Kernel PCA (normalizados)
kernel_pca_scaled = KernelPCA(n_components=10, kernel='rbf')
dt_train_kernel_pca_scaled = kernel_pca_scaled.fit_transform(X_train_scaled)
dt_test_kernel_pca_scaled = kernel_pca_scaled.transform(X_test_scaled)

# Aplicar Kernel PCA (discretizados)
kernel_pca_disc = KernelPCA(n_components=10, kernel='rbf')
dt_train_kernel_pca_disc = kernel_pca_disc.fit_transform(X_train_disc)
dt_test_kernel_pca_disc = kernel_pca_disc.transform(X_test_disc)

# Aplicar Incremental PCA (originales)
incremental_pca_orig = IncrementalPCA(n_components=10)
dt_train_incremental_pca_orig = incremental_pca_orig.fit_transform(X_train_orig)
dt_test_incremental_pca_orig = incremental_pca_orig.transform(X_test_orig)

# Aplicar Incremental PCA (normalizados)
incremental_pca_scaled = IncrementalPCA(n_components=10)
dt_train_incremental_pca_scaled = incremental_pca_scaled.fit_transform(X_train_scaled)
dt_test_incremental_pca_scaled = incremental_pca_scaled.transform(X_test_scaled)

# Aplicar Incremental PCA (discretizados)
incremental_pca_disc = IncrementalPCA(n_components=10)
dt_train_incremental_pca_disc = incremental_pca_disc.fit_transform(X_train_disc)
dt_test_incremental_pca_disc = incremental_pca_disc.transform(X_test_disc)

# Aplicar la regresión logística a los datos originales
logistic_orig = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_orig.fit(X_train_orig, y_train)
score_orig = logistic_orig.score(X_test_orig, y_test)

# Aplicar la regresión logística a los datos normalizados
logistic_scaled = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_scaled.fit(X_train_scaled, y_train)
score_scaled = logistic_scaled.score(X_test_scaled, y_test)

# Aplicar la regresión logística a los datos de PCA (originales)
logistic_pca_orig = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_pca_orig.fit(dt_train_pca_disc, y_train)
score_pca_orig = logistic_pca_orig.score(dt_test_pca_disc, y_test)

# Aplicar la regresión logística a los datos de PCA (normalizados)
logistic_pca_scaled = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_pca_scaled.fit(dt_train_pca_scaled, y_train)
score_pca_scaled = logistic_pca_scaled.score(dt_test_pca_scaled, y_test)

# Aplicar la regresión logística a los datos de PCA (discretizados)
logistic_pca_disc = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_pca_disc.fit(dt_train_pca_disc, y_train)
score_pca_disc = logistic_pca_disc.score(dt_test_pca_disc, y_test)

# Aplicar la regresión logística a los datos de Kernel PCA (originales)
logistic_kernel_pca_orig = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_kernel_pca_orig.fit(dt_train_kernel_pca_orig, y_train)
score_kernel_pca_orig = logistic_kernel_pca_orig.score(dt_test_kernel_pca_orig, y_test)

# Aplicar la regresión logística a los datos de Kernel PCA (normalizados)
logistic_kernel_pca_scaled = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_kernel_pca_scaled.fit(dt_train_kernel_pca_scaled, y_train)
score_kernel_pca_scaled = logistic_kernel_pca_scaled.score(dt_test_kernel_pca_scaled, y_test)

# Aplicar la regresión logística a los datos de Kernel PCA (discretizados)
logistic_kernel_pca_disc = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_kernel_pca_disc.fit(dt_train_kernel_pca_disc, y_train)
score_kernel_pca_disc = logistic_kernel_pca_disc.score(dt_test_kernel_pca_disc, y_test)

# Aplicar la regresión logística a los datos de Incremental PCA (originales)
logistic_incremental_pca_orig = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_incremental_pca_orig.fit(dt_train_incremental_pca_orig, y_train)
score_incremental_pca_orig = logistic_incremental_pca_orig.score(dt_test_incremental_pca_orig, y_test)

# Aplicar la regresión logística a los datos de Incremental PCA (normalizados)
logistic_incremental_pca_scaled = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_incremental_pca_scaled.fit(dt_train_incremental_pca_scaled, y_train)
score_incremental_pca_scaled = logistic_incremental_pca_scaled.score(dt_test_incremental_pca_scaled, y_test)

# Aplicar la regresión logística a los datos de Incremental PCA (discretizados)
logistic_incremental_pca_disc = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_incremental_pca_disc.fit(dt_train_incremental_pca_disc, y_train)
score_incremental_pca_disc = logistic_incremental_pca_disc.score(dt_test_incremental_pca_disc, y_test)

# Imprimir los resultados
print("SCORE Datos Originales:", score_orig)
print("SCORE Datos Normalizados:", score_scaled)
print("SCORE PCA (Originales):", score_pca_orig)
print("SCORE PCA (Normalizados):", score_pca_scaled)
print("SCORE PCA (Discretizados):", score_pca_disc)
print("SCORE Kernel PCA (Originales):", score_kernel_pca_orig)
print("SCORE Kernel PCA (Normalizados):", score_kernel_pca_scaled)
print("SCORE Kernel PCA (Discretizados):", score_kernel_pca_disc)
print("SCORE Incremental PCA (Originales):", score_incremental_pca_orig)
print("SCORE Incremental PCA (Normalizados):", score_incremental_pca_scaled)
print("SCORE Incremental PCA (Discretizados):", score_incremental_pca_disc)


