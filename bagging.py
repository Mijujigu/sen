import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    dt_heart = pd.read_csv('./data/DatasetFinal.csv')
    
    # Manejar los valores faltantes (NaN)
    dt_heart.dropna(inplace=True)
    
    # Separar los datos de entrada (X) y las etiquetas (y)
    x = dt_heart.drop(['%Toxicos'], axis=1)
    y = dt_heart['%Toxicos']
    
    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.35, random_state=1)
    
    # Normalizar los datos 
    scaler = StandardScaler()
    X_train_normalized = scaler.fit_transform(X_train)
    X_test_normalized = scaler.transform(X_test)
    
    # Entrenar y evaluar los modelos con los datos sin normalizar
    knn_class = KNeighborsClassifier().fit(X_train, y_train)
    knn_prediction = knn_class.predict(X_test)
    print('='*64)
    print('SCORE con KNN sin normalizar: ', accuracy_score(knn_prediction, y_test))
    
    estimators = {
        'LogisticRegression' : LogisticRegression(),
        'SVC' : SVC(),
        'LinearSVC' : LinearSVC(),
        'SGD' : SGDClassifier(loss="hinge", penalty="l2", max_iter=5),
        'KNN' : KNeighborsClassifier(),
        'DecisionTreeClf' : DecisionTreeClassifier(),
        'RandomTreeForest' : RandomForestClassifier(random_state=0)
    }
    
    print('='*64)
    print('Resultados con los datos sin normalizar:')
    print('='*64)
    
    for name, estimator in estimators.items():
        bag_class = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train, y_train)
        bag_predict = bag_class.predict(X_test)
        print('SCORE Bagging with {} sin normalizar: {}'.format(name, accuracy_score(bag_predict, y_test)))
    
    # Entrenar y evaluar los modelos con los datos normalizados
    knn_class_normalized = KNeighborsClassifier().fit(X_train_normalized, y_train)
    knn_prediction_normalized = knn_class_normalized.predict(X_test_normalized)
    print('='*64)
    print('SCORE con KNN normalizado: ', accuracy_score(knn_prediction_normalized, y_test))
    
    print('='*64)
    print('Resultados con los datos normalizados:')
    print('='*64)
    
    for name, estimator in estimators.items():
        bag_class_normalized = BaggingClassifier(base_estimator=estimator, n_estimators=50).fit(X_train_normalized, y_train)
        bag_predict_normalized = bag_class_normalized.predict(X_test_normalized)
        print('SCORE Bagging with {} normalizado: {}'.format(name, accuracy_score(bag_predict_normalized, y_test)))
