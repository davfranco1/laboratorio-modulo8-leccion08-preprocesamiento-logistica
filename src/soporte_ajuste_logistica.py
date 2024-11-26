# Tratamiento de datos
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Visualizaciones
# -----------------------------------------------------------------------
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree

# Para realizar la clasificación y la evaluación del modelo
# -----------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV, cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
    confusion_matrix,
    roc_curve
)

import xgboost as xgb
import pickle

# Para realizar cross validation
# -----------------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, cross_val_score, KFold
from sklearn.preprocessing import KBinsDiscretizer


class AnalisisModelosClasificacion:
    def __init__(self, dataframe, variable_dependiente):
        self.dataframe = dataframe
        self.variable_dependiente = variable_dependiente
        self.X = dataframe.drop(variable_dependiente, axis=1)
        self.y = dataframe[variable_dependiente]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, train_size=0.8, random_state=42, shuffle=True
        )

        # Diccionario de modelos y resultados
        self.modelos = {
            "logistic_regression": LogisticRegression(),
            "tree": DecisionTreeClassifier(),
            "random_forest": RandomForestClassifier(),
            "gradient_boosting": GradientBoostingClassifier(),
            "xgboost": xgb.XGBClassifier()
        }
        self.resultados = {nombre: {"mejor_modelo": None, "pred_train": None, "pred_test": None} for nombre in self.modelos}

    def ajustar_modelo(self, modelo_nombre, param_grid=None, cross_validation = 5):
        """
        Ajusta el modelo seleccionado con GridSearchCV.
        """
        if modelo_nombre not in self.modelos:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.modelos[modelo_nombre]

        # Parámetros predeterminados por modelo
        parametros_default = {
            "logistic_regression": [
                {'penalty': ['l1'], 'solver': ['saga'], 'C': [0.0000000001, 0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': ['l2'], 'solver': ['liblinear'], 'C': [ 0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]},
                {'penalty': ['elasticnet'], 'solver': ['saga'], 'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9], 'C': [ 0.001, 0.01, 0.1, 1, 10, 100], 'max_iter': [10000]}
            ],
            "tree": {
                'max_depth': [3, 5, 7, 10],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            "random_forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            "gradient_boosting": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'subsample': [0.8, 1.0]
            },
            "xgboost": {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        }

        if param_grid is None:
            param_grid = parametros_default.get(modelo_nombre, {})

        # Ajuste del modelo
        grid_search = GridSearchCV(estimator=modelo, 
                                   param_grid=param_grid, 
                                   cv=cross_validation, 
                                   scoring='accuracy')
        
        grid_search.fit(self.X_train, self.y_train)
        self.resultados[modelo_nombre]["mejor_modelo"] = grid_search.best_estimator_
        self.resultados[modelo_nombre]["pred_train"] = grid_search.best_estimator_.predict(self.X_train)
        self.resultados[modelo_nombre]["pred_test"] = grid_search.best_estimator_.predict(self.X_test)

        # Guardar el modelo
        with open('mejor_modelo.pkl', 'wb') as f:
            pickle.dump(grid_search.best_estimator_, f)

    def calcular_metricas(self, modelo_nombre):
        """
        Calcula métricas de rendimiento para el modelo seleccionado, incluyendo AUC y Kappa.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        pred_train = self.resultados[modelo_nombre]["pred_train"]
        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_train is None or pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular métricas.")
        
        # Calcular probabilidades para AUC (si el modelo las soporta)
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if hasattr(modelo, "predict_proba"):
            prob_train = modelo.predict_proba(self.X_train)[:, 1]
            prob_test = modelo.predict_proba(self.X_test)[:, 1]
        else:
            prob_train = prob_test = None  # Si no hay probabilidades, AUC no será calculado

        # Métricas para conjunto de entrenamiento
        metricas_train = {
            "accuracy": accuracy_score(self.y_train, pred_train),
            "precision": precision_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "recall": recall_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "f1": f1_score(self.y_train, pred_train, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_train, pred_train),
            "auc": roc_auc_score(self.y_train, prob_train) if prob_train is not None else None
        }

        # Métricas para conjunto de prueba
        metricas_test = {
            "accuracy": accuracy_score(self.y_test, pred_test),
            "precision": precision_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "recall": recall_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "f1": f1_score(self.y_test, pred_test, average='weighted', zero_division=0),
            "kappa": cohen_kappa_score(self.y_test, pred_test),
            "auc": roc_auc_score(self.y_test, prob_test) if prob_test is not None else None
        }

        # Combinar métricas en un DataFrame
        return pd.DataFrame({"train": metricas_train, "test": metricas_test})

    def plot_matriz_confusion(self, modelo_nombre):
        """
        Plotea la matriz de confusión para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")

        pred_test = self.resultados[modelo_nombre]["pred_test"]

        if pred_test is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular la matriz de confusión.")

        # Matriz de confusión
        matriz_conf = confusion_matrix(self.y_test, pred_test)
        plt.figure(figsize=(8, 6))
        sns.heatmap(matriz_conf, annot=True, fmt='g', cmap='Blues')
        plt.title(f"Matriz de Confusión ({modelo_nombre})")
        plt.xlabel("Predicción")
        plt.ylabel("Valor Real")
        plt.show()
    
    def curva_roc(self, modelo_nombre):
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Se debe ajustar el modelo '{modelo_nombre}' antes de calcular la curva ROC.")
        
        if not hasattr(modelo, "predict_proba"):
            raise ValueError(f"El modelo '{modelo_nombre}' no soporta la predicción de probabilidades.")
        
        # Get predicted probabilities for the positive class
        y_pred_test_prob = modelo.predict_proba(self.X_test)[:, 1]
        
        # Calculate ROC curve metrics
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_test_prob)
        
        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        sns.lineplot(x=fpr, y=tpr, color="blue", label="Modelo")
        sns.lineplot(x=[0, 1], y=[0, 1], color="grey", linestyle="--", label="Aleatorio")
        
        plt.xlabel("Tasa de Falsos Positivos (1 - Especificidad)")
        plt.ylabel("Tasa de Verdaderos Positivos (Sensibilidad)")
        plt.title(f"Curva ROC: {modelo_nombre}")
        plt.legend(loc="lower right")
        plt.show()
    
    def importancia_predictores(self, modelo_nombre):
        """
        Calcula y grafica la importancia de las características para el modelo seleccionado.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de calcular importancia de características.")
        
        # Verificar si el modelo tiene importancia de características
        if hasattr(modelo, "feature_importances_"):
            importancia = modelo.feature_importances_
        elif modelo_nombre == "logistic_regression" and hasattr(modelo, "coef_"):
            importancia = modelo.coef_[0]
        else:
            print(f"El modelo '{modelo_nombre}' no soporta la importancia de características.")
            return
        
        # Crear DataFrame y graficar
        importancia_df = pd.DataFrame({
            "Feature": self.X.columns,
            "Importance": importancia
        }).sort_values(by="Importance", ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x="Importance", y="Feature", data=importancia_df, palette="viridis")
        plt.title(f"Importancia de Características ({modelo_nombre})")
        plt.xlabel("Importancia")
        plt.ylabel("Características")
        plt.show()

    def filtrar_errores(self, modelo_nombre, tipo_error):
        """
        Filtra los errores de predicción del modelo especificado.

        Dependiendo del tipo de error indicado, devuelve las muestras clasificadas como
        falsos positivos o falsos negativos.

        Un falso positivo ocurre cuando el modelo predice la clase positiva (1),
        pero el valor real es negativo (0).
        Un falso negativo ocurre cuando el modelo predice la clase negativa (0),
        pero el valor real es positivo (1).

        Parámetros:
            modelo_nombre (str): Nombre del modelo cuyo rendimiento se evaluará.
            tipo_error (str): Tipo de error a filtrar. Debe ser "falsos_positivos" o "falsos_negativos".

        Devuelve:
            DataFrame: Contiene las muestras clasificadas como el tipo de error especificado,
            incluyendo las características originales y los valores reales/predichos.

        Excepciones:
            ValueError: Si el modelo no está entrenado, no es reconocido o si el tipo de error no es válido.
        """
        if modelo_nombre not in self.resultados:
            raise ValueError(f"Modelo '{modelo_nombre}' no reconocido.")
        
        # Obtener el modelo entrenado
        modelo = self.resultados[modelo_nombre]["mejor_modelo"]
        if modelo is None:
            raise ValueError(f"Debe ajustar el modelo '{modelo_nombre}' antes de filtrar errores.")
        
        if tipo_error not in ["fp", "fn"]:
            raise ValueError("El tipo de error debe ser 'fn' o 'fp'.")
        
        # Realizar predicciones
        y_pred = modelo.predict(self.X_test)
        
        # Crear un DataFrame para comparación
        resultados = pd.DataFrame({
            "real": self.y_test,
            "predicho": y_pred
        }, index=self.X_test.index)

        # Agregar los datos originales del conjunto de prueba para contexto
        resultados = pd.concat([self.dataframe.loc[resultados.index], resultados], axis=1)

        # Filtrar según el tipo de error
        if tipo_error == "fp":
            errores = resultados[(resultados["real"] == 0) & (resultados["predicho"] == 1)]
        elif tipo_error == "fn":
            errores = resultados[(resultados["real"] == 1) & (resultados["predicho"] == 0)]
        
        return errores


def revertir_datos_transformados(df_codificado, scaler_path, scaler_columns, encoders_info):
    """
    Revierte las transformaciones de codificación y estandarización realizadas en un DataFrame.

    Parámetros:
        df_codificado (pd.DataFrame): DataFrame con las características codificadas y estandarizadas.
        scaler_path (str): Ruta al archivo pickle del scaler guardado (e.g., StandardScaler o MinMaxScaler).
        scaler_columns (list): Lista de columnas que fueron escaladas y necesitan ser revertidas.
        encoders_info (dict): Diccionario que mapea encoders a una lista de columnas y sus rutas pickle.
                             Formato: {"ruta_encoder.pkl": ["columna1", "columna2", ...], ...}

    Devuelve:
        pd.DataFrame: DataFrame con los datos revertidos a su estado original.
    """
    import numpy as np

    # Cargar el scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Revertir la estandarización solo para las columnas seleccionadas
    df_revertido = df_codificado.copy()
    if scaler_columns:
        try:
            df_revertido[scaler_columns] = scaler.inverse_transform(df_codificado[scaler_columns])
        except ValueError as e:
            raise ValueError(f"Error al revertir escalado: {e}. Verifique las columnas del DataFrame.") from e

    # Revertir codificación para las columnas categóricas
    for encoder_path, columnas in encoders_info.items():
        # Cargar el encoder correspondiente
        with open(encoder_path, 'rb') as f:
            encoder = pickle.load(f)

        # Si el encoder es TargetEncoder, usar mapeo interno
        if hasattr(encoder, "mapping_"):
            for columna in columnas:
                if columna in df_revertido.columns:
                    try:
                        # Obtener el mapeo invertido del TargetEncoder
                        if columna not in encoder.mapping_:
                            raise ValueError(f"Mapping no encontrado para la columna '{columna}'.")
                        mapping = encoder.mapping_[columna]
                        inverse_mapping = {v: k for k, v in mapping.items()}
                        df_revertido[columna] = df_revertido[columna].map(inverse_mapping)
                    except KeyError as e:
                        raise ValueError(f"Error al revertir target encoding para la columna '{columna}': {e}.") from e

        # Si el encoder es OneHotEncoder, procesar múltiples columnas
        elif hasattr(encoder, "inverse_transform") and hasattr(encoder, "categories_"):
            try:
                ohe_data = df_revertido[columnas].values
                if ohe_data.shape[1] != len(encoder.categories_[0]):
                    # Agregar columnas faltantes con ceros si faltan categorías
                    missing_categories = len(encoder.categories_[0]) - ohe_data.shape[1]
                    if missing_categories > 0:
                        ohe_data = np.hstack([ohe_data, np.zeros((ohe_data.shape[0], missing_categories))])
                
                # Revertir transformación
                reverted_col = encoder.inverse_transform(ohe_data)
                df_revertido[columnas[0] + "_reverted"] = reverted_col
                df_revertido.drop(columns, axis=1, inplace=True)
            except ValueError as e:
                raise ValueError(f"Error al revertir codificación para las columnas {columnas}: {e}.") from e

        # Si el encoder es LabelEncoder u otro similar
        else:
            for columna in columnas:
                if columna in df_revertido.columns:
                    try:
                        df_revertido[columna] = encoder.inverse_transform(df_revertido[columna].astype(int))
                    except ValueError as e:
                        raise ValueError(f"Error al revertir codificación para la columna '{columna}': {e}.") from e
    
    return df_revertido