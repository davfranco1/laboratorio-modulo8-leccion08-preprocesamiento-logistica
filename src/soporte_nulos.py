from itertools import product
from sklearn.ensemble import IsolationForest
import numpy as np
import pandas as pd
from itertools import combinations
import seaborn as sns
import matplotlib.pyplot as plt


def outliers_isolation_forest(df, niveles_contaminacion, lista_estimadores):
    """
    Agrega columnas al DataFrame con la detección de outliers utilizando Isolation Forest
    para diferentes niveles de contaminación y números de estimadores.

    Parámetros:
    - df (pd.DataFrame): El DataFrame de entrada.
    - niveles_contaminacion (list): Lista de niveles de contaminación a probar (por ejemplo, [0.01, 0.05, 0.1]).
    - lista_estimadores (list): Lista de cantidades de estimadores a probar (por ejemplo, [10, 100, 200]).

    Returns:
    - pd.DataFrame: DataFrame con nuevas columnas para cada configuración de Isolation Forest.
    """
    # Seleccionar columnas numéricas
    col_numericas = df.select_dtypes(include=np.number).columns

    # Generar todas las combinaciones de niveles de contaminación y estimadores
    combinaciones = list(product(niveles_contaminacion, lista_estimadores))

    for cont, esti in combinaciones:
        # Inicializar Isolation Forest
        ifo = IsolationForest(
            n_estimators=esti,
            contamination=cont,
            n_jobs=-1  # Usar todos los núcleos disponibles
        )

        # Ajustar y predecir outliers
        df[f"outliers_ifo_{cont}_{esti}"] = ifo.fit_predict(df[col_numericas])
    
    return df



def visualizar_outliers(df, cols_numericas, figsize = (15,5)):
    """
    Genera visualizaciones para las combinaciones de columnas numéricas utilizando las columnas
    que identifican outliers como `hue` en gráficos scatter.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene los datos.
    - cols_numericas (list): Lista de columnas numéricas a combinar.

    Returns:
    - None: Muestra los gráficos directamente.
    """
    # Crear todas las combinaciones de pares de columnas numéricas
    combinaciones_viz = list(combinations(cols_numericas, 2))
    columnas_hue = df.filter(like = "outlier").columns

    for outlier in columnas_hue:
        # Crear una figura con subplots
        fig, axes = plt.subplots(nrows=1, ncols=len(combinaciones_viz), figsize=figsize)
        axes = axes.flat if isinstance(axes, np.ndarray) else [axes]

        for indice, tupla in enumerate(combinaciones_viz):
            sns.scatterplot(
                x=tupla[0],
                y=tupla[1],
                ax=axes[indice],
                data=df,
                hue=outlier,
                palette="Set1",
                style=outlier,
                alpha=0.4)
        
        plt.suptitle(outlier)
        plt.tight_layout()