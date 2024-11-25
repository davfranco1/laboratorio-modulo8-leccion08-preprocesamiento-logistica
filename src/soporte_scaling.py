# Tratamiento de datos
# -----------------------------------------------------------------------
import numpy as np
import pandas as pd

# Feature scaling
# -----------------------------------------------------------------------
from sklearn.preprocessing import RobustScaler, MinMaxScaler, Normalizer, StandardScaler

# Gráficos
# -----------------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns


def aplicar_escaladores(df, columnas, escaladores, return_scaler=False):
    """
    Aplica múltiples escaladores a columnas específicas de un DataFrame y devuelve el DataFrame modificado.
    Puede opcionalmente devolver el primer escalador utilizado.

    Parámetros:
    - df: DataFrame de entrada en formato pandas.
    - columnas: Una lista de nombres de columnas que se quieren escalar.
    - escaladores: Una lista de instancias de escaladores (por ejemplo, [RobustScaler(), MinMaxScaler()]).
    - return_scaler: Booleano, si es True devuelve también el primer escalador utilizado.

    Retorna:
    - df_escalado: DataFrame con las columnas escaladas añadidas, con nombres de columnas que incluyen 
                   el sufijo correspondiente al nombre del escalador.
    - (Opcional) primer_escalador: El primer escalador utilizado para la transformación, si return_scaler=True.
    """
    # Crear una copia del DataFrame para no modificar el original
    df_escalado = df.copy()

    # Variable para guardar el primer escalador
    primer_escalador = None

    for i, escalador in enumerate(escaladores):
        # Ajustar y transformar las columnas seleccionadas
        datos_escalados = escalador.fit_transform(df[columnas])
        
        # Generar nombres de columnas basados en el nombre del escalador
        nombre_escalador = escalador.__class__.__name__.replace("Scaler", "").lower()
        nuevas_columnas = [f"{col}_{nombre_escalador}" for col in columnas]
        
        # Añadir las columnas escaladas al DataFrame
        df_escalado[nuevas_columnas] = datos_escalados

        # Guardar el primer escalador utilizado
        if i == 0:
            primer_escalador = escalador

    if return_scaler:
        return df_escalado, primer_escalador
    else:
        return df_escalado


def graficar_escaladores(df, variables_originales, variables_escaladas):
    """
    Genera gráficos comparativos (boxplots e histogramas) donde cada fila corresponde a una variable original
    y cada columna incluye la variable original y sus versiones escaladas.

    Parámetros:
    - df (pd.DataFrame): El DataFrame que contiene las variables originales y escaladas.
    - variables_originales (list): Lista de columnas originales.
    - variables_escaladas (list): Lista de columnas escaladas (deben incluir el nombre de la variable original en el nombre).

    Returns:
    - None: Muestra los gráficos directamente.
    """
    # Total de gráficos por fila (1 original + columnas escaladas asociadas)
    num_columnas = 5  # 4 escaladas + 1 original
    total_graficos = len(variables_originales) * 2 * num_columnas  # 2 tipos de gráficos: boxplot e histograma

    # Configurar figura y ejes
    filas = len(variables_originales) * 2  # 2 filas por variable original (boxplot e histograma)
    fig, axes = plt.subplots(
        nrows=filas, 
        ncols=num_columnas, 
        figsize=(num_columnas * 5, filas * 2.5)
    )
    axes = axes.flat

    # Generar gráficos para cada variable original
    current_plot = 0
    for variable in variables_originales:
        # Filtrar las variables escaladas correspondientes
        escaladas = [col for col in variables_escaladas if variable in col]

        # Boxplots (primera fila por variable)
        for col in [variable] + escaladas:
            sns.boxplot(x=col, data=df, ax=axes[current_plot])
            axes[current_plot].set_title(f"Boxplot: {col}")
            current_plot += 1

        # Histogramas (segunda fila por variable)
        for col in [variable] + escaladas:
            sns.histplot(x=col, data=df, ax=axes[current_plot])
            axes[current_plot].set_title(f"Histograma: {col}")
            current_plot += 1

    # Ajustar diseño
    plt.tight_layout()
    plt.show()