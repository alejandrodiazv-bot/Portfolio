"""Análisis de Precios de Viviendas en California """

import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
import numpy as np

def cargar_datos(ruta_archivo):
    """Carga y muestra información básica del dataset"""
    datos = pd.read_csv(ruta_archivo)               #'housing.csv'
    print("=== INFORMACIÓN DEL DATASET ===")
    print(f"Dimensiones: {datos.shape}")
    print(f"Columnas: {datos.columns.tolist()}")
    return datos

def explorar_datos(datos):
    """Realiza análisis exploratorio de datos"""
    print("\n=== ANÁLISIS EXPLORATORIO ===")
    print("\nPrimeras 5 filas:")
    print(datos.head())
    
    print("\nValores únicos en ocean_proximity:")
    print(datos['ocean_proximity'].value_counts())
    
    print("\nInformación del dataset:")
    datos.info()
    
    print("\nEstadísticas descriptivas:")
    print(datos.describe())

def visualizar_datos(datos):
    """Crea visualizaciones para entender los datos"""
    print("\n=== VISUALIZACIONES ===")
    
    # Histogramas
    plt.figure(figsize=(15, 8))
    datos.hist(bins=30, edgecolor="black")
    plt.tight_layout()
    plt.show()
    
    # Mapa de calor de correlaciones
    datos_numericos = datos.select_dtypes(include=[np.number])
    plt.figure(figsize=(12, 8))
    sb.heatmap(datos_numericos.corr(), annot=True, cmap='YlGnBu')
    plt.title('Mapa de Calor - Correlaciones')
    plt.show()

def preprocesar_datos(datos):
    """Limpia y preprocesa los datos para modelado"""
    print("\n=== PREPROCESAMIENTO ===")
    
    # Eliminar filas con valores nulos
    datos_limpios = datos.dropna().copy()
    print(f"Datos después de limpieza: {datos_limpios.shape}")
    
    # Convertir variable categórica a numérica
    dummies = pd.get_dummies(datos_limpios['ocean_proximity'], dtype=int)
    datos_limpios = datos_limpios.join(dummies)
    datos_limpios = datos_limpios.drop(['ocean_proximity'], axis=1)
    
    # Aplicar filtros simultáneos (CORRECCIÓN CRÍTICA)
    filtro_edad = datos_limpios['housing_median_age'] < 50
    filtro_precio = datos_limpios['median_house_value'] < 500000
    filtro_ingreso = datos_limpios['median_income'] < 15
    
    datos_filtrados = datos_limpios[filtro_edad & filtro_precio & filtro_ingreso]
    print(f"Datos después de filtrado: {datos_filtrados.shape}")
    
    return datos_filtrados

def entrenar_y_evaluar_modelo(datos):
    """Entrena y evalúa el modelo de regresión lineal"""
    print("\n=== ENTRENAMIENTO DEL MODELO ===")
    
    # Preparar características y variable objetivo
    X = datos.drop(['median_house_value'], axis=1)
    y = datos['median_house_value']
    
    # Dividir en entrenamiento y prueba
    X_ent, X_pru, y_ent, y_pru = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Conjunto de entrenamiento: {X_ent.shape}")
    print(f"Conjunto de prueba: {X_pru.shape}")
    
    # Escalar características (CORRECCIÓN CRÍTICA)
    scaler = StandardScaler()
    X_ent_escalado = scaler.fit_transform(X_ent)
    X_pru_escalado = scaler.transform(X_pru)  # SOLO transform, NO fit_transform
    
    # Entrenar modelo
    modelo = LinearRegression()
    modelo.fit(X_ent_escalado, y_ent)
    
    # Evaluar modelo
    predicciones = modelo.predict(X_pru_escalado)
    rmse = np.sqrt(mean_squared_error(y_pru, predicciones))
    
    print(f"\n=== RESULTADOS ===")
    print(f"R² Entrenamiento: {modelo.score(X_ent_escalado, y_ent):.4f}")
    print(f"R² Prueba: {modelo.score(X_pru_escalado, y_pru):.4f}")
    print(f"RMSE: {rmse:.2f}")
    
    # Validación cruzada (MEJORA ADICIONAL)
    scores = cross_val_score(modelo, X_ent_escalado, y_ent, cv=5, scoring='r2')
    print(f"R² Validación Cruzada (5-fold): {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
    
    return modelo, rmse

# EJECUCIÓN PRINCIPAL
if __name__ == "__main__":
    # Cargar datos
    datos = cargar_datos('housing.csv')
    
    # Análisis exploratorio
    explorar_datos(datos)
    visualizar_datos(datos)
    
    # Preprocesamiento
    datos_procesados = preprocesar_datos(datos)
    
    # Modelado
    modelo, rmse = entrenar_y_evaluar_modelo(datos_procesados)
