import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.fft import fft
from datetime import datetime, timedelta
import glob
import joblib
from scipy import stats
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA

class AnalizadorVibraciones:
    def __init__(self):
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.classifier = RandomForestClassifier(random_state=42)
        self.pca = PCA(n_components=2)
        
    def cargar_datos(self, ruta_archivos):
        """Carga todos los archivos CSV de la ruta especificada"""
        archivos = glob.glob(ruta_archivos + "/*.csv")
        dataframes = []
        
        for archivo in archivos:
            df = pd.read_csv(archivo)
            df['timestamp'] = pd.to_datetime(df['Fecha'] + ' ' + df['Hora'])
            dataframes.append(df)
            
        self.datos = pd.concat(dataframes).sort_values('timestamp')
        return self.datos
    
    def preprocesar_datos(self):
        """Preprocesa los datos y extrae características"""
        # Calcular características en ventanas de tiempo
        ventana = '1H'  # Ventana de 1 hora
        
        features = pd.DataFrame()
        features['media'] = self.datos.groupby(pd.Grouper(key='timestamp', freq=ventana))['Vibracion'].mean()
        features['std'] = self.datos.groupby(pd.Grouper(key='timestamp', freq=ventana))['Vibracion'].std()
        features['max'] = self.datos.groupby(pd.Grouper(key='timestamp', freq=ventana))['Vibracion'].max()
        features['min'] = self.datos.groupby(pd.Grouper(key='timestamp', freq=ventana))['Vibracion'].min()
        features['rms'] = np.sqrt(np.mean(self.datos.groupby(pd.Grouper(key='timestamp', freq=ventana))['Vibracion'].apply(lambda x: x**2)))
        
        # Calcular FFT para cada ventana
        def calcular_caracteristicas_fft(grupo):
            fft_vals = np.abs(fft(grupo['Vibracion'].values))
            return pd.Series({
                'fft_max': np.max(fft_vals),
                'fft_mean': np.mean(fft_vals),
                'fft_std': np.std(fft_vals)
            })
        
        fft_features = self.datos.groupby(pd.Grouper(key='timestamp', freq=ventana)).apply(calcular_caracteristicas_fft)
        features = pd.concat([features, fft_features], axis=1)
        
        self.features = features.dropna()
        return self.features
    
    def detectar_anomalias(self):
        """Detecta anomalías usando Isolation Forest"""
        X = self.scaler.fit_transform(self.features)
        anomalias = self.isolation_forest.fit_predict(X)
        self.features['anomalia'] = anomalias
        return anomalias
    
    def entrenar_clasificador(self, X_train, y_train):
        """Entrena un clasificador para predecir estados de la máquina"""
        self.classifier.fit(X_train, y_train)
        
    def predecir_estado(self, X):
        """Predice el estado de la máquina"""
        return self.classifier.predict(X)
    
    def analizar_tendencias(self):
        """Analiza tendencias en los datos"""
        # Calcular tendencia lineal
        x = np.arange(len(self.features['media']))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, self.features['media'])
        
        tendencia = {
            'pendiente': slope,
            'r_cuadrado': r_value**2,
            'p_valor': p_value
        }
        
        return tendencia
    
    def visualizar_resultados(self):
        """Genera visualizaciones de los resultados"""
        # Reducción de dimensionalidad para visualización
        X_pca = self.pca.fit_transform(self.scaler.transform(self.features.drop('anomalia', axis=1)))
        self.features['PCA1'] = X_pca[:, 0]
        self.features['PCA2'] = X_pca[:, 1]
        
        return self.features

def crear_app_streamlit():
    st.title('Análisis de Vibraciones con Machine Learning')
    
    # Sidebar para controles
    st.sidebar.header('Configuración')
    uploaded_files = st.sidebar.file_uploader("Cargar archivos CSV", accept_multiple_files=True, type=['csv'])
    
    if uploaded_files:
        # Inicializar analizador
        analizador = AnalizadorVibraciones()
        
        # Cargar y procesar datos
        all_data = []
        for file in uploaded_files:
            df = pd.read_csv(file)
            all_data.append(df)
        
        datos = pd.concat(all_data)
        datos['timestamp'] = pd.to_datetime(datos['Fecha'] + ' ' + datos['Hora'])
        
        # Procesar datos
        analizador.datos = datos
        features = analizador.preprocesar_datos()
        anomalias = analizador.detectar_anomalias()
        tendencias = analizador.analizar_tendencias()
        
        # Dashboard principal
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader('Estadísticas Generales')
            st.write(f"Total de mediciones: {len(datos)}")
            st.write(f"Anomalías detectadas: {sum(anomalias == -1)}")
            st.write(f"Período de análisis: {datos['timestamp'].min()} a {datos['timestamp'].max()}")
        
        with col2:
            st.subheader('Análisis de Tendencias')
            st.write(f"Tendencia: {'Creciente' if tendencias['pendiente'] > 0 else 'Decreciente'}")
            st.write(f"Confianza (R²): {tendencias['r_cuadrado']:.3f}")
        
        # Gráficos
        st.subheader('Serie Temporal de Vibraciones')
        fig_timeline = px.line(datos, x='timestamp', y='Vibracion', 
                             title='Vibraciones a lo largo del tiempo')
        st.plotly_chart(fig_timeline)
        
        # Visualización de anomalías
        st.subheader('Detección de Anomalías')
        features_viz = analizador.visualizar_resultados()
        fig_anomalias = px.scatter(features_viz, x='PCA1', y='PCA2', 
                                 color='anomalia',
                                 title='Visualización de Anomalías (PCA)',
                                 color_discrete_map={1: 'blue', -1: 'red'})
        st.plotly_chart(fig_anomalias)
        
        # Análisis de frecuencias
        st.subheader('Análisis de Frecuencias')
        fft_vals = np.abs(fft(datos['Vibracion'].values))
        freq = np.fft.fftfreq(len(datos['Vibracion'].values))
        fig_fft = px.line(x=freq[:len(freq)//2], y=fft_vals[:len(freq)//2], 
                         title='Espectro de Frecuencias')
        st.plotly_chart(fig_fft)
        
        # Exportar modelo
        if st.sidebar.button('Exportar Modelo'):
            modelo_export = {
                'scaler': analizador.scaler,
                'isolation_forest': analizador.isolation_forest,
                'classifier': analizador.classifier
            }
            joblib.dump(modelo_export, 'modelo_vibraciones.joblib')
            st.sidebar.success('Modelo exportado correctamente')

if __name__ == "__main__":
    crear_app_streamlit()
