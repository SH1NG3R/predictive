import sys
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QLabel, QFileDialog, 
                            QDoubleSpinBox, QGroupBox, QGridLayout, QMessageBox,
                            QTabWidget)
from PyQt6.QtCore import Qt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import warnings
warnings.filterwarnings('ignore')

class MotorPredictorApp(QMainWindow):
    """Aplicación principal para predicción de fallas en motores."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sistema de Predicción de Fallas en Motores")
        self.setGeometry(100, 100, 1200, 800)
        
        # Inicializar variables de estado
        self.rf_model = None
        self.dl_model = None
        self.scaler = None
        self.training_history = None
        self.data = None
        self.confusion_matrix = None
        
        # Configurar interfaz
        self.init_ui()
        
        # Cargar datos y modelos
        self.load_data()
        self.load_models()
    
    def init_ui(self):
        """Inicializar la interfaz de usuario con todas las pestañas."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Crear widget de pestañas
        tab_widget = QTabWidget()
        
        # 1. Pestaña de predicción
        prediction_tab = self.create_prediction_tab()
        tab_widget.addTab(prediction_tab, "Predicción")
        
        # 2. Pestaña de estadísticas
        stats_tab = self.create_stats_tab()
        tab_widget.addTab(stats_tab, "Estadísticas del Dataset")
        
        # 3. Pestaña de métricas de entrenamiento
        training_tab = self.create_training_tab()
        tab_widget.addTab(training_tab, "Métricas de Entrenamiento")
        
        main_layout.addWidget(tab_widget)
    
    def create_prediction_tab(self):
        """Crear pestaña de predicción con controles de entrada."""
        tab = QWidget()
        layout = QHBoxLayout(tab)
        
        # Panel de entrada
        input_group = QGroupBox("Datos del Motor")
        input_layout = QGridLayout()
        
        # Crear campos de entrada
        self.inputs = {}
        labels = {
            'temp_aire': ('Temperatura del Aire (°C):', 0, 50, 25),
            'temp_proceso': ('Temperatura del Proceso (°C):', 0, 100, 35),
            'vibracion': ('Nivel de Vibración:', 0, 100, 30),
            'horas': ('Horas de Operación:', 0, 1000, 50)
        }
        
        for i, (key, (label, min_val, max_val, default)) in enumerate(labels.items()):
            input_layout.addWidget(QLabel(label), i, 0)
            self.inputs[key] = QDoubleSpinBox()
            self.inputs[key].setRange(min_val, max_val)
            self.inputs[key].setValue(default)
            input_layout.addWidget(self.inputs[key], i, 1)
        
        # Botón de predicción
        predict_button = QPushButton("Predecir Estado")
        predict_button.clicked.connect(self.make_prediction)
        input_layout.addWidget(predict_button, len(labels), 0, 1, 2)
        
        input_group.setLayout(input_layout)
        layout.addWidget(input_group)
        
        # Panel de resultados
        results_group = QGroupBox("Resultados del Análisis")
        results_layout = QVBoxLayout()
        
        self.prediction_canvas = FigureCanvas(Figure(figsize=(8, 6)))
        results_layout.addWidget(self.prediction_canvas)
        
        self.result_label = QLabel("Esperando datos...")
        self.result_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        results_layout.addWidget(self.result_label)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        return tab
    
    def create_stats_tab(self):
        """Crear pestaña de estadísticas del dataset."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.stats_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.stats_canvas)
        return tab
    
    def create_training_tab(self):
        """Crear pestaña de métricas de entrenamiento."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.training_canvas = FigureCanvas(Figure(figsize=(10, 8)))
        layout.addWidget(self.training_canvas)
        return tab
    
    def load_data(self):
        """Cargar y procesar el dataset."""
        try:
            self.data = pd.read_csv('Dataset_clean.csv')
            self.update_statistics_visualization()
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error al cargar datos: {str(e)}")
##
    def load_models(self):
        """Cargar modelos pre-entrenados o entrenar nuevos."""
        try:
            # Intentar cargar modelos existentes
            self.rf_model = joblib.load('random_forest_model.joblib')
            self.dl_model = load_model('deep_learning_model.h5')
            self.scaler = joblib.load('scaler.joblib')
            
            # Cargar historial de entrenamiento si existe
            try:
                training_history = joblib.load('training_history.joblib')
                self.training_history = training_history
                
                # Cargar matriz de confusión guardada
                self.confusion_matrix = joblib.load('confusion_matrix.joblib')
                
                # Actualizar visualización de métricas
                self.update_training_metrics()
                print("Modelos y métricas cargados exitosamente")
            except:
                print("Modelos cargados, pero sin historial de entrenamiento")
                # Realizar un entrenamiento rápido para generar métricas
                self.generate_training_metrics()
                
        except:
            print("Entrenando nuevos modelos...")
            self.train_new_models()

    def generate_training_metrics(self):
        """Genera métricas para modelos pre-entrenados."""
        if self.data is None or self.rf_model is None or self.dl_model is None:
            return
            
        # Preparar datos
        X = self.data[['Air temperature [K]', 'Process temperature [K]', 
                    'Vibration Levels', 'Operational Hours']]
        y = self.data['Failure Type']
        
        # División de datos
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        
        # Escalar datos
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Generar matriz de confusión
        y_pred = self.rf_model.predict(X_test_scaled)
        self.confusion_matrix = confusion_matrix(y_test, y_pred)
        
        # Generar historial simulado para visualización
        # Esto nos permite mostrar métricas incluso con modelos pre-entrenados
        self.training_history = type('', (), {})()
        self.training_history.history = {
            'loss': [self.dl_model.evaluate(X_train_scaled, 
                                        pd.Categorical(y_train).codes, 
                                        verbose=0)[0]],
            'val_loss': [self.dl_model.evaluate(X_test_scaled, 
                                            pd.Categorical(y_test).codes, 
                                            verbose=0)[0]],
            'accuracy': [self.dl_model.evaluate(X_train_scaled, 
                                            pd.Categorical(y_train).codes, 
                                            verbose=0)[1]],
            'val_accuracy': [self.dl_model.evaluate(X_test_scaled, 
                                                pd.Categorical(y_test).codes, 
                                                verbose=0)[1]]
        }
        
        # Guardar métricas
        joblib.dump(self.training_history, 'training_history.joblib')
        joblib.dump(self.confusion_matrix, 'confusion_matrix.joblib')
        
        # Actualizar visualización
        self.update_training_metrics()

    def train_new_models(self):
        """Entrenar nuevos modelos y guardar métricas."""
        try:
            if self.data is None:
                raise ValueError("No hay datos cargados")
            
            # Preparar datos
            X = self.data[['Air temperature [K]', 'Process temperature [K]', 
                        'Vibration Levels', 'Operational Hours']]
            y = self.data['Failure Type']
            
            # División de datos
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42)
            
            # Escalar datos
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Entrenar Random Forest
            print("Entrenando Random Forest...")
            self.rf_model = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced'
            )
            self.rf_model.fit(X_train_scaled, y_train)
            
            # Generar y guardar matriz de confusión
            y_pred = self.rf_model.predict(X_test_scaled)
            self.confusion_matrix = confusion_matrix(y_test, y_pred)
            joblib.dump(self.confusion_matrix, 'confusion_matrix.joblib')
            
            # Entrenar modelo Deep Learning
            print("Entrenando Deep Learning...")
            self.dl_model = Sequential([
                Dense(64, activation='relu', input_shape=(4,)),
                Dropout(0.3),
                Dense(32, activation='relu'),
                Dropout(0.2),
                Dense(16, activation='relu'),
                Dense(len(np.unique(y)), activation='softmax')
            ])
            
            self.dl_model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Entrenar con registro de historia
            self.training_history = self.dl_model.fit(
                X_train_scaled,
                pd.Categorical(y_train).codes,
                epochs=50,
                validation_split=0.2,
                callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
                verbose=1
            )
            
            # Guardar todos los modelos y métricas
            joblib.dump(self.rf_model, 'random_forest_model.joblib')
            self.dl_model.save('deep_learning_model.h5')
            joblib.dump(self.scaler, 'scaler.joblib')
            joblib.dump(self.training_history, 'training_history.joblib')
            
            # Actualizar visualización
            self.update_training_metrics()
            print("Entrenamiento completado y métricas actualizadas")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Error en entrenamiento: {str(e)}")

##    
    def make_prediction(self):
        """Realizar predicción con los datos ingresados."""
        if not all([self.rf_model, self.dl_model, self.scaler]):
            QMessageBox.warning(self, "Error", "Modelos no cargados correctamente")
            return
        
        # Obtener datos de entrada
        input_data = np.array([[
            self.inputs['temp_aire'].value() + 273.15,
            self.inputs['temp_proceso'].value() + 273.15,
            self.inputs['vibracion'].value(),
            self.inputs['horas'].value()
        ]])
        
        # Escalar datos
        input_scaled = self.scaler.transform(input_data)
        
        # Obtener predicciones
        rf_pred = self.rf_model.predict_proba(input_scaled)[0]
        dl_pred = self.dl_model.predict(input_scaled)[0]
        
        # Combinar predicciones (60% DL, 40% RF)
        combined_pred = 0.4 * rf_pred + 0.6 * dl_pred
        
        # Actualizar visualización
        self.update_prediction_visualization(combined_pred)
        
        # Generar resultado
        failure_types = ['Sin Falla', 'Falla por Desgaste', 
                        'Falla de Energía', 'Falla por Sobrecarga']
        pred_class = failure_types[np.argmax(combined_pred)]
        confidence = np.max(combined_pred) * 100
        
        if pred_class == 'Sin Falla':
            time_to_failure = self.estimate_time_to_failure(input_scaled)
            result_text = f"Estado: {pred_class} (Confianza: {confidence:.1f}%)\n"
            result_text += f"Tiempo estimado hasta posible falla: {time_to_failure:.1f} horas"
        else:
            result_text = f"¡Atención! Se detectó: {pred_class}\n"
            result_text += f"Confianza: {confidence:.1f}%"
        
        self.result_label.setText(result_text)
    
    def update_prediction_visualization(self, probabilities):
        """Actualizar visualización de predicciones."""
        fig = self.prediction_canvas.figure
        fig.clear()
        
        ax = fig.add_subplot(111)
        labels = ['Sin Falla', 'Desgaste', 'Falla Energía', 'Sobrecarga']
        colors = ['green', 'yellow', 'red', 'orange']
        
        bars = ax.bar(labels, probabilities, color=colors)
        ax.set_ylabel('Probabilidad')
        ax.set_title('Probabilidad de cada tipo de falla')
        
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height*100:.1f}%',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45)
        fig.tight_layout()
        self.prediction_canvas.draw()
    
    def update_statistics_visualization(self):
        """Actualizar visualización de estadísticas del dataset."""
        if self.data is None:
            return
        
        fig = self.stats_canvas.figure
        fig.clear()
        
        gs = fig.add_gridspec(2, 2)
        
        # 1. Distribución de tipos de falla
        ax1 = fig.add_subplot(gs[0, 0])
        failure_counts = self.data['Failure Type'].value_counts()
        sns.barplot(x=failure_counts.values, y=failure_counts.index, ax=ax1)
        ax1.set_title('Distribución de Tipos de Falla')
        
        # 2. Correlación entre variables
        ax2 = fig.add_subplot(gs[0, 1])
        numeric_cols = ['Air temperature [K]', 'Process temperature [K]', 
                       'Vibration Levels', 'Operational Hours']
        correlation_matrix = self.data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax2)
        ax2.set_title('Matriz de Correlación')
        
        # 3. Distribución de vibraciones
        ax3 = fig.add_subplot(gs[1, 0])
        sns.boxplot(data=self.data, x='Failure Type', y='Vibration Levels', ax=ax3)
        ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45)
        ax3.set_title('Niveles de Vibración por Tipo de Falla')
        
        # 4. Temperatura vs Vibración
        ax4 = fig.add_subplot(gs[1, 1])
        sns.scatterplot(data=self.data, x='Process temperature [K]',
                       y='Vibration Levels', hue='Failure Type', ax=ax4)
        ax4.set_title('Temperatura vs Vibración')
        
        fig.tight_layout()
        self.stats_canvas.draw()
    
    def update_training_metrics(self):
        """Actualizar visualización de métricas de entrenamiento."""
        if self.training_history is None:
            return
        
        fig = self.training_canvas.figure
        fig.clear()
        
        gs = fig.add_gridspec(2, 2)
        
        # 1. Curvas de aprendizaje
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(self.training_history.history['loss'], label='Training')
        ax1.plot(self.training_history.history['val_loss'], label='Validation')
        ax1.set_title('Curvas de Pérdida')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Pérdida')
        ax1.legend()
        
        # 2. Accuracy durante entrenamiento
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(self.training_history.history['accuracy'], label='Training')
        ax2.plot(self.training_history.history['val_accuracy'], label='Validation')
        ax2.set_title('Precisión durante Entrenamiento')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Precisión')
        ax2.legend()
        
        # 3. Matriz de confusión
        if hasattr(self, 'confusion_matrix'):
            ax3 = fig.add_subplot(gs[1, 0])
            sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax3)
            ax3.set_title('Matriz de Confusión')
            ax3.set_xlabel('Predicción')
            ax3.set_ylabel('Real')
        
        # 4. Importancia de características
        if self.rf_model is not None:
            ax4 = fig.add_subplot(gs[1, 1])
            feature_importance = pd.Series(
                self.rf_model.feature_importances_,
                index=['Temp. Aire', 'Temp. Proceso', 'Vibración', 'Horas']
            )
            sns.barplot(x=feature_importance.values, y=feature_importance.index, ax=ax4)
            ax4.set_title('Importancia de Variables')
        
        fig.tight_layout()
        self.training_canvas.draw()
    
    def estimate_time_to_failure(self, current_state):
        """
        Estimar tiempo hasta la próxima falla basado en el estado actual.
        Utiliza un modelo basado en reglas y estadísticas del dataset.
        """
        # Tiempo base de operación
        base_hours = 100
        
        # Factores de ajuste basados en las variables actuales
        vibration_factor = 1 - (current_state[0][2] / 100)  # Menor vibración = más tiempo
        temp_factor = 1 - ((current_state[0][1] - 308) / 10)  # Temperatura más alta = menos tiempo
        operation_factor = 1 - (current_state[0][3] / 1000)  # Más horas = menos tiempo restante
        
        # Combinar factores con pesos
        estimated_hours = base_hours * (
            0.4 * vibration_factor +
            0.3 * temp_factor +
            0.3 * operation_factor
        )
        
        # Ajustar según probabilidades del modelo
        if self.rf_model is not None:
            failure_prob = 1 - self.rf_model.predict_proba(current_state)[0][0]
            estimated_hours *= (1 - failure_prob)
        
        return max(estimated_hours, 0)

def main():
    """Función principal para iniciar la aplicación."""
    app = QApplication(sys.argv)
    
    # Configurar estilo de la aplicación
    app.setStyle('Fusion')
    
    # Crear y mostrar la ventana principal
    window = MotorPredictorApp()
    window.show()
    
    # Iniciar el loop de eventos
    sys.exit(app.exec())

if __name__ == '__main__':
    main()