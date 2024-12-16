import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import random
from datetime import timedelta

class IndustrialDataAugmenter:
    """
    Clase para aumentar datos industriales y convertir unidades y etiquetas
    """
    
    # Diccionario de traducción de fallas
    FAILURE_TRANSLATIONS = {
        'No Failure': 'Sin Falla',
        'Tool Wear Failure': 'Falla por Desgaste de Herramienta',
        'Power Failure': 'Falla de Energía',
        'Overstrain Failure': 'Falla por Sobrecarga'
    }
    
    def __init__(self, data):
        """Inicializar con el dataset original"""
        self.data = data.copy()
        self.scaler = StandardScaler()
        
    def kelvin_to_celsius(self, temp_k):
        """Convertir temperatura de Kelvin a Celsius"""
        return temp_k - 273.15
    
    def translate_and_convert(self):
        """Traducir etiquetas y convertir temperaturas"""
        # Convertir temperaturas
        self.data['Temperatura del Aire [°C]'] = self.kelvin_to_celsius(
            self.data['Air temperature [K]']
        )
        self.data['Temperatura del Proceso [°C]'] = self.kelvin_to_celsius(
            self.data['Process temperature [K]']
        )
        
        # Traducir etiquetas de fallas
        self.data['Tipo de Falla'] = self.data['Failure Type'].map(self.FAILURE_TRANSLATIONS)
        
        # Renombrar columnas restantes
        self.data['Nivel de Vibración'] = self.data['Vibration Levels']
        self.data['Horas de Operación'] = self.data['Operational Hours']
        
        # Eliminar columnas originales en inglés
        self.data = self.data[[
            'Temperatura del Aire [°C]',
            'Temperatura del Proceso [°C]',
            'Nivel de Vibración',
            'Horas de Operación',
            'Tipo de Falla'
        ]]
        
        return self.data
    
    def _get_failure_statistics(self, failure_type_spanish):
        """Calcular estadísticas para un tipo específico de falla"""
        failure_data = self.data[self.data['Tipo de Falla'] == failure_type_spanish]
        stats = {}
        
        for column in ['Temperatura del Aire [°C]', 'Temperatura del Proceso [°C]', 'Nivel de Vibración']:
            stats[column] = {
                'mean': failure_data[column].mean(),
                'std': failure_data[column].std()
            }
        return stats
    
    def generate_synthetic_failure(self, failure_type_spanish, n_samples=1):
        """Generar datos sintéticos de falla basados en la distribución original"""
        stats = self._get_failure_statistics(failure_type_spanish)
        synthetic_data = []
        
        typical_hours = self.data[self.data['Tipo de Falla'] == failure_type_spanish]['Horas de Operación'].mean()
        
        for _ in range(n_samples):
            sample = {
                'Temperatura del Aire [°C]': np.random.normal(
                    stats['Temperatura del Aire [°C]']['mean'],
                    stats['Temperatura del Aire [°C]']['std']
                ),
                'Temperatura del Proceso [°C]': np.random.normal(
                    stats['Temperatura del Proceso [°C]']['mean'],
                    stats['Temperatura del Proceso [°C]']['std']
                ),
                'Nivel de Vibración': np.random.normal(
                    stats['Nivel de Vibración']['mean'],
                    stats['Nivel de Vibración']['std']
                ),
                'Horas de Operación': typical_hours + np.random.normal(0, 5),
                'Tipo de Falla': failure_type_spanish
            }
            synthetic_data.append(sample)
            
        return pd.DataFrame(synthetic_data)
    
    def generate_failure_sequence(self, failure_type_spanish, sequence_length=10):
        """Generar una secuencia que lleva a una falla"""
        end_stats = self._get_failure_statistics(failure_type_spanish)
        normal_stats = self._get_failure_statistics('Sin Falla')
        
        sequence_data = []
        
        for i in range(sequence_length):
            progress = i / sequence_length
            
            sample = {}
            for column in ['Temperatura del Aire [°C]', 'Temperatura del Proceso [°C]', 'Nivel de Vibración']:
                mean = normal_stats[column]['mean'] * (1 - progress) + end_stats[column]['mean'] * progress
                std = normal_stats[column]['std']
                sample[column] = np.random.normal(mean, std)
            
            sample['Horas de Operación'] = self.data['Horas de Operación'].mean() + i
            sample['Tipo de Falla'] = 'Sin Falla' if i < sequence_length-1 else failure_type_spanish
            sequence_data.append(sample)
            
        return pd.DataFrame(sequence_data)
    
    def augment_dataset(self, target_ratio=0.3):
        """Aumentar dataset para alcanzar la proporción objetivo de fallas"""
        current_failures = self.data[self.data['Tipo de Falla'] != 'Sin Falla'].shape[0]
        total_samples = self.data.shape[0]
        target_failures = int(total_samples * target_ratio)
        samples_to_generate = target_failures - current_failures
        
        if samples_to_generate <= 0:
            return self.data
        
        failure_types = self.data['Tipo de Falla'].unique()
        failure_types = [f for f in failure_types if f != 'Sin Falla']
        
        augmented_data = [self.data]
        
        for failure_type in failure_types:
            n_samples = samples_to_generate // len(failure_types)
            new_failures = self.generate_synthetic_failure(failure_type, n_samples)
            new_sequences = pd.concat([
                self.generate_failure_sequence(failure_type) 
                for _ in range(n_samples // 2)
            ])
            
            augmented_data.extend([new_failures, new_sequences])
            
        return pd.concat(augmented_data, ignore_index=True)

def process_and_augment_data(input_file, output_file, target_ratio=0.3):
    """
    Función principal para procesar y aumentar los datos
    
    Parámetros:
    input_file: str - Ruta al archivo CSV de entrada
    output_file: str - Ruta donde guardar el nuevo CSV
    target_ratio: float - Proporción deseada de fallas en el dataset final
    """
    # Cargar datos
    print("Cargando datos originales...")
    data = pd.read_csv(input_file)
    
    # Crear aumentador
    augmenter = IndustrialDataAugmenter(data)
    
    # Convertir unidades y traducir
    print("Convirtiendo unidades y traduciendo etiquetas...")
    processed_data = augmenter.translate_and_convert()
    
    # Aumentar dataset
    print("Generando datos sintéticos...")
    augmented_data = augmenter.augment_dataset(target_ratio)
    
    # Guardar resultados
    print(f"Guardando resultados en {output_file}...")
    augmented_data.to_csv(output_file, index=False)
    
    print("\nEstadísticas del dataset:")
    print(f"Tamaño original: {len(data)}")
    print(f"Tamaño aumentado: {len(augmented_data)}")
    print("\nDistribución de fallas:")
    print(augmented_data['Tipo de Falla'].value_counts(normalize=True))

# Ejemplo de uso
if __name__ == "__main__":
    process_and_augment_data(
        input_file='Dataset_clean.csv',
        output_file='dataset_aumentado_es.csv',
        target_ratio=0.3
    )
