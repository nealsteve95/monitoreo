import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

# Definimos la frecuencia de rotación en Hz para el motor (1800 RPM)
frecuencia_hz = 30

# Cargar el archivo con el delimitador correcto
ruta_archivo_csv = '/Users/prom1/Desktop/Algoritmo-AMs-gratis/algoritmos/monitoreo/acelerationtotal.csv'
df = pd.read_csv(ruta_archivo_csv, sep=";")

# Mantener solo las columnas relevantes
df = df.drop(columns=['spot_id', 'spot_dyid', 'spot_name', 'spot_type', 'spot_rpm', 
                      'spot_power', 'spot_model', 'machine_id', 'machine_name', 
                      'battery_level', 'telemetry_interval', 'interval_unit', 
                      'dynamic_range_in_g', 'data_source', 'spot_path'])

# Función para convertir de aceleración (in/s²) a velocidad (mm/s)
def convertir_a_velocidad(df, frecuencia_hz):
    df['value_mm_s'] = (df['value'] * 25.4) / (2 * math.pi * frecuencia_hz)
    return df

# Aplicar la conversión
df = convertir_a_velocidad(df, frecuencia_hz)

# Función para calcular RMS en los ejes x, y, z
def calcular_rms(df):
    df_xyz = df[df['axis'].isin(['x', 'y', 'z'])]
    df_xyz = df_xyz.groupby('timestamp').filter(lambda x: len(x) == 3)

    rms_values = []

    for timestamp, group in df_xyz.groupby('timestamp'):
        x_value = group.loc[group['axis'] == 'x', 'value_mm_s'].values[0]
        y_value = group.loc[group['axis'] == 'y', 'value_mm_s'].values[0]
        z_value = group.loc[group['axis'] == 'z', 'value_mm_s'].values[0]

        # Calcular RMS
        rms = math.sqrt(x_value**2 + y_value**2 + z_value**2) / math.sqrt(3)
        rms_values.append({'timestamp': timestamp, 'Rms': rms})

    # Crear nuevo DataFrame para RMS y unirlo al original
    df_rms = pd.DataFrame(rms_values)
    df = pd.merge(df, df_rms, on='timestamp', how='left')

    return df

# Función para clasificar el estado global RMS según ISO 10816
def clasificar_rms_global(rms_value):
    if rms_value < 0.28:
        return 'Normal'
    elif rms_value < 0.45:
        return 'Alerta'
    else:
        return 'Crítico'


# Aplicar el cálculo de RMS en velocidad
df = calcular_rms(df)

# Calcular RMS global
rms_global = df['Rms'].mean()

# Umbral para detectar picos anormales: el doble del RMS global
threshold_abnormal_peak = 2 * rms_global
df['Anomalous'] = df['Rms'] > threshold_abnormal_peak  # Identificar picos anormales

# Gráfico de RMS con picos anormales resaltados
plt.figure(figsize=(12, 6))
timestamps = pd.to_datetime(df['timestamp'])
plt.plot(timestamps, df['Rms'], label='RMS (mm/s)')
plt.scatter(timestamps[df['Anomalous']], df['Rms'][df['Anomalous']], color='red', label='Picos Anormales')
plt.axhline(rms_global, color='green', linestyle='--', label=f'RMS Global: {rms_global:.2f} mm/s')
plt.xlabel('Tiempo')
plt.ylabel('RMS (mm/s)')
plt.title('Evolución del RMS con Picos Anormales')
plt.legend()
plt.grid()
plt.show()

# Preparar datos para el espectro de frecuencia (FFT)
df_fft = df.dropna(subset=['Rms'])  # Eliminar valores nulos en RMS
sampling_rate = 1 / (10 * 60)  # Suponiendo 10 minutos entre muestras, en Hz
rms_values = df_fft['Rms'].values
n = len(rms_values)
frequencies = np.fft.fftfreq(n, d=sampling_rate)
fft_values = np.fft.fft(rms_values)
fft_magnitude = np.abs(fft_values)[:n//2]  # Magnitud de la FFT

# Identificar picos en el espectro que superan el umbral
anomalous_fft = fft_magnitude > threshold_abnormal_peak

# Gráfico de espectro de frecuencia con picos anormales resaltados
plt.figure(figsize=(12, 6))
plt.plot(frequencies[:n//2], fft_magnitude, label=f'Espectro de Frecuencia\nRMS Global: {rms_global:.2f} mm/s')
plt.scatter(frequencies[:n//2][anomalous_fft], fft_magnitude[anomalous_fft], color='red', label='Picos Anormales')
plt.xlabel('Frecuencia (Hz)')
plt.ylabel('Amplitud')
plt.title('Espectro de Frecuencia (FFT) del RMS con Picos Anormales')
plt.legend()
plt.grid()
plt.show()

# Contar la cantidad de picos anormales
cantidad_picos_anormales = df['Anomalous'].sum()

# Salida de resultados
print(f"Cantidad de picos de vibración anormales encontrados: {cantidad_picos_anormales}")
print(f"Valor global RMS calculado del dataset: {rms_global} ({clasificar_rms_global(rms_global)})")
print("Ejemplos de picos anormales detectados:")
print(df[df['Anomalous']][['timestamp', 'value', 'value_mm_s', 'Rms']].head())
