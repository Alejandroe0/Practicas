import pandas as pd
import cv2
import os

def timecode_to_seconds(timecode, fps):
    """
    Convierte un timecode en formato 'HH:MM:SS:FF' a segundos

    Args:
        timecode (str): Cadena con el timecode en formato 'HH:MM:SS:FF', donde:
                        HH = horas, MM = minutos, SS = segundos, FF = fotogramas.
        fps (int o float): Número de fotogramas por segundo (frames per second).

    Returns:
        float: Tiempo total en segundos, incluyendo la parte fraccional de los fotogramas
    """
    
    
    hh, mm, ss, ff = map(int, timecode.split(':'))
    return hh * 3600 + mm * 60 + ss + ff / fps

def cortar_video(video_path, csv_path, output_folder, output_prefix, clip_start_index):
    """
    Corta un video en múltiples clips basándose en un archivo CSV con timecodes.

    Args:
        video_path (str): Ruta al archivo de video de entrada.
        csv_path (str): Ruta al archivo CSV que contiene los timecodes.
                        La primera fila debe contener el fps, y las siguientes los timecodes 
                        en formato HH:MM:SS:FF.
                        Los timecodes deben estar en pares (inicio, fin).
        output_folder (str): Carpeta donde se guardarán los clips generados.
        output_prefix (str): Prefijo para nombrar los archivos de salida.
        clip_start_index (int): Índice inicial para numerar los clips exportados.

    Returns:
        int: El siguiente índice disponible después de cortar los clips, útil para continuar
        sin sobrescribir.
    """
    
    df = pd.read_csv(csv_path, header=None)
    fps = float(df.iloc[0, 0])  # Primera fila, primer valor
    tiempos = df.iloc[1:].values.flatten()
    
    if len(tiempos) % 2 != 0:
        raise ValueError(f"CSV inválido: {csv_path}")
    
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    for i in range(0, len(tiempos), 2):
        start_time = timecode_to_seconds(tiempos[i], fps)
        end_time = timecode_to_seconds(tiempos[i+1], fps)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        output_filename = os.path.join(output_folder, f"{output_prefix}_{clip_start_index:03d}.mp4")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

        video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_num in range(start_frame, end_frame):
            ret, frame = video.read()
            if not ret:
                break
            out.write(frame)
        out.release()
        print(f"Guardado: {output_filename}")
        clip_start_index += 1

    video.release()
    return clip_start_index

# Directorios
carpeta_clips = "/home/alejandro/Practicas/Toma_de_datos/clips"
carpeta_salida = "/home/alejandro/Practicas/Toma_de_datos/clips_indi"
os.makedirs(carpeta_salida, exist_ok=True)

# Inicializar índice global de clips
clip_index = 1
prefix = "clip"

# Recorrer todos los archivos mp4
for archivo in sorted(os.listdir(carpeta_clips)):
    if archivo.endswith(".mp4"):
        nombre_base = os.path.splitext(archivo)[0]
        video_path = os.path.join(carpeta_clips, archivo)
        csv_path = os.path.join(carpeta_clips, f"{nombre_base}.csv")
        
        if not os.path.exists(csv_path):
            print(f"No se encontró CSV para: {archivo}")
            continue
        
        clip_index = cortar_video(
            video_path, 
            csv_path, 
            output_folder=carpeta_salida, 
            output_prefix=prefix,
            clip_start_index=clip_index
        )


