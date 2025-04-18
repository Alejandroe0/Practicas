import pandas as pd
from moviepy.editor import VideoFileClip

def timecode_to_seconds(timecode, fps):
    """ Convierte un timecode HH:MM:SS:FF a segundos """
    hh, mm, ss, ff = map(int, timecode.split(':'))
    return hh * 3600 + mm * 60 + ss + ff / fps

def cortar_video(video_path, csv_path, output_prefix="clip"):
    # Leer el CSV
    tiempos = pd.read_csv(csv_path, header=None)
    
    # Obtener los tiempos (omitir la primera fila que es el número de fotogramas)
    tiempos = tiempos.iloc[1:].values.flatten()
    
    # Cargar el video para obtener el FPS
    video = VideoFileClip(video_path)
    fps = video.fps
    
    # Asegurar que hay pares de tiempos
    if len(tiempos) % 2 != 0:
        raise ValueError("El archivo CSV debe contener un número par de tiempos.")
    
    # Crear los clips
    for i in range(0, len(tiempos), 2):
        start_time = timecode_to_seconds(tiempos[i], fps)
        end_time = timecode_to_seconds(tiempos[i+1], fps)
        
        # Cortar el clip
        clip = video.subclip(start_time, end_time)
        output_filename = f"{output_prefix}_{i//2 + 1}.mp4"
        clip.write_videofile(output_filename, codec="libx264")
        print(f"Clip guardado: {output_filename}")
    
    # Cerrar el video
    video.close()

# Ejemplo de uso

path_video = '/home/alejandro/Practicas/Toma_de_datos/clips/0001-0021.mp4'
path_csv = '/home/alejandro/Practicas/Toma_de_datos/clips/0001-0021.csv'

cortar_video(path_video, path_csv)
