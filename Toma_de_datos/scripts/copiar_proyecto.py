import os
import shutil

# Nombre base del archivo .trk original
archivo_base = "../tracker_proyects/clip_550.trk"

# Número final hasta donde quieres hacer las copias
numero_final = 609  # Ajusta este número según de videos

# Ruta relativa o nombre base del video (por ejemplo: video_001.mp4, video_002.mp4, ...)
nombre_video_base = "../clips_indi/clip"

# Verifica que el archivo base existe
if not os.path.exists(archivo_base):
    print(f"El archivo '{archivo_base}' no existe.")
else:
    with open(archivo_base, "r", encoding="utf-8") as f:
        contenido_base = f.read()

    for i in range(338, numero_final + 1):
        nuevo_nombre = f"../tracker_proyects/clip_{i:03d}.trk"
        nuevo_video = f"{nombre_video_base}_{i:03d}.mp4"

        # Reemplaza el nombre del video en el contenido
        # Se asume que el nombre original del video es clip_001.mp4
        contenido_modificado = contenido_base.replace("clip_550.mp4", nuevo_video)

        with open(nuevo_nombre, "w", encoding="utf-8") as f:
            f.write(contenido_modificado)

        print(f"{nuevo_nombre} creado con video '{nuevo_video}'")
