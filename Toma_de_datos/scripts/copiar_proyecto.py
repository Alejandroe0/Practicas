import shutil
import os

# Nombre del archivo original
archivo_base = "../tracker_proyects/clip_001.trk"

# Carpeta destino
clips_path = "../tracker_proyects/"

# Cantidad de clips
numero_final = 609  # Cantidad de clips

# Verifica que el archivo base exista
if not os.path.exists(archivo_base):
    print(f"El archivo '{archivo_base}' no existe en esta carpeta.")
else:
    for i in range(2, numero_final + 1):
        
        # Nombre de la copia
        nuevo_nombre = f"clip_{i:03d}.trk"
        
        # Construir el path de salida
        path_out = os.path.join(clips_path, nuevo_nombre)
        
        shutil.copyfile(archivo_base, path_out)
        print(f"Copiado: {archivo_base} → {nuevo_nombre}")
