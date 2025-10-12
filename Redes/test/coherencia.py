import glob
import numpy as np
from scipy import interpolate

def check_consistency(a):
    t=a[:,0]; y=a[:,2]; vy=a[:,4]; ay=a[:,6]
    
    # Si se elimina un punto temporal en t o y debido a NaN, eliminar el mismo 
    # punto temporal en vy y ay para que todos los arrays tengan la misma longitud 
    # y estén sincronizados en el tiempo.
    mask = ~np.isnan(t) & ~np.isnan(y)
    t=t[mask]; y=y[mask]; vy=vy[mask]; ay=ay[mask]
    
    if len(t) < 5: return None
    # derivada numérica central (dt > 0)
    dt = np.diff(t)
    if np.any(dt==0): return None

    # derivadas numéricas
    dy_dt = np.gradient(y, t)
    d2y = np.gradient(dy_dt, t)
    
    # calcular RMSE entre measured vy/ay y derivadas
    vy_err = np.sqrt(np.nanmean((vy - dy_dt)**2))
    ay_err = np.sqrt(np.nanmean((ay - d2y)**2))
    return vy_err, ay_err

def read_file(p):
    arr=[]
    with open(p) as f:
        for line in f:
            line=line.strip()
            if not line or line.lower().startswith("t"):
                continue
            parts=[x for x in line.split(",") if x.strip()!='']
            try:
                vals=[float(x) for x in parts]
            except:
                continue
            if len(vals)<7:
                vals += [np.nan]*(7-len(vals))
            arr.append(vals[:7])
    return np.array(arr)

# test en los primeros archivos
files = sorted(glob.glob("/home/alejandro/Practicas/Toma_de_datos/datos/*.txt"))
for p in files[:50]:
    a = read_file(p)
    res = check_consistency(a)
    print('vy_err, y_err:', res)

############################## Datos completos ##############################

# def check_consistency(a):
#     # Verificar que el array sea bidimensional y tenga datos
#     if a.ndim != 2 or a.shape[0] == 0:
#         return None
    
#     # Verificar que tenga al menos 7 columnas
#     if a.shape[1] < 7:
#         return None
        
#     t=a[:,0]; y=a[:,2]; vy=a[:,4]; ay=a[:,6]
    
#     # Si se elimina un punto temporal en t o y debido a NaN, eliminar el mismo 
#     # punto temporal en vy y ay para que todos los arrays tengan la misma longitud 
#     # y estén sincronizados en el tiempo.
#     mask = ~np.isnan(t) & ~np.isnan(y)
#     t=t[mask]; y=y[mask]; vy=vy[mask]; ay=ay[mask]
    
#     if len(t) < 5: return None
#     # derivada numérica central (dt > 0)
#     dt = np.diff(t)
#     if np.any(dt==0): return None

#     # derivadas numéricas
#     dy_dt = np.gradient(y, t)
#     d2y = np.gradient(dy_dt, t)
    
#     # calcular RMSE entre measured vy/ay y derivadas
#     vy_err = np.sqrt(np.nanmean((vy - dy_dt)**2))
#     ay_err = np.sqrt(np.nanmean((ay - d2y)**2))
#     return vy_err, ay_err

# def read_file(p):
#     arr=[]
#     with open(p) as f:
#         for line in f:
#             line=line.strip()
#             if not line or line.lower().startswith("t"):
#                 continue
#             parts=[x for x in line.split(",") if x.strip()!='']
#             try:
#                 vals=[float(x) for x in parts]
#             except:
#                 continue
#             if len(vals)<7:
#                 vals += [np.nan]*(7-len(vals))
#             arr.append(vals[:7])
    
#     # Convertir a array numpy y verificar que no esté vacío
#     if not arr:
#         return np.array([])  # Array vacío
#     return np.array(arr)

# # Listas para almacenar todos los errores
# all_vy_errors = []
# all_ay_errors = []
# valid_files = []
# invalid_files = []

# # Procesar todos los archivos
# files = sorted(glob.glob("/home/alejandro/Practicas/Toma_de_datos/datos/*.txt"))

# print(f"Procesando {len(files)} archivos...\n")

# for i, p in enumerate(files):
#     a = read_file(p)
    
#     # Debug: mostrar forma del array
#     print(f"Archivo {i+1}: {p.split('/')[-1]} - Forma: {a.shape if a.size > 0 else 'Array vacío'}")
    
#     # Verificar si el array es válido
#     if a.size == 0 or a.ndim != 2 or a.shape[1] < 7:
#         invalid_files.append(p)
#         print(f"  -> Formato inválido o datos insuficientes")
#         continue
    
#     res = check_consistency(a)
    
#     if res is not None:
#         vy_err, ay_err = res
#         all_vy_errors.append(vy_err)
#         all_ay_errors.append(ay_err)
#         valid_files.append(p)
#         print(f"  -> VÁLIDO - vy_err: {vy_err:.6f}, ay_err: {ay_err:.6f}")
#     else:
#         invalid_files.append(p)
#         print(f"  -> INCONSISTENTE - Datos insuficientes o tiempos duplicados")

# # Calcular estadísticas de todos los errores
# if all_vy_errors:
#     print("\n" + "="*60)
#     print("ESTADÍSTICAS DE TODOS LOS ERRORES VÁLIDOS:")
#     print("="*60)
    
#     print(f"Total de archivos procesados: {len(files)}")
#     print(f"Archivos válidos para análisis: {len(valid_files)}")
#     print(f"Archivos inválidos/descartados: {len(invalid_files)}")
    
#     print(f"\nERROR DE VELOCIDAD (vy_err):")
#     print(f"  Media: {np.mean(all_vy_errors):.6f}")
#     print(f"  Desviación estándar: {np.std(all_vy_errors):.6f}")
#     print(f"  Mínimo: {np.min(all_vy_errors):.6f}")
#     print(f"  Máximo: {np.max(all_vy_errors):.6f}")
#     print(f"  Mediana: {np.median(all_vy_errors):.6f}")
    
#     print(f"\nERROR DE ACELERACIÓN (ay_err):")
#     print(f"  Media: {np.mean(all_ay_errors):.6f}")
#     print(f"  Desviación estándar: {np.std(all_ay_errors):.6f}")
#     print(f"  Mínimo: {np.min(all_ay_errors):.6f}")
#     print(f"  Máximo: {np.max(all_ay_errors):.6f}")
#     print(f"  Mediana: {np.median(all_ay_errors):.6f}")
    
#     # Mostrar archivos con mejores y peores resultados
#     if len(valid_files) > 0:
#         best_vy_idx = np.argmin(all_vy_errors)
#         worst_vy_idx = np.argmax(all_vy_errors)
        
#         print(f"\nMEJOR archivo (menor error de velocidad):")
#         print(f"  {valid_files[best_vy_idx].split('/')[-1]}")
#         print(f"  vy_err: {all_vy_errors[best_vy_idx]:.6f}")
        
#         print(f"\nPEOR archivo (mayor error de velocidad):")
#         print(f"  {valid_files[worst_vy_idx].split('/')[-1]}")
#         print(f"  vy_err: {all_vy_errors[worst_vy_idx]:.6f}")

# else:
#     print("\nNo se encontraron archivos válidos para análisis.")

# # Mostrar algunos archivos inválidos para debugging
# if invalid_files:
#     print(f"\nPrimeros 5 archivos inválidos:")
#     for i, p in enumerate(invalid_files[:5]):
#         print(f"  {i+1}. {p.split('/')[-1]}")