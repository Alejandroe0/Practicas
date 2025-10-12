import glob
import numpy as np
from scipy import interpolate

def check_consistency(a):
    t=a[:,0]; y=a[:,2]; vy=a[:,4]; ay=a[:,6]
    mask = ~np.isnan(t) & ~np.isnan(y)
    t=t[mask]; y=y[mask]; vy=vy[mask]; ay=ay[mask]
    if len(t) < 5: return None
    # derivada numérica central
    dt = np.diff(t)
    if np.any(dt==0): return None
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
for p in files[:10]:
    a = read_file(p)
    res = check_consistency(a)
    print(p, res)
