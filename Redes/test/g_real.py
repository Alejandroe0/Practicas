import glob, numpy as np, pandas as pd

files = sorted(glob.glob("/home/alejandro/Practicas/Toma_de_datos/datos/*.txt"))

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

g_glob_list=[]
for p in files:
    a = read_file(p)
    if a.size==0: continue
    t = a[:,0]; y = a[:,2]
    mask = ~np.isnan(t) & ~np.isnan(y)
    t = t[mask]; y = y[mask]
    if len(t) < 5: continue
    T = np.vstack([np.ones_like(t), t, t**2]).T
    coef, *_ = np.linalg.lstsq(T, y, rcond=None)
    c2 = coef.flatten()[2]
    g_glob_list.append(2*c2)
print("Baseline global (media de g por archivo):", np.nanmean(g_glob_list))
print("Baseline global (mediana):", np.nanmedian(g_glob_list))
print("Baseline por archivo (primeros 10):", g_glob_list[:10])
