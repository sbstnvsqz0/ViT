from os.path import isfile, join
import os

def contar_clase(d,n_class,path):
    index = 0 if n_class == 6 else 1 if n_class==20 else 2
    paths = {}
    for i in range(n_class):
        paths[i]=0
        
    for k in d.keys():
        clase = d[k][index]
        aux = [path+"/"+k+"/"+f for f in os.listdir(path+"/"+k) if isfile(join(path+"/"+k,f))]
        paths[clase]+=len(aux)
    return paths