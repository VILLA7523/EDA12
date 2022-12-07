# -------------------------------------------------------------------------
# Tiempo de ejecución para consultar 1-vecino en data sets 1000, 10000, 20000
# -------------------------------------------------------------------------
"""
KNN - BRUTE FORCE
"""
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
import time
import random

n_datos = 10

queries_1000 = [(11, 850, 664), (207, 639, 704), (262, 310, 509), (273, 30, 14), (490, 255, 570), (632, 732, 537), (691, 881, 689), (745, 489, 393), (843, 791, 817), (885, 915, 116)]
queries_10000 = [(893, 8906, 2355), (2565, 5805, 633), (3104, 781, 8376), (3760, 1562, 1626), (4514, 2679, 4655), (6454, 9895, 5888), (7198, 4225, 709), (7558, 8738, 6959), (7901, 2231, 7129), (8898, 2095, 3829)]
queries_20000 = [(2863, 12030, 8217) , (3268, 12927, 2714), (5037, 19169, 4791), (6117, 7670, 3173), (7495, 2897, 4857), (7507, 765, 19003), (11538, 6689, 2942), (13916, 16922, 13400), (16900, 11807, 7612), (19221, 1893, 8179)]
#queries_20000 = [queries_20000_test[4]]


#data
#3268, 12927, 2714), (5037, 19169, 4791), (6117, 7670, 3173), (7495, 2897, 4857), (7507, 765, 19003), (11538, 6689, 2942), (13916, 16922, 13400), (16900, 11807, 7612), (19221, 1893, 8179)

p_queries = queries_20000
print(f"Puntos de consulta: \t{p_queries}")


df = pd.read_csv("20000.csv", header=None)        #Update
start_time = time.time()
d_t = list(df.itertuples(index=False, name=None))
elapsed_time = time.time() - start_time
elapsed_time = elapsed_time * 1000
print("Elapsed time of insert: %0.10f ms." % elapsed_time)

d_t.sort()

# 2. Creating 2 classes for data train
c_t = []
o = (d_t[len(d_t) - 1][0] - d_t[0][0]) / 2
# print(o)
for i in range(len(d_t)):
    if d_t[i][0] <= o:
        c_t.append(0)
    else:
        c_t.append(1)

# 3. KNN BRUTE FORCE : Calcular los 1-vecinos más cercanos para cada punto de consulta
_k = 1
search_time = []
distancias = [None] * 10
pointt = [None] * 10

for _i in range(1, _k+1):
    t1 = KNN_BF(d_t, c_t, _i)
    dist_index_1 = []

    for q in p_queries:
        t_ini = time.time()
        temp = t1.query_neighbors(q)
        t_end = time.time()
        elapsed = t_end - t_ini
        search_time.append(elapsed*1000)
        dist_index_1.append(temp)
    time_dist = []

    for i in range(len(dist_index_1)):  # n_datos = len(dist_index_1)
        _p = dist_index_1[i]            # _p : array con k elementos
        for k in range(_i):
            d= _p[k][0]
            p = _p[k][2]
            time_dist.append((search_time[i],d,p))
    time_dist.sort()

    idx = 0
    for t, d, p in time_dist:
        distancias[idx] = d
        search_time[idx] = t
        pointt[idx] = p;
        idx+=1

# Gráficar los puntos del data train
#fig_05 = plt.figure(figsize=(10, 10))
x = search_time                         # tiempo ordenado de menor a mayor/
y = distancias        
z = pointt                  # distancia
# x = [58.22157859802246, 58.47573280334473, 58.47907066345215, 58.6700439453125, 59.60440635681152, 61.31124496459961, 63.601016998291016, 64.07737731933594, 64.44621086120605, 74.1276741027832]
# y = [386.48285861083156, 370.5792762689247, 111.57508682497182, 818.6562160027859, 374.20048102588004, 535.9645510665794, 348.85097104637674, 445.2381385281364, 439.50654147577825, 279.7659736279593]

print(f"Knn time: {x}")
print(f"Knn distance: {y}")
print(f"Knn points: {z}")

#fig_05 = plt.figure(figsize=(10, 10))

# sp5_1 = fig_05.add_subplot()
# sp5_1.set_xlabel('Tiempo de Búsqueda (ms)')
# sp5_1.set_ylabel('Distancia a punto ás cercano')

# sp5_1.plot(x, y)

# fig_05.savefig('./fig_05.png')
# fig_05.show()