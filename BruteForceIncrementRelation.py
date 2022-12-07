# --------------------------------------------------------
# KNN -BF Relación del incremento del parámetro k
# --------------------------------------------------------
path_1000csv = "./Test/1000.csv"
path_10000csv = "./Test/10000.csv"
path_20000csv = "./Test/20000.csv"

df = pd.read_csv("1000.csv", header=None)
d_t = list(df.itertuples(index=False, name=None))
d_t.sort()

# Creating 2 classes for data train
c_t = []
print(f"{(data_train[len(data_train) - 1][0] - data_train[0][0]) / 2}")
o = (data_train[len(data_train) - 1][0] - data_train[0][0]) / 2
for i in range(len(data_train)):
    if data_train[i][0] <= o:
        c_t.append(0)
    else:
        c_t.append(1)

# Gráficar los puntos del data train
fig_02 = plt.figure(figsize=(10, 10))

sp2_1 = fig_02.add_subplot(projection='3d')
sp2_1.set_xlabel('X')
sp2_1.set_ylabel('Y')
sp2_1.set_zlabel('Z')

x = []
y = []
z = []
for p in d_t:
    x.append(p[0])
    y.append(p[1])
    z.append(p[2])

sp2_1.scatter(x, y, z, c=class_train)


"""
A partir de un conjunto de 5 puntos aleatorios
se hará la búsqueda de los  k-vecinos  
"""
n_datos = 5

p_queries = [(412, 777, 96), (199, 299, 980), (239, 228, 263), (882, 543, 874), (913, 591, 936)]
p_queries.sort()
print(f"Puntos de consulta: \t{p_queries}")

# Gráfico del data train
x2_2 =[]
y2_2 =[]
z2_2 =[]

for p in p_queries:
    x2_2.append(p[0])
    y2_2.append(p[1])
    z2_2.append(p[2])

sp2_1.scatter(x2_2, y2_2, z2_2, c="#f00")
fig_02.savefig('./fig_02.png')
fig_02.show()

# Calcular los k vecinos más cercanos
_k = 100
time_execution = []
for _i in range(1, _k+1):
    t1 = KNN_BF(d_t, c_t, _i)
    dist_index_1 = []

    t_ini = time.time()
    for q in p_queries:
        temp = t1.query_neighbors(q)
        dist_index_1.append(temp)
    t_end = time.time()

    k_points_1 = [None] * n_datos
    for i in range(len(dist_index_1)):  # n_datos = len(dist_index_1)
        _p = dist_index_1[i]            # _p : array con k elementos
        neigh = []
        for k in range(_i):
            idx= _p[k][1]
            neigh.append(t1.data_train[idx])
            # print(neigh)
        k_points_1[i] = neigh

    elapsed = t_end - t_ini
    time_execution.append(elapsed*1000)
print(f"Time execution: {time_execution}")


# Graph k por tiempo de ejecución
x3 = [*range(1, _k+1) ]
print(f"K: {x3}")
fig_03 = plt.figure(figsize=(10, 10))

# plt.plot(x3, time_execution)
# plt.show()
sp3_1 = fig_03.add_subplot()
sp3_1.set_xlabel('k-vecinos más cercanos')
sp3_1.set_ylabel('Tiempo de ejecución (ms)')

sp3_1.plot(x3, time_execution)
# fig_03.savefig('./fig_03.png')
fig_03.show()

"""
Tiempo de ejecución para predecir la clase de los puntos por k-vecinos
"""

# Calcular los k vecinos más cercanos
_k = 100
time_predective = []
for _i in range(1, _k+1):
    t1 = KNN_BF(d_t, c_t, _i)
    dist_index_1 = []

    t1.set_data_test(p_queries)

    t_ini = time.time()
    predected_class = t1.predict();
    t_end = time.time()

    elapsed = t_end - t_ini
    time_predective.append(elapsed*1000)
    # print(f"{_i}-NN de los {n_datos} consultados:\t{k_points_1}")
    # print(f"Tiempo de ejecución para k = {_i} = {elapsed * 1000}")
print(f"Time predicción: {time_predective}")

# Graph k por tiempo de ejecución
x4 = [*range(1, _k+1) ]
print(f"K: {x4}")
fig_04 = plt.figure(figsize=(10, 10))

sp4_1 = fig_04.add_subplot()
sp4_1.set_xlabel('k-vecinos más cercanos')
sp4_1.set_ylabel('Tiempo de Predicción (ms)')

sp4_1.plot(x4, time_predective)
fig_04.savefig('./fig_04.png')
fig_04.show()