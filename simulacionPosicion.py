import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit # para poder encontrar una de mejor ajuste sinusoidal para los datos empíricos

# parámetros obtenidos de manera experimental. 
m1 = 0.050
m2 = 0.082
k1 = 15.519
k2 = 187.33

# cond. iniciales
x10 = -0.076    # m
x20 = -0.015   # m
# se encuentra en reposo
v10 = 0
v20 = 0

# definimos el sistema en primer orden.
def f(t, y):
    """
    y = [x1, v1, x2, v2]
    """
    x1, v1, x2, v2 = y

    dx1 = v1
    dv1 = (-2*k1*x1 - 2*k2*(x1 - x2)) / m1

    dx2 = v2
    dv2 = (-2*k2*(x2 - x1)) / m2

    return np.array([dx1, dv1, dx2, dv2])


# creamos una función d runge kutta 4to orden. retorna el sig. valor de y. 
def rk4_step(t, y, h):
    k1 = f(t, y)
    k2 = f(t + 0.5*h, y + 0.5*h*k1)
    k3 = f(t + 0.5*h, y + 0.5*h*k2)
    k4 = f(t + h, y + h*k3)

    return y + (h/6)*(k1 + 2*k2 + 2*k3 + k4)

T = 18.35     # segundos de sim
h = 0.001 # paso 
N = int(T/h)

t = np.linspace(0, T, N)
X1 = np.zeros(N)
V1 = np.zeros(N)
X2 = np.zeros(N)
V2 = np.zeros(N)
y = np.array([x10, v10, x20, v20])

for i in range(N):
    X1[i] = y[0]
    V1[i] = y[1]   
    X2[i] = y[2]
    V2[i] = y[3]   
    y = rk4_step(t[i], y, h)

w1 = 15.106
w2 = 111.63

def x1_analit(t):
    return -0.0390*np.cos(w1*t) - 0.0370 *np.cos(w2*t)

def x2_analit(t):
    return  -0.0479 *np.cos(w1*t) + 0.0270*np.cos(w2*t)

Xa1 = x1_analit(t)
Xa2 = x2_analit(t)

# vectores con los tiempos y desplazamientos empíricos.
import numpy as np

t_emp = np.array([
0,0.1666667,0.3333333,0.5,0.6666667,0.8333333,1,1.166667,1.333333,1.5,1.668333,1.835,
2.001667,2.168333,2.335,2.501667,2.668333,2.835,3.001667,3.168333,3.336667,3.503333,
3.67,3.836667,4.003333,4.17,4.336667,4.503333,4.67,4.836667,5.003333,5.171667,
5.338333,5.505,5.671667,5.838333,6.005,6.171667,6.338333,6.505,6.671667,6.84,
7.006667,7.173333,7.34,7.506667,7.673333,7.84,8.006667,8.173333,8.34,8.508333,
8.675,8.841667,9.008333,9.175,9.341667,9.508333,9.675,9.841667,10.00833,10.17667,
10.34333,10.51,10.67667,10.84333,11.01,11.17667,11.34333,11.51,11.67667,11.845,
12.01167,12.17833,12.345,12.51167,12.67833,12.845,13.01167,13.17833,13.345,
13.51333,13.68,13.84667,14.01333,14.18,14.34667,14.51333,14.68,14.84667,15.01333,
15.18167,15.34833,15.515,15.68167,15.84833,16.015,16.18167,16.34833,16.515,
16.68167,16.84833,17.01667,17.18333,17.35,17.51667,17.68333,17.85,18.01667,
18.18333,18.35
])

x1_empirica = np.array([
0.1721912,0.1001214,0.1548511,0.07411125,0.1266734,0.1049983,0.0925351,
0.133176,0.0703181,0.1407623,0.07898815,0.1250478,0.101747,0.1022889,
0.1185452,0.08982571,0.1255897,0.09416073,0.1147521,0.1055402,0.09795388,
0.1190871,0.09036759,0.1207128,0.09957952,0.1087914,0.1071658,0.09849576,
0.115294,0.09578637,0.1136683,0.1028308,0.1049983,0.1125846,0.1006633,
0.1142102,0.1006633,0.1125846,0.1066239,0.1071658,0.1087914,0.09849576,
0.1115008,0.1049983,0.1060821,0.1066239,0.1022889,0.1131265,0.1028308,
0.1125846,0.1033727,0.1066239,0.1098752,0.1033727,0.1136683,0.09957952,
0.1082496,0.1033727,0.1033727,0.1033727,0.1006633,0.1093333,0.1001214,
0.1104171,0.1039145,0.1039145,0.1071658,0.1006633,0.1087914,0.09687013,
0.1093333,0.1028308,0.1044564,0.1049983,0.1039145,0.1093333,0.101747,
0.1087914,0.1055402,0.1055402,0.1082496,0.1049983,0.1087914,0.1028308,
0.1087914,0.1044564,0.1077077,0.1071658,0.1066239,0.1071658,0.1028308,
0.1093333,0.1033727,0.1093333,0.1039145,0.1039145,0.1082496,0.1028308,
0.110959,0.1049983,0.1077077,0.1039145,0.1049983,0.1093333,0.1022889,
0.1087914,0.1044564,0.1077077,0.1044564,0.1071658,0.1071658
])

x2_empirica = np.array([
0.1031017,0.02778067,0.08467787,-0.004732016,0.05541645,0.03319945,
0.02398752,0.06137711,-0.005273894,0.07004716,0.00502179,0.05379082,
0.02615504,0.03103194,0.05704209,0.01423372,0.05053955,0.01640123,
0.04403701,0.03916011,0.02290377,0.04457889,0.01802686,0.05053955,
0.02886443,0.03374133,0.03319945,0.02561316,0.04566265,0.01911062,
0.0369926,0.02398752,0.03428321,0.04295326,0.02290377,0.04024387,
0.02398752,0.04349513,0.03590884,0.02507128,0.03645072,0.02723879,
0.0418695,0.03157382,0.0294063,0.03590884,0.02778067,0.04620452,
0.02507128,0.0369926,0.02778067,0.03590884,0.04078574,0.02886443,
0.03536696,0.02290377,0.03807635,0.02778067,0.02723879,0.03157382,
0.02290377,0.03590884,0.02182001,0.03265757,0.02507128,0.03049006,
0.03211569,0.02182001,0.03319945,0.02615504,0.03536696,0.02615504,
0.02778067,0.03103194,0.02507128,0.03374133,0.02723879,0.03211569,
0.02561316,0.03374133,0.03157382,0.02723879,0.0294063,0.02832255,
0.03916011,0.02561316,0.03536696,0.02723879,0.03428321,0.03374133,
0.02723879,0.03157382,0.02561316,0.03211569,0.03049006,0.03157382,
0.0294063,0.03049006,0.03482508,0.02723879,0.03049006,0.02832255,
0.03157382,0.03157382,0.03049006,0.03211569,0.02886443,0.03374133,
0.03211569,0.02886443,0.03157382
])

# para hacer el mejor ajuste sinusoidal. como la solución analítica resultó con coseno, utilizaremos la fase de coseno como base: 

def modelo_cos(t, A, w, phi, C):
    return A * np.cos(w * t + phi) + C

# usamos curve_fit para enocntrar la curva de mejor ajuste de x1 y x2
popt1, pcov1 = curve_fit(modelo_cos, t_emp, x1_empirica, p0=[0.1, 2*np.pi, 0, 0])
A1, w1_emp, phi1, C1 = popt1

popt2, pcov2 = curve_fit(modelo_cos, t_emp, x2_empirica, p0=[0.05, 2*np.pi, 0, 0])
A2, w2_emp, phi2, C2 = popt2

print(f"{A1} * cos({w1_emp}t{phi1})+{C1})")
print(f"{A2} * cos({w2_emp}t{phi2})+{C2})")

# graficamos las de mejor ajuste

t_fino = np.linspace(min(t_emp), max(t_emp), 1000) # suavizamos t
# metemos el modelo 
x1_fit = modelo_cos(t_fino, *popt1)
x2_fit = modelo_cos(t_fino, *popt2)

# para la velocidad, hacemos un modelo seno pues es la derivada del cos. 
def modelo_sin(t, A, w, phi, C): 
    return -A * w * np.sin(w*t + phi)

v1_fit = modelo_sin(t_fino, *popt1)
v2_fit = modelo_cos(t_fino, *popt2)

plt.figure(figsize=(10,4))
plt.scatter(t_emp, x1_empirica, color = "lightblue", label = f"Gráfico empírico de $x_1(t)$")
plt.plot(t_fino, x1_fit, "--", color = "blue", label="Ajuste coseno $x_1(t)$", linewidth=2)
plt.scatter(t_emp, x2_empirica, color = "violet", label = "Gráfico empírico $x_2(t)$")
plt.plot(t_fino, x2_fit, "--", color = "purple", label="Ajuste coseno $x_2(t)$", linewidth=2)
plt.xlabel(f"$t$ (s)")
plt.ylabel(f"Desplazamiento (m)")
plt.title(f"Comparación de posiciones empíricas para $x_1(t)$ y $x_2(t)$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# graficamos la velocidad. 

# Definimos las funciones
v1_emp = 8.7356e-3 * np.sin(6.1436 * t - 0.2328)
v2_emp = 1.0390e-2 * np.sin(6.4179 * t - 0.3976)

plt.figure(figsize = (10,5))
plt.plot(t, v1_emp, color = "blue", label=r"$v_1(t) = 8.7356\times 10^{-3}\,\sin(6.1436t - 0.2328)$")
plt.plot(t, V1, color = "purple",label=r"$v_2(t) = 1.0390\times 10^{-2}\,\sin(6.4179t - 0.3976)$")
plt.grid(True)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()


# Graficamos
plt.figure(figsize=(10,5))
plt.plot(t, v1_emp, color = "blue", label=r"$v_1(t) = 8.7356\times 10^{-3}\,\sin(6.1436t - 0.2328)$")
plt.plot(t, v2_emp, color = "purple",label=r"$v_2(t) = 1.0390\times 10^{-2}\,\sin(6.4179t - 0.3976)$")
plt.xlabel("t (s)")
plt.ylabel("Velocidad (m/s)")
plt.title(f"Funciones ajustadas de velocidad $v_1(t)$ y $v_2(t)$")
plt.grid(True)
plt.legend(loc='lower left')
plt.tight_layout()
plt.show()

# gráfica de scatters vs RK 
plt.figure(figsize= (10,4))
plt.scatter(t_emp, x1_empirica, color = "red", label = f"Gráfico empírico de $x_1(t)$")
plt.plot(t_emp, x1_empirica, color = "red")
plt.plot(t_fino, x1_fit, "--", color = "green", label=f"Ajuste coseno $x_1(t)$", linewidth=2)
plt.plot(t, X1, color = "blue", label=f'RK4 $x_1(t)$')
plt.xlim([-0.5, 7.5])
plt.xlabel("t (s)")
plt.ylabel(f"$x_1(t)$ (m)")
plt.title(f"Comparación para $x_1(t)$: RK4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# gráfica de scatters vs RK 
plt.figure(figsize= (10,4))
plt.scatter(t_emp, x2_empirica, color = "red", label = f"Gráfico empírico de $x_2(t)$")
plt.plot(t_emp, x2_empirica, color = "red")
plt.plot(t_fino, x2_fit, "--", color = "green", label=f"Ajuste coseno $x_2(t)$", linewidth=2)
plt.plot(t, X1, color = "blue", label=f'RK4 $x_2(t)$')
plt.xlim([-0.5, 7.5])
plt.xlabel("t (s)")
plt.ylabel(f"$x_2(t)$ (m)")
plt.title(f"Comparación para $x_2(t)$: RK4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# velocidad empírica curva de mejor ajuste y RK 
plt.figure(figsize= (10,4))
plt.plot(t_fino, v1_fit, "--", color = "green", label=f"Ajuste seno $v_1(t)$", linewidth=2)
plt.plot(t, V1, color = "blue", label=f'RK4 $v_1(t)$')
plt.xlim([-0.5, 3])
plt.xlabel("t (s)")
plt.ylabel(f"$v_1(t)$ (m/s)")
plt.title(f"Comparación para $v_1(t)$: RK4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize= (10,4))
plt.plot(t_fino, v1_fit, "--", color = "green", label=f"Ajuste seno $v_2(t)$", linewidth=2)
plt.plot(t, V1, color = "blue", label=f'RK4 $v_2(t)$')
plt.xlim([-0.5, 3])
plt.xlabel("t (s)")
plt.ylabel(f"$v_2(t)$ (m/s)")
plt.title(f"Comparación para $v_2(t)$: RK4")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


