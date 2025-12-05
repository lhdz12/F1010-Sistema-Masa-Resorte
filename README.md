# Modelación de un Sistema Masa-Resorte
---
Autores: 
- Laurie C. Hernández P.
- Emilio A. González H.
- Alejandro Mattar A.
- H. Fernando Piñón S.

---
## Descripción del Proyecto

Este proyecto implementa la simulación de un sistema de dos masas acopladas mediante resortes, resolviendo numéricamente las ecuaciones diferenciales mediante el método de **Runge–Kutta** (RK45) y comparando los resultados con un **modelo analítico aproximado** mediante ajustes de funciones armónicas utilizando `curve_fit`.

La simulación permite analizar:

- La evolución temporal de las posiciones de ambas masas.  
- Las velocidades obtenidas numéricamente vs. las derivadas del modelo ajustado.  
- El comportamiento acoplado y la transferencia de energía entre masas.

---

## Archivo principal: `simulacionPosicion.py`

Incluye:

- Definición del sistema de ecuaciones diferenciales.
- Implementación del método de Runge–Kutta.
- Ajuste de modelos cosenoidales.
- Obtención de posiciones y velocidades ajustadas.
- Generación automática de gráficas comparativas.

---

## Resultados Visuales

### Comparación de posiciones
**x₁(t): RK vs Ajuste**
![](A003RKX1.png)

**x₂(t): RK vs Ajuste**
![](A004RKX2.png)

---

### Comparación de velocidades
**v₁(t): RK vs Ajuste**
![](A005RKV1.png)

**v₂(t): RK vs Ajuste**
![](A006RKV2.png)

---

## Conclusiones

### Reflexión sobre la eficiencia energética
El análisis del sistema masa-resorte permitió observar cómo la energía oscila entre ambas masas de manera periódica. La simulación numérica confirmó que, en ausencia de amortiguamiento, la energía total del sistema se conserva, lo cual es coherente con un sistema mecánico ideal.  
Comparar el método numérico con el ajuste armónico mostró que los métodos de integración como RK45 mantienen la estabilidad energética adecuadamente para tiempos moderados, sin introducir errores acumulativos significativos.

### Aprendizajes clave del proyecto
- El método de Runge–Kutta es altamente confiable para resolver sistemas acoplados sin necesidad de simplificación.
- Los ajustes cosenoidales permiten obtener parámetros físicos como frecuencia y amplitud de forma precisa.
- La comparación entre datos numéricos y analíticos ayuda a validar modelos y a identificar discrepancias por no linealidades o condiciones iniciales particulares.
- Visualizar tanto posición como velocidad ofrece una interpretación más completa del comportamiento del sistema.

### Mejoras y aplicaciones futuras
- Añadir **amortiguamiento** para modelar sistemas reales con disipación de energía.
- Extender el modelo a más masas (n cuerpos) para simular cadenas vibracionales.

---

## Requisitos para ejecutar
- `python 3.11`
- `numpy`
- `scipy`
- `matplotlib`




## 📂 Archivos incluidos

