# Localización de puntas de grieta mediante Deep Learning

## Descripción del proyecto

Este proyecto utiliza técnicas de **Deep Learning** para localizar la posición de la punta de una grieta (*crack tip*) a partir de campos de desplazamiento mecánico.

El modelo recibe como entrada dos campos de desplazamiento:

- $U_x$: desplazamiento horizontal.
- $U_y$: desplazamiento vertical.

A partir de esta información, una red neuronal convolucional (**CNN**) debe predecir las coordenadas:

```text
(x, y)
```

correspondientes a la posición de la punta de la grieta.

El problema se aborda como una tarea de **regresión bidimensional**.

---

# Dataset

El proyecto utiliza el dataset **CrackMNIST**, compuesto por simulaciones de campos de desplazamiento asociados a procesos de propagación de grietas.

Cada muestra contiene:

```text
(Ux, Uy), máscara
```

donde:

- `Ux` representa el campo de desplazamiento horizontal.
- `Uy` representa el campo de desplazamiento vertical.
- `máscara` contiene la información utilizada para determinar la posición de la punta de la grieta.

Las coordenadas reales de la punta de la grieta se obtienen a partir de la posición del píxel correspondiente en la máscara.

El dataset se divide en distintos experimentos, utilizados para:

- Entrenamiento.
- Validación.
- Evaluación final.

Esto permite evaluar la capacidad del modelo para generalizar a configuraciones no utilizadas durante el entrenamiento.

---

# Modelo

Se utiliza una **Red Neuronal Convolucional (CNN)** para extraer información espacial de los campos de desplazamiento.

La entrada del modelo tiene la forma:

```text
(2, H, W)
```

donde los dos canales corresponden a:

```text
Canal 1 → Ux
Canal 2 → Uy
```

La salida final de la red consiste en dos valores continuos:

```text
(x_pred, y_pred)
```

que representan la posición predicha de la punta de la grieta.

---

# Entrenamiento

El problema se trata como una regresión de coordenadas.

Durante el entrenamiento, el modelo aprende a minimizar la diferencia entre las coordenadas reales y las coordenadas predichas.

Para evaluar el error espacial se utiliza la distancia euclídea:

$$
d = \sqrt{(x_{pred} - x_{real})^2 + (y_{pred} - y_{real})^2}
$$

Esta métrica representa directamente la distancia, en píxeles, entre la posición predicha y la posición real de la punta de la grieta.

---

# Métricas de evaluación

Para analizar el rendimiento del modelo se calculan diferentes métricas.

## Error absoluto medio

Mide el error promedio entre las coordenadas predichas y las reales.

## RMSE

El **Root Mean Squared Error** penaliza más fuertemente los errores grandes.

## Error euclídeo medio

Representa la distancia promedio entre la punta de grieta real y la predicha.

## Error mediano

Permite analizar el error típico reduciendo la influencia de valores extremos.

## Precisión dentro de un determinado número de píxeles

También se calcula el porcentaje de predicciones que cumplen:

- Error ≤ 1 píxel.
- Error ≤ 2 píxeles.
- Error ≤ 5 píxeles.

Estas métricas permiten interpretar de manera más intuitiva la precisión espacial del modelo.

---

# Resultados

Los resultados obtenidos muestran que el modelo es capaz de localizar correctamente la punta de la grieta en la gran mayoría de las muestras.

En los experimentos realizados se obtuvieron aproximadamente los siguientes resultados:

| Métrica | Resultado |
|---|---:|
| Error medio | ~0.6 píxeles |
| Error mediano | 0 píxeles |
| Predicciones con error ≤ 1 píxel | ~89% |
| Predicciones con error ≤ 2 píxeles | ~99% |
| Predicciones con error ≤ 5 píxeles | >99% |

Sin embargo, existen algunas muestras con errores significativamente mayores.

Estos casos son particularmente interesantes, ya que podrían estar relacionados con:

- Configuraciones difíciles para el modelo.
- Efectos de borde.
- Ambigüedades en los campos de desplazamiento.
- Muestras sin un píxel positivo en la máscara.
- Posibles inconsistencias entre la visualización de los datos y las etiquetas almacenadas.

Por este motivo, además de las métricas globales, se realiza un análisis específico de las predicciones con mayor error.

---

# Análisis de errores

Además de calcular métricas globales, el proyecto analiza cómo se distribuyen espacialmente los errores del modelo.

Se generan diferentes gráficos para estudiar su comportamiento.

## Distribución del error

Se representa un histograma de los errores de localización.

Esto permite observar si:

- La mayoría de las predicciones presentan errores pequeños.
- Existen errores extremos.
- La distribución presenta una cola asociada a casos problemáticos.

## Distribución acumulada

Se calcula la distribución acumulada del error para visualizar qué porcentaje de las predicciones se encuentra por debajo de una determinada distancia.

Por ejemplo:

```text
¿Qué porcentaje de las predicciones tiene un error menor a 1 píxel?
```

## Errores en las coordenadas

Se analizan por separado:

$$
\Delta x = x_{pred} - x_{real}
$$

$$
\Delta y = y_{pred} - y_{real}
$$

Esto permite detectar posibles sesgos sistemáticos en alguna dirección.

## Mapa espacial del error

Se analiza el error del modelo en función de la posición real de la punta de la grieta.

Este análisis permite investigar si existen regiones de la imagen donde el modelo presenta un rendimiento peor.

Por ejemplo, puede revelar si las predicciones empeoran:

- Cerca de los bordes.
- En determinadas posiciones de propagación.
- En configuraciones poco frecuentes dentro del dataset.

## Mejores y peores predicciones

También se visualizan automáticamente las muestras con:

- Menor error.
- Mayor error.

Para cada muestra se comparan:

- Los campos de desplazamiento.
- La posición real.
- La posición predicha.
- El error de localización.

Este análisis es especialmente útil para investigar posibles errores de etiquetado o problemas de visualización.

---

# Experimentos con diferentes resoluciones

Además de trabajar con la resolución original, el proyecto permite realizar experimentos con imágenes de diferentes tamaños:

```text
28 × 28
64 × 64
128 × 128
```

Esto permite estudiar cómo afecta la resolución espacial al rendimiento del modelo.

El objetivo es analizar la relación entre:

- Precisión en la localización.
- Resolución de los campos.
- Costo computacional.
- Capacidad del modelo para identificar patrones locales alrededor de la grieta.

---

# Estructura del proyecto

Una posible organización del repositorio es:

```text
CrackMNIST-Crack-Tip-Localization/
│
├── exp2.ipynb
├── README.md
├── requirements.txt
│
├── models/
│   ├── modelo_28x28.pth
│   ├── modelo_64x64.pth
│   └── modelo_128x128.pth
│
└── resultados/
    ├── entrenamiento.png
    ├── histograma_error.png
    ├── distribucion_acumulada.png
    ├── mapa_error_espacial.png
    ├── errores_coordenadas.png
    ├── predicciones_vs_reales.png
    └── peores_predicciones.png
```

---

# Instalación

Primero se debe clonar el repositorio:

```bash
git clone https://github.com/TU_USUARIO/CrackMNIST-Crack-Tip-Localization.git
```

Luego ingresar a la carpeta:

```bash
cd CrackMNIST-Crack-Tip-Localization
```

Instalar las dependencias:

```bash
pip install -r requirements.txt
```

---

# Librerías utilizadas

Las principales librerías utilizadas son:

```text
numpy
matplotlib
pandas
torch
torchvision
scikit-learn
h5py
```

---

# Uso

El flujo completo del proyecto se encuentra en el notebook:

```text
exp2.ipynb
```

El notebook incluye:

1. Carga del dataset CrackMNIST.
2. Exploración de los campos de desplazamiento.
3. Obtención de las coordenadas de la punta de la grieta.
4. Preprocesamiento de los datos.
5. Definición del modelo CNN.
6. Entrenamiento.
7. Validación.
8. Evaluación del modelo.
9. Cálculo de métricas.
10. Análisis estadístico de los errores.
11. Visualización de las mejores y peores predicciones.

---

# Conclusiones

Los resultados muestran que los campos de desplazamiento contienen información suficiente para localizar con alta precisión la posición de la punta de una grieta.

El modelo obtiene errores muy bajos para la mayoría de las muestras, alcanzando una precisión inferior a un píxel en una gran proporción del conjunto evaluado.

Sin embargo, la existencia de algunos errores grandes muestra que las métricas promedio no son suficientes para caracterizar completamente el comportamiento del modelo.

Por este motivo, el proyecto incluye un análisis detallado de los errores y de las predicciones individuales.

Este análisis permite diferenciar entre posibles:

- Fallos reales del modelo.
- Configuraciones físicamente más difíciles.
- Efectos asociados a los límites de la imagen.
- Problemas o inconsistencias en las etiquetas del dataset.

---

# Posibles mejoras futuras

Algunas posibles extensiones del proyecto son:

- Comparar diferentes arquitecturas de redes neuronales.
- Entrenar modelos específicamente optimizados para distintas resoluciones.
- Utilizar técnicas de *data augmentation*.
- Implementar localización mediante mapas de calor en lugar de regresión directa.
- Estimar la incertidumbre de las predicciones.
- Incorporar restricciones físicas durante el entrenamiento.
- Comparar el modelo con métodos tradicionales de procesamiento de imágenes.
- Investigar automáticamente muestras potencialmente mal etiquetadas.
- Analizar la capacidad de generalización entre diferentes experimentos de propagación de grietas.

---
