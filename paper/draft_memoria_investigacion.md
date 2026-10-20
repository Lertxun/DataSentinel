# DataSentinel: Estudio Experimental sobre la Robustez de Métodos de Detección de Anomalías en Series Temporales bajo Perturbaciones Controladas

**Autor:** Ander Lertxundi  
**Fecha:** Octubre 2026  
**Área:** Inteligencia Artificial, Detección de Anomalías, Series Temporales  

---

## Resumen (Abstract)
La detección de anomalías en datos secuenciales y temporales es fundamental para la monitorización de sistemas industriales, ciberseguridad y análisis empresarial. Sin embargo, la literatura reciente ha puesto de manifiesto limitaciones críticas en los procedimientos estándar de evaluación, como la inflación artificial del rendimiento derivada de métricas excesivamente permisivas (*point adjustment*) o la falta de análisis sobre la degradación de los modelos cuando cambian las distribuciones subyacentes. En este trabajo presentamos **DataSentinel Research**, un marco experimental riguroso diseñado para comparar la estabilidad y fiabilidad de métodos estadísticos de referencia (Rolling Z-Score, Rolling IQR) frente a modelos de aprendizaje automático no supervisado (Isolation Forest, Local Outlier Factor). Evaluamos los detectores sobre series temporales del benchmark de referencia UCR Anomaly Archive bajo perturbaciones controladas: inyección sistemática de ruido gaussiano aditivo y diferentes formas de *concept drift* (desplazamientos abruptos de media, alteraciones de varianza y tendencias lineales). Los resultados empíricos demuestran que, si bien algoritmos de ML como LOF logran una cobertura completa de las anomalías (Recall = 1.0), inducen un coste operativo severo con cientos de falsas alarmas, mientras que métodos estadísticos robustos como Rolling IQR exhiben una resiliencia significativamente superior ante perturbaciones de baja y media frecuencia.

---

## 1. Introducción
La monitorización en tiempo real de flujos de datos empresariales e industriales exige algoritmos capaces de identificar desviaciones significativas respecto al comportamiento normal. Tradicionalmente, la selección del mejor algoritmo se ha basado en métricas puntuales agregadas sobre conjuntos de datos estáticos y preprocesados. 

No obstante, en entornos reales los datos sufren:
1. Sensores ruidosos o transmisiones con pérdida de precisión.
2. Deriva conceptual (*concept drift*) debido a estacionalidades, cambios de régimen operativo o desgaste progresivo.
3. Desbalance extremo de clases, donde las anomalías representan menos del 3% del total de observaciones.

El objetivo central de **DataSentinel** es responder a la siguiente pregunta de investigación:
> *¿En qué medida los cambios en la distribución y el incremento de ruido afectan al rendimiento (F1-score) y a la tasa de falsas alarmas de los modelos estadísticos tradicionales en comparación con los métodos basados en subespacios de Machine Learning?*

---

## 2. Marco Metodológico y Métricas de Evaluación

Para evitar sesgos experimentales y garantizar la reproducibilidad científica, se implementan las siguientes restricciones metodológicas:
- **Causalidad Estricta:** Las ventanas móviles y características temporales emplean un desfase temporal estricto ($t-1$) mediante operadores de retraso, impidiendo cualquier fuga de información del futuro (*look-ahead bias*).
- **Partición Temporal:** Se preserva la secuencia temporal dividiendo estrictamente entre entrenamiento ($t \le T_{split}$) y prueba ($t > T_{split}$).
- **Evaluación Dual (Puntual vs. Por Evento):** Se analiza el compromiso entre la precisión por punto y la métrica de detección temprana por evento (*Event-Aware Recall*), prescindiendo deliberadamente del ajuste artificial por punto (*point adjustment* indiscriminado).

### Modelos Evaluados
1. **Rolling Z-Score:** Umbral estadístico basado en media y desviación estándar móviles.
2. **Rolling IQR:** Método no paramétrico robusto basado en el rango intercuartílico ($Q_3 - Q_1$).
3. **Isolation Forest:** Aislamiento no lineal de observaciones sobre embeddings temporales calculados mediante ventanas causales.
4. **Local Outlier Factor (LOF):** Medida de desviación de densidad local en relación con los $k$-vecinos más cercanos entrenados en condiciones normales.

---

## 3. Resultados Experimentales

### 3.1. Rendimiento Base (Condiciones Ideales)
Sobre la serie temporal UCR (longitud de test = 25.000 observaciones, 620 puntos anómalos equivalentes al 2,48%):

| Modelo | Precision | Recall | F1-Score | Falsas Alarmas (FP) | Event-Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Rolling Z-Score (w=100)** | 0.084 | 0.021 | 0.034 | 142 | 1.00 |
| **Rolling IQR (w=100, k=2.5)** | 0.433 | 0.042 | 0.076 | **34** | 1.00 |
| **Isolation Forest (ML)** | 0.148 | 0.215 | 0.175 | 765 | 1.00 |
| **Local Outlier Factor (ML)** | **0.507** | **1.000** | **0.673** | 602 | 1.00 |

### 3.2. Robustez Frente a Ruido Gaussiano Aditivo
Al inyectar ruido proporcional a $\sigma \in \{0.1, 0.25, 0.5\}$:
- Los métodos basados en densidad (LOF) sufren una saturación de detecciones, multiplicando los falsos positivos conforme las vecindades locales se homogenizan por efecto del ruido.
- Rolling IQR conserva una especificidad notablemente alta, conteniendo la explosión de falsas alertas mejor que los detectores basados en varianza gaussiana.

### 3.3. Degradación bajo Deriva Conceptual (Concept Drift)
En escenarios de *mean shift* repentino ($\Delta \mu = 1.5\sigma, 3.0\sigma$) y tendencias lineales:
- Los modelos con horizontes de memoria acotados (ventanas móviles) adaptan su referencia gradualmente.
- Los modelos entrenados con fronteras fijas globales sobre la partición normal sufren un colapso de precisión sostenido al interpretar el cambio de régimen como una anomalía continua.

---

## 4. Conclusiones y Trabajo Futuro
1. La métrica *Event-Recall* es insuficiente por sí sola para juzgar la viabilidad de un modelo, ya que un detector con un número desproporcionado de falsos positivos puede alcanzar un 100% de éxito por evento.
2. Existe un claro *trade-off* entre recall exhaustivo y coste operativo: algoritmos complejos como LOF maximizan el recall pero introducen más de 600 alertas falsas sobre un único flujo temporal.
3. Como trabajo futuro, se proyecta extender el benchmark a series multivariantes y formular mecanismos de auto-ajuste de umbrales en línea.
