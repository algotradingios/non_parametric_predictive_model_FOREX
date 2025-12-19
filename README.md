# Non-parametric Synthetic Data Augmentation for Trading Strategy Classification
# WORK IN PROGRESS!

## 1. Objetivo del proyecto

Este proyecto implementa un **pipeline de investigación cuantitativa** para:

1. Modelar la dinámica del mercado mediante un **modelo estocástico no paramétrico** (SDE con drift y volatilidad condicionales).
2. Generar **trayectorias sintéticas de precios** realistas y coherentes con propiedades estilizadas de los mercados financieros.
3. Generar y etiquetar **operaciones de trading** (ganadoras / perdedoras) mediante reglas explícitas (EMA cross + triple barrera).
4. Utilizar los datos sintéticos como **data augmentation** para entrenar un **clasificador de estrategias de trading**.
5. Evaluar el rendimiento del clasificador usando un esquema **walk-forward sin fuga de información**.
6. Validar explícitamente el **simulador** antes de usar los datos sintéticos para entrenamiento.

El foco del proyecto no es el *pricing* exacto, sino la **generación de escenarios plausibles** que mejoren el aprendizaje supervisado de clasificadores de decisiones de trading.

---

## 2. Principios metodológicos clave

El diseño del proyecto se apoya en los siguientes principios:

- **No parametricidad**  
  No se impone una forma funcional para la media ni la volatilidad condicional.

- **Baja dimensionalidad del estado**  
  El estado del mercado se modela como:
  \[
  Z_t = (r_t,\; \log \hat\sigma_t)
  \]

- **Separación estricta temporal**  
  Ningún componente (simulador, etiquetas, clasificador) accede a información futura.

- **Walk-forward realista**  
  Cada evaluación se realiza fuera de muestra, respetando el horizonte de etiquetado.

- **Validación del simulador**  
  Los datos sintéticos **no se usan** si no reproducen adecuadamente propiedades estadísticas del mercado real.

---

## 3. Estructura del proyecto

```
project/
│
├── .env
├── README.md
├── config.py
├── pipeline_main.py
│
├── qfin_synth/
│   ├── __init__.py
│   ├── state.py
│   ├── nw.py
│   ├── sde.py
│   ├── trades.py
│   ├── validation.py
│   ├── walkforward.py
│   └── models.py
```

---

## 4. Descripción detallada de los módulos

### 4.1 `config.py` — Configuración del experimento

Configuración centralizada mediante **Pydantic Settings**, leyendo parámetros desde un archivo `.env`.  
Garantiza tipado fuerte, validación temprana y valores por defecto coherentes para reproducibilidad experimental.

### 4.2 `state.py` — Construcción del estado del mercado

Construye el estado:
\[
Z_t = (r_t, \log \hat\sigma_t)
\]
a partir de precios históricos, usando retornos logarítmicos y volatilidad EWMA.

### 4.3 `nw.py` — Estimación no paramétrica (Nadaraya–Watson)

Implementa estimadores kernel para la media y la volatilidad condicionales sin imponer forma funcional.

### 4.4 `sde.py` — Simulación estocástica de precios

Simula trayectorias sintéticas mediante un esquema de Euler–Maruyama consistente con el estimador no paramétrico.

### 4.5 `trades.py` — Generación y etiquetado de operaciones

Genera entradas por cruce de EMAs y etiqueta operaciones mediante un esquema de triple barrera basado en ATR o desviación estándar.

### 4.6 `validation.py` — Validación del simulador

Evalúa si las trayectorias sintéticas reproducen propiedades estilizadas del mercado antes de usarlas como datos de entrenamiento.

### 4.7 `walkforward.py` — Walk-forward sin fuga de información

Define splits temporales train/test con embargo explícito para evitar contaminación por el horizonte de etiquetado.

### 4.8 `models.py` — Clasificador

Entrena un clasificador supervisado (baseline con Gradient Boosting) y evalúa exclusivamente fuera de muestra.

---

## 5. Flujo completo del pipeline

1. División walk-forward del histórico.
2. Estimación del modelo no paramétrico en train.
3. Simulación de precios sintéticos.
4. Validación estadística del simulador.
5. Generación de trades sintéticos (si el simulador es válido).
6. Entrenamiento del clasificador.
7. Evaluación estrictamente fuera de muestra.

---

## 6. Ejecución

### Requisitos

```bash
pip install numpy pandas scikit-learn scipy pydantic pydantic-settings ta
```

### Ejecución

```bash
python pipeline_main.py
```

---

## 7. Nota final

Este proyecto está diseñado como una **base de investigación rigurosa**, no como un backtest optimista.