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

### Configuración

El proyecto requiere un archivo `.env` en la raíz del proyecto para configurar todos los parámetros del pipeline. El módulo `config.py` utiliza **Pydantic Settings** para cargar estos valores desde variables de entorno.

**Es necesario crear un archivo `.env` antes de ejecutar el pipeline.** A continuación se muestra un ejemplo con todos los parámetros esperados:

```env
# ============================================
# Data I/O
# ============================================
HISTORICAL_CSV=data/archivo_con_tus_datos.csv
TIME_COL=time            # nombre de la columna con fecha de tu dataset de precios
PRICE_COL=close_price    # nombre de la columna del precio que quieres modelar
PARSE_DATES=true
TIMEZONE=

# ============================================
# Walk-forward
# ============================================
TRAIN_BARS=2500
TEST_BARS=500
STEP_BARS=
EMBARGO=

# ============================================
# Trades / Labeling
# ============================================
H=50
TP_MULT=1.5
SL_MULT=1.0
FAST_MA_PERIOD=10
SLOW_MA_PERIOD=30
METHOD=atr
PAST_BARS=50
SIDE=long

# ============================================
# State / Nonparametric Estimator
# ============================================
VOL_WINDOW=60
ALPHA=
WARMUP=
H_MU=0.2
H_SIGMA=0.2

# ============================================
# Simulation
# ============================================
N_PATHS=500
N_STEPS=2000
DT=1.0
BURNIN=300
SEED=123

# ============================================
# Synthetic Usage Control
# ============================================
RHO_MAX=2.0

# ============================================
# Fold Filtering Thresholds
# ============================================
MIN_TRAIN_TRADES=50
MIN_TEST_TRADES=20

# ============================================
# Outputs
# ============================================
OUT_SUMMARY_CSV=walkforward_summary.csv         # aquí debe ir la ruta completa de salida de resultados, como en la siguiente
OUT_FULL_JSON=walkforward_full_results.json
```

**Notas importantes:**
- Los valores vacíos (`=`) utilizan los valores por defecto definidos en `config.py`.
- `HISTORICAL_CSV` es obligatorio (no tiene valor por defecto).
- Los parámetros opcionales pueden omitirse o dejarse vacíos para usar sus valores por defecto.
- Para más detalles sobre cada parámetro, consulta la documentación en `config.py`.

### Ejecución

```bash
python pipeline_main.py
```

### Ejecución rápida (test)

Para una ejecución rápida de prueba, puedes usar estos valores reducidos en tu `.env`:

```env
# Valores reducidos para test rápido
TRAIN_BARS=1000
TEST_BARS=200
STEP_BARS=200
N_PATHS=50
N_STEPS=500
BURNIN=100
MIN_TRAIN_TRADES=20
MIN_TEST_TRADES=10
```

Esto reducirá significativamente el tiempo de ejecución mientras mantiene la funcionalidad del pipeline.

---

## 7. Nota final

Este proyecto está diseñado como una **base de investigación rigurosa**, no como un backtest optimista.