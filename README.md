# Non-parametric Synthetic Data Augmentation for Trading Strategy Classification

## 1. Objetivo del proyecto

Este proyecto implementa un **pipeline de investigación cuantitativa** para:

1. Modelar la dinámica del mercado mediante un **modelo estocástico no paramétrico** (SDE con drift y volatilidad condicionales).
2. Generar **trayectorias sintéticas de precios** realistas y coherentes con propiedades estilizadas de los mercados financieros.
3. Generar y etiquetar **operaciones de trading** (ganadoras / perdedoras) mediante reglas explícitas (EMA cross + triple barrera).
4. Utilizar los datos sintéticos como **data augmentation** para entrenar un **clasificador de estrategias de trading**.
5. Evaluar el rendimiento del clasificador usando una **división train/test temporal sin fuga de información**.
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

- **Evaluación fuera de muestra**  
  Cada evaluación se realiza en un conjunto de prueba separado temporalmente, respetando el horizonte de etiquetado y el embargo para evitar fuga de información.

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

### 4.7 `walkforward.py` — Filtrado temporal de trades

Filtra trades según rangos temporales de entrenamiento y prueba, aplicando embargo explícito para evitar contaminación por el horizonte de etiquetado.

### 4.8 `models.py` — Clasificador

Entrena un clasificador supervisado (baseline con Gradient Boosting) y evalúa exclusivamente fuera de muestra.

---

## 5. Flujo completo del pipeline

El pipeline ejecuta las siguientes fases en orden secuencial:

### Fase 1: Carga y preparación de datos históricos

- **Carga del CSV**: Lee el archivo CSV especificado en `HISTORICAL_CSV` y carga únicamente el **10% de las filas** para pruebas rápidas (configurable en el código).
- **Ordenamiento temporal**: Ordena los datos por la columna temporal (`TIME_COL`) para garantizar secuencialidad.
- **Detección de columnas**: Detecta automáticamente si existen columnas de precio alto (`bid_price_high`) y bajo (`bid_price_low`) en los datos.
- **Manejo de valores faltantes**: Aplica forward-fill y backward-fill a los precios para manejar valores NaN.

### Fase 2: Extracción y cálculo de características

- **Extracción de características del CSV**: Si las características especificadas en `FEATURE_COLS` existen en el CSV (por ejemplo, `bid_tick_count`), se extraen directamente del DataFrame histórico.
- **Cálculo de características derivadas**: Si una característica está en `FEATURE_COLS` pero no existe en los datos, se calcula:
  - **`mom20`** o **`mom`**: Media móvil de retornos porcentuales sobre una ventana de `FEATURE_WINDOW` períodos.
  - **`vol20`** o **`vol`**: Desviación estándar móvil de retornos porcentuales sobre una ventana de `FEATURE_WINDOW` períodos.
  - **`ma_ratio`**: Ratio entre media móvil rápida (`FAST_MA_PERIOD`) y lenta (`SLOW_MA_PERIOD`).
- **Preparación de `price_df`**: Se crea un DataFrame con columnas de precio (`close_price`, `high_price`, `low_price` si están disponibles) y todas las características especificadas.

### Fase 3: Generación de trades reales

- **Generación de señales de entrada**: Utiliza el método de cruce de EMAs (`FAST_MA_PERIOD` vs `SLOW_MA_PERIOD`) para identificar puntos de entrada.
- **Etiquetado con triple barrera**: Etiqueta cada operación según el método especificado (`METHOD`):
  - **`atr`**: Usa Average True Range (ATR) si hay precios alto/bajo disponibles.
  - **`std`**: Usa desviación estándar móvil como alternativa.
- **Cálculo de barreras**: Define take-profit (`TP_MULT * H`), stop-loss (`SL_MULT * H`) y límite temporal (`H` barras).
- **Extracción de características por trade**: Cada trade incluye las características especificadas en `FEATURE_COLS` evaluadas en el momento de entrada.

### Fase 4: División train/test

- **División temporal simple**: Divide los datos históricos en dos conjuntos:
  - **Entrenamiento**: Primeros `TRAIN_SPLIT * 100%` de los datos (por defecto 80%).
  - **Prueba**: Resto de los datos (20%).
- **Segmentación de precios**: Separa las series de precios en `prices_train` y `prices_test` según los índices calculados.

### Fase 5: Filtrado de trades por fold

- **Filtrado temporal**: Utiliza `filter_trades_for_fold()` para separar trades reales en:
  - **`train_real`**: Trades cuya entrada (`entry_idx`) está en el rango de entrenamiento y cuya salida (`exit_idx`) respeta el embargo (`EMBARGO`).
  - **`test_real`**: Trades cuya entrada está en el rango de prueba.
- **Sin fuga de información**: El embargo garantiza que ningún trade de entrenamiento use información futura más allá del horizonte de etiquetado.

### Fase 6: Binarización y limpieza de etiquetas

- **Filtrado de etiquetas**: Elimina trades con `label == 0` (sin señal clara) y mantiene solo `label ∈ {1, -1}`.
- **Binarización**: Convierte etiquetas `{1, -1}` a `{1, 0}` donde `1` = ganador y `0` = perdedor.
- **Limpieza de NaN**: Elimina trades con características NaN (insuficiente historia para cálculo de ventanas móviles o datos faltantes).
- **Validación de umbrales**: Verifica que haya al menos `MIN_TRAIN_TRADES` trades en entrenamiento y `MIN_TEST_TRADES` en prueba. Si no se cumple, el fold se omite.

### Fase 7: Cálculo del estado del mercado

- **Construcción del estado**: Calcula el estado del mercado `Z_t = (r_t, log σ̂_t)` donde:
  - `r_t`: Retorno logarítmico en el tiempo `t`.
  - `log σ̂_t`: Logaritmo de la volatilidad estimada usando EWMA con ventana `VOL_WINDOW` y parámetro `ALPHA`.
- **Preparación de datos**: Crea un DataFrame con precios de entrenamiento para `compute_basic_state()`.
- **Validación de muestras**: Verifica que haya suficientes muestras válidas después del warmup (`WARMUP`). Si no, el fold se omite.

### Fase 8: Ajuste del estimador no paramétrico

- **Entrenamiento del estimador**: Entrena un estimador Nadaraya-Watson (`MuSigmaNonParam`) con:
  - **Anchos de banda**: `H_MU` para la media condicional y `H_SIGMA` para la volatilidad condicional.
  - **Datos de entrada**: Matriz `X` de estados del mercado y vector `y` de retornos futuros.
- **Modelo no paramétrico**: El estimador aprende la función de drift `μ(Z_t)` y volatilidad `σ(Z_t)` sin imponer forma funcional.

### Fase 9: Simulación de trayectorias sintéticas

- **Simulación SDE**: Utiliza el esquema de Euler-Maruyama para simular `N_PATHS` trayectorias sintéticas:
  - **Precio inicial**: Último precio del conjunto de entrenamiento.
  - **Número de pasos**: `N_STEPS` por trayectoria.
  - **Paso temporal**: `DT`.
  - **Burn-in**: `BURNIN` pasos iniciales descartados para estabilización.
- **Generación de precios**: Cada trayectoria sigue la dinámica:
  ```
  dS_t = μ(Z_t) S_t dt + σ(Z_t) S_t dW_t
  ```
  donde `Z_t` se actualiza dinámicamente durante la simulación.

### Fase 10: Validación del simulador

- **Validación estadística**: Compara propiedades estadísticas entre precios reales y sintéticos:
  - **Test de Kolmogorov-Smirnov**: Compara distribuciones de retornos y realized volatility.
  - **Estadísticas**: Calcula `ks_ret_stat` (KS para retornos) y `ks_rv_stat` (KS para volatilidad realizada).
- **Decisión binaria**: El simulador se considera válido (`simulator_ok = True`) si ambas estadísticas KS están por debajo de umbrales predefinidos.
- **Registro de diagnóstico**: Se guardan todas las métricas de validación para análisis posterior.

### Fase 11: Generación de trades sintéticos (condicional)

- **Generación condicional**: Solo se ejecuta si `simulator_ok == True`.
- **Aplicación de estrategia**: Aplica la misma lógica de generación de trades (`generate_trades_from_paths()`) a las trayectorias sintéticas:
  - Cruce de EMAs para entradas.
  - Triple barrera para etiquetado.
  - Extracción de características.
- **Binarización**: Convierte etiquetas `{1, -1}` a `{1, 0}` y marca `is_synth = 1`.
- **Limitación de volumen**: Limita el número de trades sintéticos a `rho_max * len(train_real)` para evitar dominancia sobre trades reales.

### Fase 12: Entrenamiento del clasificador

- **Combinación de datos**: Concatena `train_real` y `train_synth` (si están disponibles) en un único DataFrame de entrenamiento.
- **Preparación de características**: Selecciona las columnas especificadas en `FEATURE_COLS` como variables predictoras.
- **Entrenamiento**: Entrena un clasificador Gradient Boosting (`GradientBoostingClassifier` de scikit-learn) con:
  - **División interna**: 80% entrenamiento / 20% validación interna (solo para diagnóstico, no evaluación final).
  - **Métricas internas**: Calcula AUC, F1 y MCC en el conjunto de validación interna.
- **Preparación de etiquetas**: Usa la columna `label` binarizada (`{0, 1}`) como variable objetivo.

### Fase 13: Evaluación fuera de muestra

- **Predicción en test**: Evalúa el clasificador entrenado en el conjunto de prueba (`test_real`):
  - **Probabilidades**: Calcula probabilidades predichas `p = P(y=1 | X)`.
  - **Predicciones binarias**: Convierte probabilidades a predicciones binarias usando umbral 0.5.
- **Cálculo de métricas**: Calcula métricas de evaluación:
  - **AUC (Area Under ROC Curve)**: Capacidad de discriminación entre clases.
  - **F1 Score**: Media armónica de precisión y recall.
  - **MCC (Matthews Correlation Coefficient)**: Correlación entre predicciones y valores reales.
- **Sin fuga de información**: Todas las métricas se calculan exclusivamente en datos nunca vistos durante el entrenamiento.

### Fase 14: Guardado de resultados

- **Resumen CSV**: Guarda un resumen en `OUT_SUMMARY_CSV` con:
  - Rangos de entrenamiento y prueba.
  - Número de trades reales y sintéticos.
  - Métricas de evaluación (AUC, F1, MCC).
  - Estado de validación del simulador.
- **Resultados completos JSON**: Guarda resultados detallados en `OUT_FULL_JSON` incluyendo:
  - Diagnósticos completos del simulador.
  - Métricas internas de entrenamiento.
  - Información completa de la ejecución (rangos train/test, número de trades, etc.).
- **Logging**: Registra todas las fases y métricas intermedias para trazabilidad y debugging.

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
# Train/Test Split
# ============================================
TRAIN_SPLIT=0.8          # Fraction of data for training (0.8 = 80% train, 20% test)
EMBARGO=                  # If empty, defaults to H (horizon for labeling)

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
FEATURE_WINDOW=20         # Rolling window period for mom20 and vol20 features
FEATURE_COLS=bid_tick_count    # Comma-separated list of feature columns (e.g., "mom20,vol20,ma_ratio" or "bid_tick_count")

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
TRAIN_SPLIT=0.8
N_PATHS=50
N_STEPS=500
BURNIN=100
MIN_TRAIN_TRADES=20
MIN_TEST_TRADES=10
FEATURE_COLS=bid_tick_count    # O cualquier característica disponible en tu CSV
```

**Nota importante**: El pipeline carga automáticamente solo el **10% de las filas** del CSV para pruebas rápidas. Esto está implementado directamente en `pipeline_main.py` y puede modificarse según necesidades.

Esto reducirá significativamente el tiempo de ejecución mientras mantiene la funcionalidad del pipeline.

---

## 7. Nota final

Este proyecto está diseñado como una **base de investigación rigurosa**, no como un backtest optimista.