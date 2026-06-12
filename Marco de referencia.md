# Detección de Código Generado por IA — SemEval-2026 Task 13 (Subtask A)

**Notebook:** [`RetoPlagio.ipynb`](RetoPlagio.ipynb)
**Equipo (EquipoCompiladores):** Juan Pablo Chávez Leal · Mauricio Anguiano Juárez · Daniel Contreras Chávez — ITESM, Campus Querétaro (TC3002B).

---

## 1. Contexto y problemática

La adopción masiva de asistentes de programación basados en LLM (ChatGPT, Copilot, Gemini) ha hecho cada vez más difícil distinguir el código escrito por una persona del generado por una IA. En cursos introductorios de programación esto representa un nuevo tipo de deshonestidad académica que los detectores clásicos de similitud (MOSS, JPlag) no pueden cubrir, ya que no existe una fuente "original" con la cual comparar: cada fragmento generado por un LLM es sintácticamente único.

**El problema:** clasificación binaria a nivel de fragmento de código.

- `0` = humano
- `1` = máquina (generado por IA)

**Objetivo general:** replicar el enfoque de Ramachandra et al. (2026) para la detección de código generado por IA y evaluarlo de forma reproducible sobre el dataset oficial de la competencia SemEval-2026 Task 13, adaptando su arquitectura a este dataset.

## 2. Dataset

- **Fuente:** Kaggle [`daniilor/semeval-2026-task13`](https://www.kaggle.com/datasets/daniilor/semeval-2026-task13) (SemEval-2026 Task 13, Subtask A), versión 3, ~1.03 GB.
- **Columnas:** `code` (fragmento de código fuente), `label` (0 = humano, 1 = máquina), `language` (C++, Python, Java, entre otros), `generator` (modelo/origen del fragmento).
- **Tamaño:** 601,000 filas en total.
- **Splits:** se usan los **splits oficiales de la competencia** — el conjunto de entrenamiento (`task_a_training_set_1.parquet`) para entrenar y el de validación (`task_a_validation_set.parquet`) para evaluar.
- La descarga se realiza vía la API de Kaggle (requiere credenciales del usuario).

## 3. Arquitectura y metodología (replicación del paper)

Siguiendo a Ramachandra et al. (2026), en lugar de redes neuronales profundas se detecta el código generado por IA combinando **características extraídas del código fuente** con representaciones de tokens, clasificadas mediante **modelos de aprendizaje automático tradicional**.

### Features (11 estructurales/estilométricas)

Calculadas por fragmento: `n_lines`, `n_chars`, `n_blank`, `n_comment`, `avg_indent`, `avg_line_len`, `n_space`, `n_tab`, `ratio_ws`, `ratio_comment`, `ratio_blank`.

### TF-IDF sobre tokens del código

`lowercase=False` (preserva identificadores), `ngram_range=(1,2)`, `max_features=20000`, `min_df=2`.

Ambos bloques se concatenan en una matriz sparse: **20,000 columnas TF-IDF + 11 estructurales = 20,011 features**.

> Para evitar fuga de datos (*data leakage*), el vectorizador TF-IDF y el escalador (`StandardScaler`) se ajustan **únicamente con el conjunto de entrenamiento** y solo se aplican (transform) sobre el de validación.

### Modelos

- `RandomForestClassifier(n_estimators=300, class_weight='balanced')`
- `XGBClassifier(n_estimators=400, max_depth=8, learning_rate=0.1, subsample=0.9, colsample_bytree=0.9)`

La métrica de evaluación es **macro F1**, la métrica oficial de la Subtask A.

## 4. Resultados

Métrica: **macro F1** sobre el conjunto de evaluación.


| Modelo | Macro F1 |
|--------|----------|
| **XGBoost** | **0.9917** |
| Random Forest | 0.9832 |

Estos resultados son consistentes con los reportados por Ramachandra et al. (F1 = 0.988 con TF-IDF + Ensemble sobre su propio dataset): la representación estadística (features + TF-IDF) con clasificadores de árboles es suficiente para separar código humano de código generado.

## 5. Reproducción y persistencia

1. **Probar el modelo sin reentrenar (flujo recomendado al clonar el repo):** ejecutar la CELDA 1 (imports), la celda **CARGAR (11)**, la CELDA 15 (`predict_code`) y la celda de **PRUEBAS (16)**. El modelo XGBoost ya entrenado viene versionado en el repo (`modelo_reto.joblib`, ~1 MB, con el TF-IDF y el scaler incluidos), así que no se necesita descargar el dataset ni entrenar nada. Para probar también con el Random Forest, la **CELDA 12** (opcional) lo descarga ya entrenado (~376 MB) desde el Release `modelos-v1` del repositorio.
2. **Reentrenar desde cero:** ejecutar el notebook completo (requiere credenciales de Kaggle). La celda **GUARDAR (10)** regenera `modelo_reto.joblib` y además serializa el bundle completo (incluido el Random Forest) en `artefactos_reto/` (~380 MB; excluido de git porque GitHub rechaza archivos > 100 MB).
3. El notebook también genera gráficas de evaluación en `figuras/`: matrices de confusión (`matrices_confusion.png`), curvas ROC (`curvas_roc.png`), importancia de features (`importancia_features.png`) y el diagnóstico de overfitting (`train_vs_val_loss.png`).

## 6. Marco de referencia

### Marco teórico y conceptual

**Plagio tradicional vs. detección de origen.** Los detectores clásicos de plagio en código (MOSS, JPlag) operan por **comparación entre pares**: dado un conjunto de entregas, buscan parejas con alta similitud estructural (huellas digitales de tokens, *greedy string tiling*), bajo el supuesto de que el plagio implica copiar de una fuente existente. El código generado por un LLM rompe ese supuesto: cada generación es sintácticamente única y no existe una "fuente original" contra la cual comparar. Por ello, el problema se replantea como **detección de origen**: clasificar un fragmento *de forma intrínseca* — sin compararlo con nada — según las huellas que su proceso de generación deja en él. Formalmente, es una tarea de **clasificación binaria supervisada**: aprender una función *f(código) → {humano, máquina}* a partir de ejemplos etiquetados.

**Estilometría de código.** Es la base teórica que hace viable la detección: así como la estilometría de texto identifica autores por sus patrones de escritura, el código tiene una "huella estilística" medible. Los LLM, al generar el token más probable según su entrenamiento, producen código con regularidades características — indentación perfectamente consistente, densidad de comentarios alta y uniforme, nombres descriptivos, estructura canónica —, mientras que el código humano (especialmente de estudiantes principiantes) exhibe mayor variabilidad: indentación irregular, mezcla de estilos, comentarios escasos o ausentes. Las 11 features estructurales del modelo (proporción de comentarios, indentación promedio, proporción de espacios en blanco, etc.) operacionalizan directamente esta teoría.

**Representaciones del código.** Un clasificador no consume texto, consume vectores; la elección de representación determina qué patrones puede aprender. Además de las 11 features estructurales, el código se representa con **TF-IDF sobre n-gramas de tokens**: cada fragmento se convierte en una bolsa de tokens y pares de tokens ponderados por *term frequency × inverse document frequency* — un token pesa más cuanto más aparece en el fragmento y menos común es en el corpus. Esta representación captura el léxico y los patrones locales (qué identificadores, operadores y construcciones usa el código), es interpretable y barata de calcular; su limitación es que no captura semántica (dos códigos equivalentes con identificadores distintos producen vectores distintos), aunque para detectar *estilo* —no significado— resulta suficiente, como muestran los resultados.

**Clasificadores de ensamble de árboles.** Un árbol de decisión particiona el espacio de features con reglas secuenciales; es expresivo pero propenso al sobreajuste. Los métodos de ensamble corrigen esto combinando muchos árboles bajo dos estrategias opuestas:

- ***Bagging* — Random Forest:** entrena cientos de árboles independientes, cada uno sobre una muestra bootstrap de los datos y un subconjunto aleatorio de features, y predice por votación mayoritaria. Al promediar árboles descorrelacionados, los errores individuales se cancelan: reduce la **varianza** sin aumentar el sesgo.
- ***Boosting* — XGBoost:** entrena árboles en secuencia, donde cada árbol nuevo se ajusta a los errores residuales del ensamble acumulado, siguiendo el gradiente de la función de pérdida (log-loss), con regularización y una tasa de aprendizaje que modera cada paso. Reduce el **sesgo** capturando patrones progresivamente más finos.

Se prefieren sobre redes neuronales profundas para este tipo de entrada (matriz tabular dispersa de 20,011 columnas) porque rinden bien sin ajuste extenso, entrenan más rápido, y ofrecen interpretabilidad vía importancia de features — además de ser los modelos del paper de referencia.

**Conceptos de evaluación.** El **macro F1** promedia el F1 de cada clase sin ponderar por su frecuencia, por lo que un modelo no puede obtener buen puntaje ignorando la clase minoritaria. La **fuga de datos** (*data leakage*) ocurre cuando información del conjunto de evaluación contamina el entrenamiento (p. ej., ajustar el vectorizador con todos los datos); se controla ajustando todos los preprocesadores solo con el train. El **sobreajuste** (*overfitting*) es la brecha entre el desempeño en entrenamiento y en datos no vistos; se diagnostica comparando la pérdida en ambos conjuntos conforme crece la capacidad del modelo (rondas de boosting).

### Estado del arte y trabajos relacionados

El trabajo de referencia principal es **Ramachandra et al. (2026)**, *Detecting AI-Generated Code in Introductory Programming Courses* (SIGCSE TS 2026). Sus aportaciones centrales: (a) cuantifica la incidencia del problema — entre 10% y 15% de los estudiantes de CS1 cometen deshonestidad asistida por IA; (b) demuestra que los detectores de similitud tradicionales no son efectivos contra código generado por LLM; y (c) propone detectarlo con features estilométricas y representaciones de tokens clasificadas con modelos de ensamble, reportando F1 = 0.988 con TF-IDF + Ensemble sobre 3,360 muestras de C++. Este proyecto **replica su arquitectura** sobre un dataset distinto, dos órdenes de magnitud más grande y multilingüe (SemEval-2026), obteniendo resultados consistentes (macro F1 = 0.9917 con XGBoost), lo que aporta evidencia de que el enfoque generaliza más allá del dataset original del paper.

Los demás trabajos del marco complementan el panorama: **Liu et al. (2025)** documenta que los detectores actuales de código generado por IA son vulnerables a técnicas de ofuscación (renombrado de variables, reformateo, paráfrasis del código), lo que delimita una amenaza a la validez de cualquier detector estilométrico, incluido el nuestro. **Boukili et al. (2026)** y **Apsara et al. (2026)** confirman la vigencia del problema en contextos educativos y la disponibilidad de datasets multilingües etiquetados que habilitan el enfoque supervisado. Finalmente, **Orel et al. (2025)** aporta los recursos **Droid** y **CoDet-M4**, de los que deriva el dataset de la competencia SemEval-2026 Task 13 utilizado aquí, y establece los resultados de referencia de la propia tarea contra los cuales se comparan nuestros modelos.

El hueco que este proyecto cubre respecto al estado del arte: validar la arquitectura de Ramachandra et al. a escala (600k fragmentos vs. 3,360) y en tres lenguajes (C++, Python, Java), con una comparación controlada entre las dos estrategias de ensamble (bagging vs. boosting) sobre la misma representación.

### Marco metodológico

#### Tipo y enfoque de la investigación

La investigación es de tipo **aplicado** y de naturaleza **cuantitativa**, con un enfoque **experimental**. Se sigue el método científico aplicado a las ciencias computacionales: a partir de un problema observable (el plagio en cursos de programación introductoria cuando los estudiantes entregan como propio código generado por un LLM), se formula una hipótesis de trabajo — *es posible clasificar un fragmento de código según su origen (humano o máquina) con un desempeño significativamente superior al azar y competitivo con el estado del arte, mediante técnicas de aprendizaje automático supervisado* —, se diseña un experimento controlado de entrenamiento y evaluación de modelos sobre un dataset etiquetado, y se contrastan los resultados con métricas objetivas y con los valores reportados en la literatura.

El diseño experimental es **comparativo**: sobre una misma representación del código (features estructurales + TF-IDF) se enfrentan dos clasificadores de ensamble con estrategias opuestas (Random Forest, basado en *bagging*, y XGBoost, basado en *boosting*), de modo que la diferencia de desempeño sea atribuible a la estrategia de ensamble y no a los datos.

#### Etapas de la metodología

El trabajo se organizó en cuatro etapas secuenciales:

1. **Revisión de literatura y definición del marco de trabajo**, tomando a Ramachandra et al. (2026) como referencia principal y a Liu et al. (2025), Boukili et al. (2026) y Apsara et al. (2026) como trabajos de contexto.
2. **Obtención y preparación de los datos** a partir del dataset SemEval-2026 Task 13 (descarga vía API de Kaggle con `kagglehub`), incluyendo limpieza (eliminación de filas sin código o sin etiqueta), verificación del balance de clases (~48/52) y uso de los **splits oficiales** de la competencia para entrenamiento y validación.
3. **Replicación del enfoque del paper (línea principal):** extracción de 11 features estructurales/estilométricas por fragmento, vectorización TF-IDF sobre n-gramas de tokens (1–2 gramas, 20,000 features), y entrenamiento de clasificadores de ensamble de árboles (Random Forest y XGBoost). En una corrida preliminar exploratoria, una línea base más simple (TF-IDF + regresión logística) alcanzó un macro F1 de 0.8978, lo que motivó el uso de ensambles de árboles como en el paper.
4. **Evaluación comparativa y análisis:** métricas estándar sobre la validación oficial, diagnóstico de overfitting (curva train vs. validation loss), desglose del error por lenguaje y por modelo generador, y contraste crítico con el estado del arte.

#### Datos, variables y parámetros experimentales

- **Variable dependiente:** la etiqueta de origen del fragmento (`label`: 0 = humano, 1 = generado por máquina), correspondiente a la Subtask A (clasificación binaria).
- **Variables independientes:** las representaciones del código fuente — (a) 11 features estructurales (líneas, caracteres, comentarios, indentación, espacios en blanco y sus proporciones) y (b) 20,000 features TF-IDF de unigramas y bigramas de tokens.
- **Particiones:** entrenamiento con ~500,000 ejemplos y validación con ~100,000, usando los splits oficiales de la competencia; la validación se reserva exclusivamente para evaluación. Para evitar fuga de información, el vectorizador TF-IDF y el `StandardScaler` se ajustan únicamente con el entrenamiento.
- **Hiperparámetros principales:** Random Forest con 300 árboles, profundidad sin límite y `class_weight='balanced'`; XGBoost con 400 árboles, profundidad máxima 8, tasa de aprendizaje 0.1, `subsample=0.9`, `colsample_bytree=0.9` y log-loss como función objetivo; TF-IDF con `lowercase=False` (preserva identificadores), rango de n-gramas (1,2), 20,000 features máximas y `min_df=2`.
- **Control de aleatoriedad:** semilla fija (`RANDOM_STATE = 42`) en muestreos y modelos.

#### Métricas de evaluación

La métrica principal es el **macro F1** (promedio no ponderado del F1 de cada clase), métrica oficial de la SemEval-2026 Task 13, apropiada porque trata por igual a ambas clases ante el desbalance. Como métricas complementarias se reportan la exactitud (accuracy) y la precisión, el recall y el F1 por clase. Para el análisis de errores se utilizan la matriz de confusión, la curva ROC con su AUC, y el desglose del macro F1 por lenguaje y de la tasa de acierto por modelo generador. Adicionalmente se verifica la generalización con un diagnóstico de overfitting: la curva de log-loss en entrenamiento contra validación conforme crecen las rondas de boosting de XGBoost, y el gap de macro F1 train/validación de ambos modelos.

El protocolo de evaluación consiste en entrenar sobre la partición de entrenamiento, evaluar sobre la partición de validación oficial y seleccionar el mejor modelo según el macro F1 en validación.

#### Herramientas y reproducibilidad

La experimentación se realiza en Jupyter Notebooks, ejecutable tanto localmente como en plataformas como Kaggle o Colab. Se emplean `pandas`, `numpy` y `scikit-learn` para el procesamiento y los clasificadores, y `xgboost` para el modelo de gradient boosting. La descarga del dataset se automatiza con `kagglehub`. Se fija una semilla aleatoria, se documentan los hiperparámetros en el propio notebook y los modelos entrenados se serializan con `joblib` (celdas GUARDAR/CARGAR), lo que permite reproducir la evaluación completa sin reentrenar.

## 7. Limitaciones

- El dataset es grande (601k filas), lo que hace el entrenamiento pesado (especialmente el Random Forest sin límite de profundidad).
- En fragmentos muy cortos (pocas líneas) hay poca señal estilométrica y el clasificador puede equivocarse, como muestra el ejemplo de la celda de predicción: una función trivial escrita a mano puede clasificarse como generada por IA.

## 8. Referencias

**Referencia principal:**

Ramachandra, S., Chaudhary, A., Tran, J., Desai, P., Pang, A., & Salloum, M. (2026). *Detecting AI-Generated Code in Introductory Programming Courses.* SIGCSE TS 2026, UC Riverside. https://doi.org/10.1145/3770762.3772522

Contexto del paper (no son resultados de este notebook): reporta F1 = 0.988 (TF-IDF + Ensemble) y F1 = 0.995 (CodeBERT + Ensemble) sobre 3,360 muestras de C++, y que ~10–15% de los estudiantes de CS1 cometen deshonestidad asistida por IA.

**Trabajos del marco de referencia:**

- Liu et al. (2025) — vigencia del problema de detección de código generado por IA y vulnerabilidad de los detectores a ofuscación.
- Boukili et al. (2026) y Apsara et al. (2026) — trabajos recientes sobre detección de código generado por LLM en contextos educativos.
- Orel, D., et al. (2025). *Droid / CoDet-M4* — recursos de los que deriva el dataset de la SemEval-2026 Task 13.

**Dataset:** SemEval-2026 Task 13, *Detecting Machine-Generated Code* — Kaggle: [`daniilor/semeval-2026-task13`](https://www.kaggle.com/datasets/daniilor/semeval-2026-task13).
