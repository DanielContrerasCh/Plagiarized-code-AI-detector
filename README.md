# Reto-Compiladores-Plagio-

Reto en equipo del bloque de compiladores: **detección de plagio en código mediante la identificación de código generado por LLM**, replicando a Ramachandra et al. (2026) sobre el dataset de la SemEval-2026 Task 13 (Subtask A).

**Equipo (EquipoCompiladores):** Juan Pablo Chávez Leal · Mauricio Anguiano Juárez · Daniel Contreras Chávez — ITESM, Campus Querétaro (TC3002B).

## Contenido

| Archivo | Qué es |
|---|---|
| [`RetoPlagio.ipynb`](RetoPlagio.ipynb) | Notebook con todo el código: features + TF-IDF + Random Forest / XGBoost |
| [`Marco de referencia.md`](Marco%20de%20referencia.md) | Problemática, marco teórico, estado del arte, metodología, resultados y limitaciones |
| `modelo_reto.joblib` | Modelo XGBoost ya entrenado (~1 MB) — permite probar el notebook sin reentrenar |
| [`figuras/`](figuras/) | Gráficas de evaluación generadas por el notebook |

## Probar el modelo sin entrenar

1. Clonar el repo y abrir `RetoPlagio.ipynb`.
2. Correr la CELDA 1 (imports), la celda CARGAR (11), la CELDA 15 y la celda de PRUEBAS (16).
3. (Opcional) Correr la CELDA 12 para descargar el Random Forest ya entrenado (~376 MB, desde el Release del repo) y probar con los dos modelos.
4. Pegar cualquier fragmento de código en el diccionario `pruebas` para clasificarlo como humano o generado por IA, con su probabilidad.

El entrenamiento completo (~500k fragmentos, requiere credenciales de Kaggle) solo es necesario para reproducir los resultados desde cero.

## Resultados (macro F1, validación oficial)

| Modelo | Macro F1 |
|--------|----------|
| **XGBoost** | **0.9917** |
| Random Forest | 0.9832 |

## Referencia

Ramachandra, S., Chaudhary, A., Tran, J., Desai, P., Pang, A., & Salloum, M. (2026). *Detecting AI-Generated Code in Introductory Programming Courses.* SIGCSE TS 2026, UC Riverside. https://doi.org/10.1145/3770762.3772522
