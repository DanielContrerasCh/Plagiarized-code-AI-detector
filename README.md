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

## Resultados (macro F1, validación oficial)

| Modelo | Macro F1 |
|--------|----------|
| **XGBoost** | **0.9917** |
| Random Forest | 0.9832 |

En el paper de `Ramachandra et al. (2026)` se reporta una F1 score de **0.988** (TF-IDF + Ensemble) sobre su propio dataset de 3,360 muestras de C++. Nuestros resultados son consistentes y se obtienen sobre un dataset dos órdenes de magnitud más grande (~600k fragmentos) y multilingüe (C++, Python, Java).

El hecho de que tengamos un dataset mucho más grande y multilingüe explica el hecho de que nuestro modelo haya podido aprender patrones más robustos. Además, en el paper se menciona que incluyeron "datos envenenados", mientras que nuestro dataset tenía un balance cercano al 48/52.

## Arquitectura del modelo
 
```
Fragmento de código
        │
        ├── 11 features estructurales   ── StandardScaler
        │   (líneas, indentación,
        │    comentarios, espacios…)
        │
        └── TF-IDF (20,000 features)
            (unigramas + bigramas,
             fit solo en train)
                         │
              scipy.sparse.hstack → 20,011 features
                    ┌────┴────┐
             Random Forest   XGBoost
             (300 árboles,   (400 árboles,
              balanced)       depth=8, lr=0.1)
```
 
---


## Referencia

Ramachandra, S., Chaudhary, A., Tran, J., Desai, P., Pang, A., & Salloum, M. (2026). *Detecting AI-Generated Code in Introductory Programming Courses.* SIGCSE TS 2026, UC Riverside. https://doi.org/10.1145/3770762.3772522
