"""clm_losses: comparación controlada de funciones de pérdida para scorers estado→acción
sobre un LLM congelado (tipo CLM-8B).

Estructura:
    config    rutas (Drive / caché local), perfiles de datos, hiperparámetros por defecto
    utils     guardado atómico, registro de corridas, semillas, entorno
    heads     cabezales idénticos a CLM (servibles con clm-serve)
    losses    las 15 pérdidas + modelo ciego + término auxiliar de HN
    data      stores de embeddings (HF o sintéticos), splits por documento, streams reanudables
    train     entrenador con checkpoints y reanudación exacta
    evaluate  top-1, abstención, calibración, geometría
    stats     todos-contra-todos (bootstrap por documento, Holm, TOST, Friedman/Nemenyi)
    hpo       búsqueda aleatoria reanudable + curvas de desempeño esperado vs presupuesto
    encoder   Qwen3-8B (transformers o vLLM), last-token pooling
    texts     tokens de Nemotron: textos, longitudes, negativos minados / generados
    generators panel de generadores para T-gen (ENMIENDA_03): registro, generación, auditoría, macro
    genworker subproceso de generación (transformers; vLLM opcional; backend falso para pruebas)
    f0        pasos y compuertas de F0 (incluye F0.5b y F0.7b de la enmienda A1)
    trex      T-Rex con y sin shield (arnés de CLM)
    deepswe   verificación de identidad del set de evaluación de DeepSWE
    pipeline  funciones de alto nivel que llama cada celda del notebook
    analysis  uso del estado y transferencia entre generadores (EXP-006)
    preenvio  notebook pre-envío (fuera del protocolo): extracción de Drive, calidad de los sets (juez de falsos
              negativos, etiquetas a mano, fuga entre splits), condición de borde de B5 y referencias sin entrenar

v0.3.0 (2026-10-01): enmienda A1 (selección por val-mine, T-abst-mine, estados barajados, receta de CLM,
F1b, denominador "HN + in-batch") y panel de generadores T-gen. Las corridas de v0.2 conservan su huella.
v0.3.2 (2026-10-02): construcciones del panel T-gen (S, P, E) con piloto y regla fija (ENMIENDA_03 §11),
positivos propios en la evaluación, respaldos de §3.6 aunque el macro quede vacío, reglas fijas para B1-R2.
v0.3.8 (2026-10-11): notebook pre-envío. Denominadores nuevos de InfoNCE+HN (los existentes no cambian), juez de
falsos negativos en genworker y el módulo preenvio. No cambia nada de lo que produjo F0–F3.
"""
__version__ = "0.3.8"
