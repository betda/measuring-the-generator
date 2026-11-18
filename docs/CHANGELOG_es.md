# clm_losses — código del estudio de funciones de pérdida (v0.3.8)

Comparación controlada de 15 pérdidas para scorers estado→acción tipo CLM-8B: Qwen3-8B congelado y dos cabezales de proyección entrenados desde cero. El análisis es todos contra todos.

Proyecto: "cosas como jev". Diseño: `DISENO_01_decisiones_y_F0.md`, `ENMIENDA_03_A1_y_panel_Tgen.md` (enmienda A1 y panel T-gen) e `IMPLEMENTACION_02_codigo_colab.md` (en el proyecto).

**v0.3.0 (2026-10-01):**

- Selección por val-mine.
- Bloques nuevos en la evaluación: T-abst-mine, estados barajados, T-mine-50 y panel T-gen.
- Receta de CLM (LR constante, batch 16,384) y F0.7b.
- F1b.
- Panel de 16 generadores (`generators`, `genworker`).
- Análisis de transferencia (`analysis`).

**v0.3.1 (2026-10-01):**

- Panel ampliado a 23 generadores: 7 modelos de 14–34B (Gemma 4 31B, OLMo 3.1 32B, GLM-4 32B, Yi-1.5 34B, Aya Expanse 32B, Mistral Small 24B, Phi-4).
- Macro de T-gen por tamaño del generador (secundario) y su interacción en EXP-006.
- Generación: batch según la memoria libre, y a la mitad si falta memoria. Salidas con bloque de razonamiento (Gemma 4). Modelos registrados sólo como multimodales.
- transformers ≥ 5.5 desde la celda 1.2, y G1 verificada otra vez con el entorno actual antes de embeber (6.2).

**v0.3.2 (2026-10-02):**

- Panel T-gen v2 (ENMIENDA_03 §11). La v1 dejó atajos en los 16 generadores (EXP-006a), así que ahora hay tres construcciones:
  - **S:** reglas de texto;
  - **P:** el oro reescrito por el mismo generador, con largo asignado al azar;
  - **E:** negativos que editan sólo el dato clave del oro.
- Un **piloto** con regla fija elige una de las tres (`run_pilot`, `BUILD.json`).
- La evaluación y la auditoría aceptan un positivo propio por set (`pos.npy`).
- Las generaciones ya hechas se reutilizan.
- Respaldos de §3.6 (`tgenall` y el nuevo `tgenfail`) aunque el macro quede vacío.
- Exclusiones por QC leídas solas de las marcas del CSV.
- Regla fija para los brazos de B1-R2 (`top_arms_rule`).

**v0.3.3 (2026-10-05):** correcciones tras la primera corrida completa (ENMIENDA_03 §12).

- **Gemma 4:** el cálculo del batch suma la caché K/V capa por capa (`genworker.kv_bytes_per_token`). Antes leía un valor global que en Gemma 4 varía por capa, y la generación fallaba.
- **Matriz de transferencia entre ciegos sin fuga por ítem:** pliegues por ítem comunes a todos los generadores (`generators.transfer_rows`).
- **EXP-006c:** separa generador y tipo de negativo con los sets del piloto (`analysis.construction_report`; sólo evalúa corridas que ya existen, con `pipeline.eval_extra`).
- **C2 en F3:** secundario pre-especificado (`analysis.c2_report`), registrado en el PROTOCOL FREEZE.

**v0.3.4 (2026-10-05):** lista para el freeze (ENMIENDA_03 §13).

- **C2b en F3:** la descomposición de EXP-006c en el test confirmatorio. `generators.build_pilot_sets` genera los sets S, P y E del piloto en ese split, y `analysis.c2b_report` hace el análisis con Holm entre los 4 contrastes.
- **Sin re-embebidos inútiles:** la firma de las generaciones es tamaño + sha1 del contenido (`generators.raw_sigs_match`). Antes usaba el mtime, que en Drive cambia entre sesiones.

**v0.3.5 (2026-10-05):** freeze confirmado por el investigador; C2c (ENMIENDA_03 §14).

- **C2c en F3:** el mecanismo de H1b en el test confirmatorio.
  - `pipeline.F3_VARIANTS` registra en el freeze la variante "HN solos" de infonce_hn: mismos HP congelados, sólo cambia el denominador.
  - `pipeline.run_confirm_variant` la entrena.
  - `analysis.c2c_report` hace las 2 pruebas con Holm (T-gem con estados barajados sube; T-mine baja) y el control por TOST (infonce_hn − infonce en T-gem barajado).
- La variante no entra al todos contra todos, a C1, a C2/C2b ni a T-Rex.

**v0.3.6 (2026-10-06):** completar F3 después del freeze (DP-07). No cambia el entrenamiento ni la evaluación.

- **Disco en la generación** (`generators`). En la rev. 4.5 el disco local se llenó en el panel del confirmatorio aunque los pesos se borraban. Ahora:
  - antes de bajar cada generador se borran los pesos de los demás y la caché de Xet (`purge_hf_caches`);
  - el subproceso de generación corre en su propio grupo, que se mata al terminar;
  - la caché de chunks de Xet queda apagada;
  - con poco disco, `disk_diagnostics` muestra qué lo ocupa, incluidos los archivos borrados que siguen abiertos;
  - el mensaje "borrados X GB" ya no cuenta dos veces los enlaces simbólicos.
- **Adenda del freeze** (`pipeline.freeze_addendum` y `frozen_variants`). Si el freeze se hizo con un código que no registraba C2c, lo registra una sola vez en `config/PROTOCOL_FREEZE_ADDENDUM.json`, con la especificación de ENMIENDA_03 §14 y sin tocar el freeze. `run_confirm_variant` y `c2c_report` la usan y dejan constancia de la desviación.
- **T-Rex** (`pipeline.trex_for_arms`): un brazo que ya tiene sus resultados se salta sin levantar los servidores.

**v0.3.7 (2026-10-07):** arreglo del disco al generar (DP-08). No cambia el entrenamiento, la evaluación ni el análisis.

- **Causa encontrada.** huggingface_hub guarda los archivos reales en una caché compartida (`hub/blobs/<xx>/<hash>`) y en `models--X/blobs` deja enlaces simbólicos. Hasta v0.3.6 se borraban sólo los enlaces y el disco no se liberaba.
- **Arreglo** (`generators`):
  - el subproceso de generación desactiva esa caché (`HF_HUB_DISABLE_SHARED_BLOBS=1`, en `worker_env`) y no hereda `HF_HUB_CACHE`;
  - `purge_model_cache` y `purge_hf_caches` borran también los archivos de la caché compartida que ningún repo conservado usa;
  - `disk_diagnostics` muestra las cachés compartidas.
- **Sensibilidad del panel en F3** (post hoc, fuera del paquete): `analisis/F3_sensibilidad/sensibilidad_panel.py` y el notebook `CLM_losses_Analisis_F3.ipynb`, que se genera con `python build_notebook.py --analisis CLM_losses_Analisis_F3.ipynb`.

**v0.3.8 (2026-10-11):** notebook pre-envío (`CLM_losses_PreEnvio.ipynb`), fuera del protocolo. No cambia nada de lo que produjeron F0–F3: el entrenamiento, la evaluación y el análisis con las opciones existentes dan exactamente lo mismo (pruebas en `tests/test_v038.py`).

- **Denominadores nuevos de InfoNCE+HN** (`losses.InfoNCE`, sólo en estado→acción; el acción→estado sigue con in-batch). Sirven para la condición de borde de B5 (INNOVATION_MAP §2.8):
  - `hn_plus_inbatch_k`: el oro, sus HN y `extra_k` oros de otros estados (al azar en cada paso);
  - `hn_plus_otherhn`: el oro, sus HN y un HN de cada uno de los otros B − 1 estados;
  - `hn_plus_otherhn_k`: lo mismo con `extra_k` de esos HN ajenos;
  - `hn_shared_inbatch`: el oro, todos los HN del batch y los oros de los otros estados (como NegCLIP).

  El muestreo usa un generador propio sembrado con la semilla de la corrida y el paso: no toca el RNG global y la reanudación es exacta. Los HN ajenos casi idénticos al oro de la fila (coseno crudo > `dup_threshold`) quedan fuera. Un nombre desconocido da error.
- **Juez de falsos negativos** (`genworker judge`): un prompt por ítem con sus 10 candidatos, decodificación voraz, parseo estricto de "número: 0/1", un segundo intento con recordatorio del formato y reanudación. Juez por defecto: `allenai/Olmo-3.1-32B-Instruct` en bf16. gpt-oss-120b no corre en la RTX PRO 6000 (sus kernels MXFP4 no cubren sm_120 y des-cuantizado no cabe).
- **`preenvio`**: los cuatro bloques del notebook. Cada bloque escribe su regla de lectura con fecha antes de correr (`results/PRE/<bloque>/REGLA.json`) y no la reescribe.
  - **Bloque 0:** extracción de Drive (entorno, F1b frente a F3, Friedman con 7 bloques, puntuaciones secundarias, B4).
  - **Bloque 1:** calidad de los sets. Juez sobre todos los ítems sellados de T-gem, T-mine y los 5 generadores del macro, 200 ítems para etiquetar a mano (CSV ciego) y fuga entre splits (Jaccard exacto sobre 3-gramas, bloqueado por oro). Con eso, sensibilidad de C1, C2 y C2c, que sin exclusiones reproduce lo confirmatorio.
  - **Bloque 2:** 6 denominadores × 3 semillas en el split de test. Antes, chequeo de que "sólo HN" y "HN + in-batch" reproducen sus corridas de referencia.
  - **Bloque 3:** Qwen3-Embedding-8B, BM25 y coseno crudo, con estados reales y barajados.

Las corridas hechas con v0.2 conservan su huella y sólo reciben evaluaciones nuevas.

## Uso

- **Colab:** abre `CLM_losses_Colab.ipynb`. El notebook ya trae este paquete embebido y tiene una celda por paso de F0, una por brazo de F1 y de F3, y una por experimento de F2.
- **Local:** sirve para desarrollo y pruebas.

```bash
pip install torch numpy scipy pandas pyarrow huggingface_hub matplotlib pytest
python -m pytest -q tests                      # 128 pruebas, ~5 min en CPU
python -c "from clm_losses.smoke import run_smoke; run_smoke('/tmp/drive', '/tmp/cache', mount=False)"
python build_notebook.py                       # regenera el notebook y clm_losses_pkg.zip
python build_notebook.py --analisis CLM_losses_Analisis_F3.ipynb   # además, el de sensibilidad de F3
python build_notebook.py --preenvio CLM_losses_PreEnvio.ipynb     # además, el notebook pre-envío
```

## Módulos

| Módulo | Contenido |
|---|---|
| `config` | Rutas: Drive para lo que producimos y disco local para los datos públicos. Perfiles `smoke`, `mini`, `colab`, `colab_large` e hiperparámetros por defecto |
| `utils` | Guardado atómico, `DONE.json`, logs JSONL, semillas y estado de RNG, hash del código, entorno |
| `heads` | Cabezales idénticos a CLM, exportables a `clm-serve` (`clm_head.pt`) |
| `losses` | Las 15 pérdidas, el modelo ciego (F0.6) y el término auxiliar de HN (régimen R2) |
| `data` | Stores de embeddings (Hugging Face o sintéticos); splits por documento (test, val y confirm sellado); sets de evaluación; streams deterministas y reanudables |
| `train` | Entrenador por etapas (pre → mid) con checkpoints, reanudación exacta y huella de configuración |
| `evaluate` | Top-1 y MRR (desempate al azar), abstención (T-abst y T-abst-mine), calibración (TS, Platt), T-rand, T-mine, T-mine-50, panel T-gen, estados barajados y geometría; val-mine para la selección |
| `stats` | Todos contra todos con bootstrap jerárquico pareado, Holm, TOST y probabilidades ROPE; Friedman + Nemenyi; C1 |
| `hpo` | Búsqueda aleatoria reanudable con el mismo presupuesto para todos los brazos; curvas de desempeño esperado |
| `pipeline` | Funciones que llama cada celda: `run_arm`, `run_variant`, `protocol_freeze`, `run_confirm_arm`, `run_all_vs_all`, `add_missing_evals`, … |
| `f0` | Pasos y compuertas G1–G6; F0.5b (pools de val-mine y T-mine-50); F0.7b (receta de CLM vs actual, G2b y su regla) |
| `encoder`, `encoder_server` | Qwen3-8B con transformers (last-token, L2) y servidor `/v1/embeddings` compatible con CLM |
| `texts` | Tokens de Nemotron: textos, longitudes, QC manual y T-gen2 de v0.2 |
| `generators`, `genworker` | Panel T-gen: 23 generadores verificados, prompts (negativos, ediciones, reescrituras) y parseo, generación en subproceso (transformers; vLLM opcional sin probar; backend falso para pruebas), construcciones S/P/E y piloto, embebido, auditoría ciega, QC, PANEL.json y macro |
| `analysis` | EXP-006 (transferencia entre generadores, lectura pre-especificada) y uso del estado por brazo |
| `preenvio` | Notebook pre-envío (v0.3.8): extracción de Drive, calidad de los sets, condición de borde de B5 y referencias sin entrenar |
| `trex`, `deepswe` | Arneses del repo de CLM (commit `bb42c6c`) sin modificar |
| `smoke` | Flujo completo con datos sintéticos (incluye un artefacto de generador detectable) |

## Decisiones de implementación que afectan resultados (para Métodos)

1. **Splits:** se asignan por documento con `sha256(sal:doc_id)`.
   - Test (10 %), val (3 %) y confirm (10 %) salen **sólo** de documentos que tienen HN.
   - Ningún documento retenido entra en el entrenamiento. Las máscaras quedan atadas al split exacto.
   - Los cabezales públicos de CLM vieron esos pares, así que se reportan sólo como referencia contaminada.
2. **Sets de evaluación:**
   - Sólo ítems con los 10 HN.
   - Como máximo 2 ítems por documento; después, submuestreo uniforme hasta el máximo del perfil.
3. **Selección y early stopping:**
   - F1 piloto (v0.2): top-1 en el val de T-gem.
   - **F1b en adelante (v0.3, A1.2): top-1 en val-mine**, es decir, val con 10 respuestas minadas de otros documentos de val.
   - El punto de partida de cada etapa (paso 0) **no** compite. Así cada etapa entrega un modelo entrenado en ella, y una pérdida que colapsa no se reporta como la proyección aleatoria inicial.
4. **Métricas:**
   - Top-1 y MRR son valores **esperados** con desempate al azar. Una puntuación no finita cuenta como error.
   - AUPRC, AURC y ECE tampoco favorecen al oro en un empate.
5. **Calibración:**
   - Temperature scaling (listwise) y Platt **sin pesos** (pointwise; queda calibrado para la proporción real de 1 oro por 10 HN). Ambos se ajustan en val.
   - ECE con 15 bins.
6. **Abstención:**
   - Conjunto con oro = oro + 9 HN; conjunto sin oro = 10 HN (mismo tamaño).
   - Puntuaciones: coseno máximo (primaria), margen, probabilidad máxima y candidato nulo (B2).
7. **SIGReg:** idéntico a `lejepa.univariate.EppsPulley` (verificado en el código oficial): pesos 2·dt con dt en los extremos, por φ(t), por N. En MSE+SIGReg el término MSE se promedia por dimensión; esa convención está **[PENDING VERIFICATION]**.
8. **Triplet:** negativo semi-hard; si no hay ninguno, el más lejano (convención de TF).
9. **LR y gradiente:**
   - Regla de CLM: 2e-3·√(1024/ancho)·√(batch/1024).
   - OneCycleLR con los valores por defecto de torch, como en CLM (pct_start 0.1, coseno). Se garantizan ≥ 2 pasos de warm-up.
   - AdamW y clip de 1.0.
10. **Falsos negativos in-batch:** se enmascaran los pares con coseno crudo > 0.999.
11. **Estadística:**
    - Bootstrap jerárquico pareado: pesos de Poisson por documento, compartidos entre brazos **y semillas** en cada réplica; semillas remuestreadas dentro de cada brazo.
    - p con aproximación normal; Holm sobre los 105 pares de cada métrica; TOST con IC90 dentro de ±1 pp.
    - Friedman + Nemenyi con un bloque por tarea. Hay pocos bloques, así que la diferencia crítica es grande y el análisis principal son las comparaciones por pares.
12. **HPO:**
    - 32 trials aleatorios por brazo (P9; configurable en la celda 1.1); el trial 0 son los valores por defecto publicados.
    - Cada trial usa el 25 % de las filas con semilla 0. Después, 3 semillas con el 100 %.
    - El HPO no ve el test.
13. **Reanudación exacta:**
    - Checkpoint cada 200 pasos o cada 10 minutos.
    - Restaura RNG, optimizador, scheduler, estado de la pérdida (EMA, centros, prototipos) y la posición en el stream.
    - Verificado con cortes a mitad de etapa y en el borde entre etapas con aumentación aleatoria.
14. **Integridad:**
    - Cada corrida registra su tiempo de cómputo (`train_seconds`, `eval_seconds`), acumulado entre reanudaciones.
    - Cada carpeta de corrida guarda una huella de su configuración; si se pide otra, el código se detiene.
    - `study.json` queda congelado por perfil.
    - El hash del código se guarda al inicio y al final de cada corrida, y el análisis avisa si se mezclan versiones.
    - El test confirmatorio exige `PROTOCOL_FREEZE.json`.

## Revisión independiente (2026-09-30)

Un revisor independiente encontró 0 problemas críticos y 1 alto. Todos los hallazgos están corregidos o justificados, con pruebas de regresión:

- Servidor de embeddings: respondía 422 a cada petición (rompía T-Rex y G5). **Corregido.**
- Platt con pesos balanceados: daba un ECE puntual sesgado. **Corregido.**
- IC del AUROC de abstención: eran demasiado estrechos (pesos no compartidos entre semillas). **Corregido.**
- SIGReg: era la mitad de la escala de LeJEPA. **Corregido y verificado.**
- Carpetas de corridas: se reutilizaban aunque cambiara la configuración. **Corregido.**
- Otros, menores: NaN contaba como acierto; empates en AUPRC y AURC; RNG en el borde entre etapas; early stopping al reanudar; orden congelado–clipping en SwAV; regla de reserva del triplet; máscaras sin atar al split; ítems de T-gen2 sin chequear; semillas fallidas sin reportar. **Corregidos.**
- OneCycleLR con ciclo de momentum: **se mantiene a propósito**, porque es lo que usa CLM.

## Limitaciones y pendientes

- **No se ha corrido en Colab real ni con datos reales.**
  - Se probó con datos sintéticos, con un stub de `google.colab` y con el flujo completo del smoke (F0 → F1 → F2 → freeze → F3).
  - F0.1–F0.3, F0.9 y F0.10 requieren red, Hugging Face y GPU, y no se han ejecutado.
- **T-gen2:** requiere la decisión P3. El juez LLM de falsos negativos no está implementado; sólo hay QC manual.
- **No implementados:** BFCL *irrelevance* (B2), T-Rex offline y la validación cruzada de DeepSWE (depende de G6).
- **Estimaciones sin medir:** tiempos, disco y la convención de λ en MSE+SIGReg.
