# DISEÑO 01 — Decisiones D1–D6, brazos v1 y protocolo F0

**Fecha:** 2026-09-30 · **Etapa:** diseño previo a implementación. No hay código.
**Sustituye:** §3–§4 y §6 de `DIAGNOSTICO_00_CLM_perdidas.md` en todo lo que cambie. El DIAGNÓSTICO 00 se conserva sin tocar como registro de la versión 0.

**Etiquetas:** [VERIFIED], [INFERENCE], [SUPUESTO], [PENDING VERIFICATION], [AUTHOR DECISION NEEDED].

---

## 1. Decisiones registradas

| # | Propuesta | Tu respuesta | Cómo la interpreto | Qué cambia |
|---|---|---|---|---|
| D1 | Reencuadrar el estudio | "hacer los dos" | Un solo diseño con dos capas. **A:** benchmark controlado de todas las pérdidas. **B:** preguntas focalizadas (atajos de HN, abstención, mecanismo de SIGReg), evaluadas sobre **todos** los brazos de A. | Ver §2. La decisión de publicar uno o dos papers se toma en el Journal Checkpoint 2, con la regla anti–salami slicing. |
| D2 | Quitar T-Rex como métrica | "con y sin" | Mantener T-Rex como métrica secundaria y correrlo **con y sin shield**. Por si te referías a otra cosa, el ranking agregado también se reporta con y sin T-Rex incluido. | Ver §4. |
| D3 | Fusionar H3 con H1a; H4 como mecanismo | "sí" | Aprobado. | Quedan cuatro hipótesis: H1a, H1b, H2 y H4. |
| D4 | Eliminar FastSiam | "sí, eliminar; probar los demás SSL" | FastSiam sale. Se prueban todos los demás métodos SSL de la lista; alignment + uniformity pasa de opcional a núcleo. | Ver §3. Si además querías sumar métodos nuevos, hay dos con razón científica clara (§3.3); no los incluí sin tu visto bueno. |
| D5 | DINO/SwAV | "incluye los 2" | DINO y SwAV entran como dos brazos separados, adaptados cross-modalmente. | Dos brazos nuevos (14 y 15), con muchos hiperparámetros. Esto exige un control de tunability (§3.4). |
| D6 | Cabezales desde cero; puntuación cos(z_s, z_a) | "sí" | Aprobado. | Ninguno: ya estaba en el diseño. |

---

## 2. Diseño en dos capas (D1)

| Capa | Pregunta | Hipótesis / contraste | Qué se compara |
|---|---|---|---|
| **A — Benchmark controlado** | ¿Cómo cambia la familia de pérdida la discriminación entre acciones plausibles, y qué tan estable es el ranking? (RQ1, RQ4) | **H1a / C1:** mejor no contrastiva vs InfoNCE (diferencia ≥1 pp, o equivalencia ±1 pp por TOST) | Los 15 brazos, con tuning equitativo |
| **B1 — Atajos de HN** | ¿Cuánto de la ganancia por HN viene de artefactos del generador? | **H1b / C2:** caída de top-1 con HN de otro generador y con negativos minados; modelos ciegos | Todos los brazos, sobre T-gem, T-gen2, T-mine y T-rand |
| **B2 — Abstención** | ¿Qué pérdidas producen puntuaciones absolutas útiles para decir "ninguna sirve"? | **H2 / C3:** SigLIP vs el mejor mecanismo justo con InfoNCE, en AUROC de abstención | Todos los brazos, sobre T-abst y BFCL *irrelevance* |
| **B3 — Mecanismo** | ¿SIGReg evita el sobreajuste al usar HN desde el inicio? | **H4 / C4:** InfoNCE + SIGReg vs InfoNCE, con InfoNCE + logsumexp como control simple | Experimento de currículo dedicado |

**Por qué integrarlas.** Evaluar B1 y B2 sobre todo el espectro de pérdidas convierte el benchmark en ciencia: deja de ser "¿cuál gana?" y pasa a ser "¿qué propiedad de la pérdida produce qué comportamiento?".

**Publicación [AUTHOR DECISION NEEDED en Checkpoint 2].**

- Por defecto, un solo paper integrado.
- Se divide sólo si B1 produce un hallazgo con pregunta propia (validez de la evaluación con HN sintéticos) y evidencia independiente (cambio de generador, modelos ciegos).
- Dos papers que compartan datos y brazos sin preguntas distintas serían salami slicing.

---

## 3. Brazos v1 (15) y referencias sin entrenamiento

**Reglas comunes a todos los brazos:**

- Encoder Qwen3-8B congelado.
- Mismos cabezales (4096 → 1536 → 1536 → 512, con LayerNorm; a confirmar en F0.1).
- z = salida del cabezal antes de normalizar.
- Puntuación primaria cos(z_s, z_a) (D6).
- Predictores, expanders, prototipos y targets EMA existen sólo en entrenamiento.
- Todos parten de los valores por defecto publicados para su familia.

| # | Familia | Brazo | Formulación (resumen) | ¿Usa HN? | Hiperparámetros críticos |
|---|---|---|---|---|---|
| 1 | Contrastiva | InfoNCE bidireccional | Receta CLM, τ aprendida | No | τ inicial, batch |
| 2 | Contrastiva | InfoNCE + HN | s→a con HN según la fórmula del README + a→s estándar; currículo pre → mid | Sí | nº de HN, duración del mid |
| 3 | Sigmoide | SigLIP | −Σ log σ(y_ij(t·cos_ij + b)), con t y b aprendidos | No | t′ y b iniciales (log 10, −10 según el paper; confirmar) |
| 4 | Sigmoide | SigLIP + HN | Añade los pares (s_i, h_ik) con y = −1 | Sí | ídem |
| 5 | Margen | Triplet + HN | Margen sobre distancia coseno; HN + minado semi-hard in-batch | Sí | margen |
| 6 | Predictiva | SimSiam-x | ½[D(p_s(z_s), sg(z_a)) + D(p_a(z_a), sg(z_s))] | No | predictor, lr |
| 7 | Predictiva | BYOL-x | Un target EMA por torre + predictores | No | momentum EMA |
| 8 | Redundancia | Barlow Twins-x | Correlación cruzada entre expanders de s y a | No | λ fuera de la diagonal, dimensión del expander |
| 9 | Redundancia | VICReg-x | Invarianza s↔a + varianza + covarianza por torre | No | λ, μ, ν, expander |
| 10 | Distribución | MSE + SIGReg | (1−λ)‖z_s − z_a‖² + λ·½[SIGReg(z_s) + SIGReg(z_a)] | No | λ, nº de proyecciones (el estadístico escala con N) |
| 11 | Distribución | InfoNCE + SIGReg | InfoNCE sobre z normalizado + SIGReg sobre z sin normalizar | No | λ |
| 12 | Puente | Alignment + uniformity | ‖ẑ_s − ẑ_a‖² + γ/2·[U(ẑ_s) + U(ẑ_a)], con U = log E e^{−2‖ẑ_i − ẑ_j‖²} | No | γ |
| 13 | Control | InfoNCE + logsumexp | L_InfoNCE + λ·mean_i(logsumexp_j ℓ_ij)² | No | λ |
| 14 | Prototipos | **DINO-x** (nuevo) | Ver §3.1 | No | K, τ_t, τ_s, centering, EMA |
| 15 | Prototipos | **SwAV-x** (nuevo) | Ver §3.2 | No | K, ε, τ, cola, congelado de prototipos |

**Pool de "no contrastivas" para C1 (H1a):** brazos 6, 7, 8, 9, 10, 14 y 15. La mejor se elige **en validación**.

El brazo 12 queda fuera del pool: su término de uniformidad es una repulsión entre instancias, así que sirve de puente, no de representante "sin negativos". Lo que prueba es si importa el **acoplamiento** positivo–negativos de InfoNCE o basta con términos desacoplados.

**Referencias sin entrenamiento (baselines):**

- coseno crudo (clm-raw);
- Qwen3-Embedding-8B zero-shot;
- BM25 (en QA);
- cabezales públicos de CLM, sólo fuera de Nemotron;
- modelos ciegos (§5, F0.6);
- heurística de longitud;
- azar (1/11 = 9.1 %).

### 3.1 DINO-x (autodestilación cross-modal)

**Estudiante:**
- z_s = g_s(e_s) y z_a = g_a(e_a).
- Un cabezal de prototipos **compartido** h: MLP 512 → 2048 → 256, normalización L2, y una capa lineal con weight-norm hacia K prototipos.
- P_est(x) = softmax(h(z)/τ_s), con τ_s = 0.1.

**Maestro:**
- EMA de (g_s, g_a, h), con momentum 0.996 → 1.
- P_mae(x) = softmax((h_t(z) − c)/τ_t), con τ_t = 0.04 → 0.07 (warm-up).
- Centro c actualizado por EMA con m = 0.9.

**Pérdida:** L = ½[CE(P_mae(a), P_est(s)) + CE(P_mae(s), P_est(a))]. El par estado–acción hace de las dos vistas, sin multi-crop.

**Precedente más cercano:** nCLIP (CVPR 2023), con prototipos compartidos entre modalidades.

**Riesgo específico:** un objetivo de clustering puede ser demasiado grueso para separar 1 oro de 10 HN casi idénticos. Es una pregunta empírica, no un motivo para excluirlo.

**Puntuación secundaria:** similitud entre las distribuciones de prototipos. Se declara de antemano.

### 3.2 SwAV-x (predicción intercambiada cross-modal)

**Prototipos:** C, de K × 512, con filas normalizadas y compartido por las dos torres.

**Códigos:**
- q = Sinkhorn(exp(ẑCᵀ/ε)), con 3 iteraciones, ε = 0.05 y sin gradiente.
- Cola de embeddings pasados cuando el batch es menor que K.

**Pérdida:**
- L = ½[ℓ(ẑ_s, q_a) + ℓ(ẑ_a, q_s)], donde ℓ(ẑ, q) = −Σ_k q_k log softmax(ẑCᵀ/τ)_k, con τ = 0.1.
- C se congela durante la primera época.

**Precedente:** SwALIP (Fini et al., 2023), donde SwAV entraba como término auxiliar de CLIP. Aquí se usa sola.

### 3.3 Otros SSL posibles (no incluidos sin tu visto bueno)

- **Spectral Contrastive Loss (HaoChen et al., 2021)** [PENDING VERIFICATION]. Contrastiva sin softmax, con similitudes de escala absoluta. Es útil para H2.
- **DCL (Decoupled Contrastive Learning, Yeh et al., 2022)** [PENDING VERIFICATION]. Quita el positivo del denominador y reduce la sensibilidad al batch. Es útil para RQ4.

Cada brazo extra añade ~7 % de cómputo en F1–F3 y un contraste más en el análisis exploratorio.

### 3.4 Control de tunability (nuevo, por D5)

Con el mismo número de trials, los brazos con muchos hiperparámetros (DINO-x, SwAV-x, VICReg, Barlow) pueden quedar sub-tuneados; es la injusticia opuesta a la que queríamos evitar. Controles:

- Todos parten de los valores por defecto publicados.
- Se reportan curvas de **desempeño de validación esperado vs presupuesto de tuning** para cada brazo (Dodge et al., 2019, "Show Your Work") [PENDING VERIFICATION].
- Si el ranking cambia entre 8, 16 y 32 trials, se reporta como resultado.

### 3.5 Por qué InfoNCE + logsumexp tiene doble papel

1. **Control simple para H4:** un regularizador más barato que SIGReg.
2. **Candidato justo de abstención para H2** [hipótesis nuestra, a probar]. Penalizar la log-partición por consulta podría anclar la escala absoluta de los logits, y así darle a InfoNCE un umbral global sin cambiar de pérdida.

---

## 4. T-Rex con y sin shield (D2)

**Advertencia.** Aun sin shield, T-Rex mide **seguir etiquetas**, no decidir: las opciones llegan marcadas por el planner ("Safe. … Best."). Por eso es métrica secundaria.

| Modo | Qué mide | Métricas | Cuándo |
|---|---|---|---|
| Online **con shield** (como en upstream) | El sistema combinado; comparable con CLM | Acuerdo con el planner, supervivencia, intervenciones (elecciones "Unsafe" reemplazadas + correcciones al llegar) | Checkpoint final de cada brazo; 5 cursos × 60 s |
| Online **sin shield** (`--no-shield`) | El modelo solo | Acuerdo con el planner, muertes, puntuación | Ídem |
| Offline (propuesta adicional) | Seguir etiquetas, sin ruido de tiempo | Acuerdo con el planner sobre un set fijo de estados grabados | Todas las semillas de todos los brazos; barato y determinista |

**Métrica principal de T-Rex:** acuerdo con la mejor acción del planner.

**Chequeos de sanidad:**
- La latencia debe ser idéntica entre brazos, porque tienen la misma arquitectura.
- `host_stall_seconds_dropped` debe quedar ≈ 0.

**Referencia del repo:** acuerdo de 0.658 con shield (CLM) y 0.987 (Jev).

---

## 5. Protocolo F0 — compuerta antes de cualquier comparación

**Objetivo.** Demostrar que (i) medimos lo mismo que CLM, (ii) el test principal no es hackeable y (iii) cada pérdida está implementada sin bugs. Nada de F1–F3 arranca sin pasar F0.

**Datos verificados que usa F0** [VERIFIED vía el visor de HF]:

- `qa_pairs` tiene las columnas `doc_id` (UUID), `pair_idx` (índice **dentro** del documento), `question_ids`, `answer_ids` (token IDs de Qwen3, sin tokens especiales) y los conteos de tokens.
- `neg_pairs` tiene `doc_id`, `pair_idx`, `neg_idx` y `neg_ids`.
- La clave de un par es (`doc_id`, `pair_idx`).
- Las respuestas y los negativos son muy cortos (2–8 tokens en las filas vistas), así que la heurística de longitud es una sonda de atajo necesaria.

| Paso | Qué | Por qué | Salida | Criterio (propuesta; a fijar) |
|---|---|---|---|---|
| **F0.1** Auditoría de checkpoints | Cargar los cabezales CLM pre, mid y post y el de DeepSWE: `cfg`, parámetros, `logit_scale` | Fijar la arquitectura común (D6) y la temperatura de referencia | `F0_checkpoints.json` | `cfg` documentado; si difiere de la inferida, se adopta el real para todos los brazos |
| **F0.2** Manifiesto de datos | IDs, revisiones de HF, tamaños, SHA-256, licencias y fecha de descarga de cada artefacto | Artefactos en cuentas personales pueden desaparecer o cambiar; gobernanza | `DATA_MANIFEST.md` | Completo antes de F0.3 |
| **F0.3** Equivalencia de embeddings | Re-embeber 2,000 pares a partir de sus **token IDs exactos** (vLLM pooling, last-token, L2, fp16) y comparar con los embeddings públicos por (`doc_id`, `pair_idx`) | Necesitamos embeber negativos nuevos (T-gen2, T-mine) con el mismo pipeline | Informe de cosenos | **G1:** mediana ≥ 0.999 y p01 ≥ 0.995. Si falla, probar variantes (EOS, dtype, versión de vLLM) |
| **F0.4** Splits por documento | Hash de `doc_id` → 98 % train / 1 % val / 1 % test. Excluir de val/test los documentos del prefijo de HN (3M pares) y los duplicados exactos de pregunta o respuesta normalizada; MinHash en una muestra | Evitar fuga entre pares del mismo documento o de documentos casi idénticos | Archivos de split + informe de fuga | Fuga estimada < 0.5 % de ítems de test |
| **F0.5** Sets de test | **T-gem:** 100K preguntas (≤2 por documento), 1 + 10 HN de Gemini. **T-abst:** conjuntos *con* oro (oro + 9 HN) vs *sin* oro (10 HN), del **mismo tamaño**. **T-gen2:** 10K con 10 HN de otro generador (familia distinta de Gemini y Qwen) con nuestro prompt; opcionalmente, Gemini con nuestro prompt, para separar generador de prompt. **T-mine:** 10K con 10 respuestas minadas de otros documentos. **T-rand:** 10K aleatorias | B1 y B2 necesitan negativos de distinto origen; igualar tamaños evita un confusor en la abstención | Sets embebidos + fichas | En los sets generados o minados, falsos negativos < 5 % (juez LLM + 200 ítems revisados a mano) |
| **F0.6** Sondas de atajo | (a) Modelo **ciego**: cabezal sólo sobre la acción, entrenado con HN de Gemini. (b) Heurística de longitud. Ambas se evalúan en T-gem, T-gen2, T-mine y T-rand | Medir cuánto se puede acertar sin mirar el estado | Tabla de top-1 | **G3** sobre T-gem: <15 % despreciable; 15–30 % se mantiene T-gem y se reportan métricas ajustadas; >30 % la métrica primaria pasa a T-gen2 + T-mine y H1b se vuelve central |
| **F0.7** Reproducción de InfoNCE | Pre / pre → mid / HN desde el inicio, 2 semillas cada uno. El mid se prueba en dos variantes: HN solo en el denominador s→a (como está escrito en el README) y HN + in-batch. Los cabezales públicos de CLM se evalúan aparte como referencia contaminada | Si no reproducimos el baseline, ninguna comparación vale | Curvas + tabla | **G2:** 52.1 ± 3 pp y 69.2 ± 3 pp en T-gem, y el orden mid > pico de HN desde el inicio > pre. Exige datos completos (62M); con un subconjunto, G2 baja a cualitativo |
| **F0.8** Smoke tests | Brazos 6–10, 12, 14 y 15 sobre 1M pares, con valores por defecto, con y sin aumentación en espacio de embedding. Tests unitarios de cada pérdida: InfoNCE ≈ log B al inicio, SIGReg ≈ 0 con muestras gaussianas, marginales de Sinkhorn | Distinguir colapso por bug de colapso del método | Informe de colapso | **G4:** colapso = RankMe < 26 (5 % de 512) o top-1 en T-rand a ≤2 pp del azar. Si hay colapso, reproducirlo en 2 semillas y explicarlo antes de seguir |
| **F0.9** Arnés de T-Rex | Cabezal público de CLM, con y sin shield, 5 cursos | Validar el arnés antes de usarlo con los 15 brazos | JSON por modo | **G5:** con shield, acuerdo en 0.658 ± 0.05, 5/5 sobrevive y `host_stall` ≈ 0 |
| **F0.10** Identidad de DeepSWE | `bon_eval.py` con el cabezal público de DeepSWE sobre `tarsur385/deepswe-prm-embeddings-8k` y `heldout_tasks.json` (n = 4, window = 12) | Confirmar si esos embeddings son el set de evaluación de CLM | Resultado de `bon_eval` | **G6:** reproduce exactamente 31/38. Si pasa, se habilita la CV de 3 folds sobre 113 tareas; si no, DeepSWE queda exploratorio |

**Si una compuerta falla.** Ningún fallo "se arregla" ajustando umbrales después de ver el resultado. Se registra en `EXPERIMENT_LOG.md`, se diagnostica, y el diseño cambia de forma explícita. Si G3 falla, eso **es** un hallazgo: B1 pasa a ser la contribución central.

**Costo de F0 [SUPUESTO; se mide en F0]:**
- Embeber ~1.5M textos: 1–3 GPU-h.
- Generar 100K negativos con un LLM abierto: 1–4 GPU-h. Si se usa Gemini, además el costo de API.
- Reproducción de InfoNCE: 6–40 GPU-h, según escala e I/O.
- Smoke tests: 2–4 GPU-h.
- T-Rex: ~20 min de reloj por modo.
- Almacenamiento: ~0.5 TB con subconjuntos y ~1.5 TB con datos completos.

---

## 6. Cómputo total actualizado [SUPUESTO]

| Fase | Contenido | Orden de magnitud |
|---|---|---|
| F0 | §5 | 10–50 GPU-h |
| F1 | 15 brazos × ~24 trials cortos, 3 semillas del mejor | 40–70 GPU-h |
| F2 | Batch, energía, R0/R1/R2, currículo (B3), escala, sobre top-k y representantes | 40–60 GPU-h |
| F3 | 15 brazos × 5 semillas con datos completos + evaluaciones (T-Rex online incluido) | 60–100 GPU-h |
| **Total** | | **~150–280 GPU-h**, dominado por I/O |

Se apoya en una cifra secundaria no verificada ("~1 h de pre-training completo en una RTX 4090").

---

## 7. Prompt de Deep Research v2 (sustituye §6 del DIAGNÓSTICO 00)

```
ROLE: You are a systematic literature scout for an ML study. Prioritize primary sources (arXiv, OpenReview, ACL Anthology, PMLR, CVF, NeurIPS proceedings, official GitHub/Hugging Face pages). Cover work up to today, including 2026 preprints.

STUDY: A controlled comparison of 15 training losses for two-tower state→action scorers built on a FROZEN LLM encoder, as in "Contrastive Language Models" (CLM-8B, Kwok et al., Sept 2026: frozen Qwen3-8B, last-token pooling, two small projection heads, bidirectional InfoNCE, then LLM-generated hard negatives). Losses: InfoNCE (± hard negatives), SigLIP sigmoid (± hard negatives), triplet/margin, cross-modal SimSiam and BYOL, cross-modal Barlow Twins and VICReg, MSE + SIGReg (LeJEPA) and InfoNCE + SIGReg, alignment + uniformity (Wang & Isola), InfoNCE + a logsumexp (log-partition) regularizer, and cross-modal DINO and SwAV (shared prototypes; the state–action pair acts as the two views, no multi-crop). Endpoints: top-1 among 1 gold + 10 hard negatives; robustness to the negatives' generator (cross-generator and mined negatives, blind action-only baselines); abstention when no candidate is correct (AUROC); calibration (ECE); transfer to agentic action selection and best-of-N verification; embedding geometry.

QUESTIONS (answer each separately; look for evidence FOR and AGAINST):
Q1. Has anyone compared contrastive vs non-contrastive losses (BYOL/SimSiam/Barlow/VICReg/SIGReg/DINO/SwAV) for heads on frozen LLM or text embeddings, especially for decision, verification, reranking, tool/action retrieval or agent action scoring? Include follow-ups to CLM and anything on TypeSafe's "Jev" or other "System One" decision models.
Q2. Sigmoid (pointwise/pairwise) vs softmax (InfoNCE/listwise) losses: results in frozen-encoder alignment (SAIL and ShareLock disagree), calibration, comparability of scores across queries, log-partition/logsumexp regularization and absolute score calibration, and detection of "no relevant candidate / none of the above / irrelevance" (dense retrieval, tool retrieval, BFCL irrelevance, selective prediction, OOD detection with CLIP-like scores).
Q3. Non-contrastive cross-modal learning with ASYMMETRIC towers (different encoders or heads per modality): collapse, required within-modality terms, embedding-space augmentations (noise, mixup), results vs contrastive baselines.
Q4. Prototype/self-distillation objectives (SwAV, DINO, nCLIP, SwALIP) used cross-modally without multi-crop: performance, collapse, number of prototypes, and whether clustering objectives hurt fine-grained discrimination among near-duplicate candidates.
Q5. SIGReg/LeJEPA and isotropy regularizers applied to text, LLM embeddings or multimodal alignment; InfoNCE + SIGReg hybrids; evidence that isotropy or effective rank does or does not predict downstream performance.
Q6. LLM-generated hard negatives: shortcut/artifact learning (generator-style cues, length cues), curriculum effects (hard negatives early vs late), false negatives, and evaluation practices that control for them (blind or hypothesis-only baselines, cross-generator negatives, mined negatives).
Q7. Methodology for comparing many training objectives fairly: equal hyperparameter budgets vs tunability (expected validation performance curves), batch-size sensitivity, energy function (cosine vs dot vs L2), and statistics for many methods × tasks × seeds (Friedman/Nemenyi, Bayesian hierarchical tests, rliable/IQM, TOST equivalence).
Q8. Benchmarks for candidate-action selection by agents that include abstention/irrelevance or long-horizon verification (BFCL v4, ToolRet, Terminal-Bench 2.x, DeepSWE by Datacurve, ADP, WikiRace), and their known ceiling effects.

KNOWN ITEMS (verify metadata and extract the relevant result; do not stop at these): Fini et al., TMLR 2023 (Improved Baselines for Vision-Language Pre-training; includes SwALIP, BarLIP, SiamLIP); Garrido et al., ICLR 2023 (duality between contrastive and non-contrastive SSL); Wang & Isola, ICML 2020; DirectAU (KDD 2022); SigLIP (ICCV 2023); SAIL (CVPR 2025); ShareLock (arXiv 2410.07173); FuseMix (CVPR 2024); LiT (CVPR 2022); nCLIP (CVPR 2023); SwAV (NeurIPS 2020); DINO (ICCV 2021); CLIPin (arXiv 2508.06434); SLAP (ISMIR 2025); NOVA (arXiv 2602.00653); LeJEPA (arXiv 2511.08544); LeVLJEPA (arXiv 2607.00784); AllMusicCaps (arXiv 2608.25244); Eysenbach et al., NeurIPS 2022 (contrastive RL; logsumexp regularizer); JaxGCRL (ICLR 2025); Sun et al., ICLR 2025 (Rethinking Bradley-Terry models); Language Ranker (NeurIPS 2025); CWM (arXiv 2602.22452); SugarCrepe (NeurIPS 2023); arXiv 2606.01304 (hard-negative shortcuts in retrieval); Yan et al., KDD 2022 (scale calibration of ranking models); Dodge et al., EMNLP 2019 (Show Your Work); HaoChen et al., NeurIPS 2021 (spectral contrastive loss); Yeh et al., ECCV 2022 (decoupled contrastive learning); Qwen3 Embedding (arXiv 2506.05176).

OUTPUT FORMAT:
1) One table per question: Claim | Source (title, authors, venue, year, arXiv ID/DOI, URL) | Exact passage or table supporting the claim | Setting (frozen? modalities? scale?) | Direction (supports / contradicts our hypothesis) | Confidence.
2) "Closest prior work" (max 5), each with the exact difference from our study.
3) "Evidence against our hypotheses": H1a the best loss without instance-level negatives is worse than InfoNCE on top-1; H1b part of the hard-negative gain is a generator shortcut; H2 sigmoid scoring enables better abstention than the best fair InfoNCE mechanism; H4 SIGReg reduces the overfitting seen when hard negatives are used from the start.
4) Papers that must be read in full, and why.
Rules: never invent titles, authors, IDs or numbers; mark anything you could not open as UNVERIFIED; separate primary from secondary sources; include negative and null results.
```

---

## 8. Decisiones pendientes

| # | Decisión | Propuesta por defecto |
|---|---|---|
| P1 | **Recursos:** GPU, almacenamiento y tiempo | GPU ≥ 24 GB; ~0.5 TB para F0 y ~1.5 TB para datos completos |
| P2 | **Luz verde para programar F0** | — |
| P3 | Segundo generador de HN; ¿usar la API de Gemini con nuestro prompt? | Un LLM abierto de familia distinta de Gemini y Qwen; Gemini con nuestro prompt si hay acceso |
| P4 | Umbrales de G1–G6 | Los de §5, fijados **antes** de correr F0 |
| P5 | Escala de datos: completos o subconjunto | Completos para G2; si no caben, G2 queda cualitativo |
| P6 | ¿Uno o dos papers? | Uno; se revisa en Checkpoint 2 |
| P7 | ¿Sumar Spectral Contrastive y DCL? | No, salvo que el Deep Research dé razones |
| P8 | Venue y participación de un estudiante | — |
