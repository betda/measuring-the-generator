# ENMIENDA 03 — A1 (aprobada) y panel de generadores T-gen (P3)

**Fecha:** 2026-10-01 · **Estado:** diseño pre-especificado, escrito **antes** de correr nada con el código v0.3. **§11 (2026-10-02):** construcción del panel, después de la auditoría de la v1 y antes de mirar EXP-006. **§12 (2026-10-05):** correcciones tras la corrida de rev. 4.2, EXP-006c y C2 en F3, antes del freeze. **§13 (2026-10-05):** C2b y firma por contenido, antes del freeze. **§14 (2026-10-05):** C2c y freeze confirmado por el investigador, antes de congelar. **§15 (2026-10-06):** adenda del freeze para C2c y cómo se completa F3, después del freeze. **§15.5 (2026-10-07):** cómo se completó F3.
**Aprobación:** el investigador aprobó A1 completa (A1.1–A1.7) y pidió usar "todos los generadores que podamos" para T-gen (2026-10-01).
**Complementa:** `DISENO_01` (diseño), `IMPLEMENTACION_02` (código) y `PROJECT_STATE` v7.

**Etiquetas:** [VERIFIED], [INFERENCE], [SUPUESTO], [PENDING VERIFICATION], [AUTHOR DECISION NEEDED].

Todo lo que sigue se fija ahora. Si algo se cambia después de ver resultados, esa parte pasa a ser exploratoria y se registra.

---

## 1. Hecho nuevo que cambia el diseño de T-gen

La tarjeta de `nvidia/Nemotron-CC-v2.1` atribuye el subset **High-Quality-DQA** (los pares pregunta–oro que usa CLM) a **Qwen3-30B-A3B** [VERIFIED, tarjeta del dataset, 2026-10-01].

**Consecuencia [INFERENCE]:** el oro está escrito con el estilo de Qwen3 y los HN de T-gem con el de Gemini. Parte del atajo de T-gem (modelo ciego 58 %) puede ser "reconocer el estilo del oro", no algo propio de Gemini. Si es así, **cualquier** generador de otra familia también deja un atajo, distinto. Por eso:

- cada generador se audita por separado (§3.5);
- se agregan controles de la familia del oro: Qwen2.5-7B y el mismo Qwen3-30B-A3B;
- el macro de T-gen sólo incluye generadores sin atajo alto (§3.6).

## 2. A1 — qué se implementa (código v0.3)

| # | Punto | Implementación |
|---|---|---|
| A1.1 | Métrica primaria | Top-1 en **T-mine**. **T-gen (macro del panel)** es co-primaria en C1, con Holm entre las dos. T-gem queda como secundaria, siempre con su control de estados barajados. |
| A1.2 | Selección | **val-mine**: ítems de val con 10 respuestas minadas de otros documentos de val (espacio crudo del encoder, coseno ≤ 0.95; igual que T-mine). Rige el ranking del HPO, el early stopping, la elección de checkpoint y el brazo no contrastivo de C1. T-gem-val se sigue registrando, sólo como diagnóstico. |
| A1.3 | Abstención | **T-abst-mine**: con oro = oro + 9 minadas; sin oro = 10 minadas (mismo tamaño), sobre el pool de T-mine. También por generador del panel. El T-abst de Gemini queda como secundario. |
| A1.4 | Uso del estado | Toda evaluación calcula también el top-1 con **estados barajados**, en T-gem, T-rand, T-mine, T-mine-50 y cada T-gen. Permutación fija: `np.random.default_rng(0).permutation(N_test)`, la misma de las celdas 5.4 y 5.5. |
| A1.5 | Receta | F0.7b (§4) compara la receta actual con la real de CLM. La regla de decisión está fijada en §4. |
| A1.6 | F1b | Los 15 brazos con el código v0.3, de punta a punta con una sola versión (§5). El F1 piloto se conserva y se reporta. |
| A1.7 | Denominador | InfoNCE+HN y las variantes InfoNCE con HN (B2, B3) usan **"HN + in-batch"**. "HN solos" sólo aparece como condición explícita de mecanismo (F0.7b y F2b-B1), nunca como opción del HPO. |

**Adición menor (secundaria, no cambia ninguna regla): T-mine-50.** Son 50 respuestas minadas (top-1 entre 51). Responde al riesgo de techo de T-mine (los mejores brazos están en 0.95–0.96).

## 3. Panel de generadores (T-gen; reemplaza a P3)

### 3.1 Generadores [VERIFIED en la API de HF el 2026-10-01: repo, gated, licencia, arquitectura, parámetros]

*Ampliado a 23 generadores el mismo día, antes de generar nada: ver §10.*

| Clave | Modelo | Familia | Rol | Acceso | Licencia | Parámetros |
|---|---|---|---|---|---|---|
| mistral7b | mistralai/Mistral-7B-Instruct-v0.3 | Mistral AI | primario | abierto | apache-2.0 | 7.2B |
| phi4mini | microsoft/Phi-4-mini-instruct | Microsoft | primario | abierto | mit | 3.8B |
| olmo2_7b | allenai/OLMo-2-1124-7B-Instruct | AI2 | primario | abierto | apache-2.0 | 7.3B |
| granite33_8b | ibm-granite/granite-3.3-8b-instruct | IBM | primario | abierto | apache-2.0 | 8.2B |
| falcon3_7b | tiiuae/Falcon3-7B-Instruct | TII | primario | abierto | other (Falcon) | 7.5B |
| yi15_9b | 01-ai/Yi-1.5-9B-Chat | 01.AI | primario | abierto | apache-2.0 | 8.8B |
| glm4_9b | zai-org/glm-4-9b-chat-hf | Zhipu / Z.ai | primario | abierto | other (glm-4) | 9.4B |
| smollm3_3b | HuggingFaceTB/SmolLM3-3B | Hugging Face | primario | abierto | apache-2.0 | 3.1B |
| apertus8b | swiss-ai/Apertus-8B-Instruct-2509 | Swiss AI | primario | abierto | apache-2.0 | 8.1B |
| lfm2_2b6 | LiquidAI/LFM2-2.6B | Liquid AI | primario | abierto | other (lfm1.0) | 2.6B |
| gptoss20b | openai/gpt-oss-20b | OpenAI | primario | abierto | apache-2.0 | 21B MoE (MXFP4) |
| llama31_8b | meta-llama/Llama-3.1-8B-Instruct | Meta | primario | **gated** | llama3.1 | 8.0B |
| aya8b | CohereLabs/aya-expanse-8b | Cohere | primario | **gated** | cc-by-nc-4.0 | 8.0B |
| gemma2_9b | google/gemma-2-9b-it | Google (≈ Gemini) | control Gemini | **gated** | gemma | 9.2B |
| qwen25_7b | Qwen/Qwen2.5-7B-Instruct | Alibaba (familia del oro y del encoder) | control Qwen | abierto | apache-2.0 | 7.6B |
| qwen3_30b | Qwen/Qwen3-30B-A3B | **el modelo que escribió el oro** | control oro | abierto | apache-2.0 | 30.5B MoE |

- **Gated:** requieren `HF_TOKEN` y aceptar la licencia en Hugging Face. Sin acceso, el notebook los salta y lo dice.
- **Términos de las licencias "other" y cc-by-nc:** [PENDING VERIFICATION]; uso de investigación no comercial [AUTHOR DECISION NEEDED antes de publicar los negativos].
- **Modo de razonamiento:** SmolLM3 y Qwen3 se usan sin modo de pensamiento (`enable_thinking=False`). gpt-oss se usa con razonamiento bajo y se guarda sólo su canal final.

### 3.2 Ítems

- **2,000 ítems del test exploratorio**, como máximo 1 por documento, elegidos al azar (semilla 0).
- Los mismos ítems y el mismo orden para todos los generadores.
- Los controles pesados (qwen3_30b) usan los primeros 1,000.
- El test confirmatorio tendrá su propio panel, generado recién después del freeze (Sección 8 del notebook).

### 3.3 Prompt y muestreo (idénticos para todos)

```
You are helping build a benchmark of hard incorrect answers for question answering.

Question: {question}
Correct answer: {answer}

Write {k} different answers to the question that are plausible and on-topic but factually INCORRECT.
Rules:
- Each answer must have about the same length as the correct answer (around {n_words} words) and the same style and format.
- Do not copy, paraphrase, or simply negate the correct answer; each one must state something wrong.
- Do not say that the answers are incorrect and do not add explanations.
Return exactly {k} lines, one answer per line, with no numbering, bullets, or extra text.
```

- Se piden k = 14 (10 + 4 de reserva).
- temperature 0.8, top-p 0.95, bf16 (gpt-oss: sus pesos MXFP4 se des-cuantizan a bf16; ver §9).
- Backend transformers en un subproceso, un generador a la vez.
- Hasta 2 intentos por ítem.

### 3.4 Filtros y embebido

1. **Parseo:** una respuesta por línea. Se quitan numeración, viñetas, comillas y preámbulos ("Here are…"). Se descartan las vacías y los duplicados exactos del oro o entre sí.
2. **Embebido** con Qwen3-8B y nuestro pipeline: G1 lo validó en respuestas y negativos; sólo falló en estados.
3. **Falsos negativos:** se descarta toda respuesta con coseno crudo al oro > 0.95, el mismo umbral de T-mine. *(Reemplazado en §11: el filtro descartaba las respuestas que imitaban el formato del oro y no atrapaba los falsos negativos reales; ahora el coseno sólo se reporta.)*
4. Se guardan las primeras 10 válidas. Un ítem con menos de 10 queda fuera **de ese generador**. *(§11: ahora, las 10 de largo más parecido.)*

### 3.5 Auditoría por generador (no usa ningún modelo entrenado)

- **Heurísticas de longitud:** elegir la más larga o la más corta, en tokens de Qwen3.
- **Ciego de Gemini transferido:** el modelo ciego de F0.6, entrenado con los HN de Gemini.
- **Ciego propio:** validación cruzada de 5 pliegues sobre los ítems del generador.
- **Coseno crudo** (línea base sin entrenamiento).
- **Matriz de transferencia entre ciegos:** el ciego entrenado con el generador A, evaluado en B.
  - Si los ciegos de todos los generadores no-Qwen se transfieren entre sí y fallan en los controles Qwen, el atajo es el estilo del oro (§1).
- **Atajo ciego** = el máximo de las cuatro sondas. Bandas de G3: < 15 % despreciable; 15–30 % moderado; > 30 % alto.
- **QC manual:** CSV de 20 ítems × 10 negativos por generador. El criterio de DISENO_01 es < 5 % de falsos negativos [pendiente de revisión humana].

### 3.6 Macro de T-gen (regla fija; no depende de ningún resultado de modelos)

**Generadores incluidos:** cumplen todas estas condiciones:

- rol primario;
- generación **terminada** (no interrumpida);
- embebido y auditoría hechos;
- **cobertura ≥ 80 %**: ítems con 10 negativos válidos, sobre los pedidos;
- atajo ciego no alto;
- sin exclusión por el QC manual (> 5 % de falsos negativos; se registra en PANEL.json).

Se toma la intersección de sus ítems.

**Cómo se calcula:** por ítem, el promedio de acierto sobre los generadores incluidos. El bootstrap es por documento, como el resto.

**Dónde queda registrado:** `data/eval/tgen/test/PANEL.json`, con fecha, hash y lista. Si se agregan generadores, el panel cambia de versión y el análisis indica cuál usó.

**Robustez mínima:** ≥ 5 generadores en el macro. Si quedan menos:

- se reporta igualmente el macro sobre todos los primarios, marcado "con atajo" (`tgenall`: las mismas condiciones salvo la del atajo);
- se reportan los controles Qwen;
- se reporta el subconjunto "el ciego propio falla" de cada generador.

**Extensión condicionada [AUTHOR DECISION NEEDED si ocurre]:** si la mayoría de los primarios tiene atajo alto, se propone T-gen con **oro parafraseado por el mismo generador**, para igualar el estilo. *Ocurrió en la v1 (16 de 16, EXP-006a). Se activó el 2026-10-02 como la construcción P, que compite en un piloto con regla fija junto con S y E (§11).*

## 4. F0.7b — receta (A1.5)

**Recetas:**

| | Actual (F1 piloto) | CLM real (de los `cfg` de sus checkpoints [VERIFIED]) |
|---|---|---|
| Batch pre / mid | 2,048 / 512 | 16,384 / 16,384 |
| LR | OneCycle; regla de CLM por batch | **Constante**: pre 6.53e-3 (regla de CLM a 16,384); mid 1e-3 |
| Weight decay | 0 | 0 |
| Épocas (tope) | pre 2 · mid 1 | pre 4 · mid 8 **[SUPUESTO: CLM no publica cuántas; con 294K filas con HN, una época de mid a 16,384 son 18 pasos]** |
| Evaluación / paciencia | cada 100 pasos / 8 | 20 evaluaciones por etapa / 8 |

**Condiciones:** las mismas en ambas recetas, todas con selección por val-mine y semillas 0 y 1.

- **pre**: sólo pre-training.
- **mid_inbatch**: mid con HN + in-batch, desde el checkpoint de pre (elegido por val-mine) de la misma semilla.
- **start_inbatch**: HN + in-batch desde el inicio, con el mismo número de muestras que pre + mid.
- **mid_hnonly**: mid con HN solos. Es referencia de mecanismo y **no entra en la regla**.

**G2b (orden de CLM medido en T-mine, test):** media de semillas en el orden mid_inbatch > start_inbatch > pre.

**Regla (pre-especificada):** F1b usa la receta de CLM **si y sólo si** G2b se cumple con ella. Si no, usa la actual y se documenta la desviación.

- Para decidir hacen falta las 2 semillas de las 3 condiciones; si falta alguna, no hay decisión.
- Se reportan, sin que cambien la regla: los IC por bootstrap pareado de cada paso del orden; T-gem y T-gen como contexto.

## 5. F1b (A1.6)

| Elemento | Valor |
|---|---|
| Brazos | Los 15 brazos |
| HPO | 32 trials al 25 % del presupuesto; trial 0 = valores por defecto; mismo muestreador. InfoNCE+HN sin la opción de denominador (A1.7). |
| Semillas | 3 |
| Receta | La que salga de §4 |
| Selección | val-mine |

**Evaluación:**

- T-mine (primaria);
- T-gen: macro y por generador, con abstención;
- T-rand;
- T-gem;
- T-mine-50;
- T-abst-mine;
- estados barajados en todos;
- calibración y geometría.

**C1:** la mejor no contrastiva (por val-mine en el HPO de F1b) menos InfoNCE, en T-mine y en el macro de T-gen, con Holm entre las dos. T-gem se reporta como secundaria.

**F2b:**

- **B3:** InfoNCE, +SIGReg y +LSE con HN desde el inicio, con HN + in-batch.
- **B2:** candidato "ninguna".
- **B1:** HN solos como condición explícita; R2 auxiliar.

Todas con los hiperparámetros de F1b. Freeze y F3 parten de F1b.

## 6. Análisis pre-especificados (exploratorios hasta F3)

### EXP-006 — transferencia, sin entrenar nada

T-gen sobre los checkpoints ya entrenados de F0.7 y del F1 piloto.

**Contrastes:**

- F0.7: HN + in-batch − pre; HN solos − HN + in-batch; HN desde el inicio − pre.
- F1 piloto: infonce_hn − infonce; siglip_hn − siglip.

**Cobertura:** cada contraste se calcula en el macro, por generador y con estados barajados.

**Lectura fijada:**

- "Transfiere": Δ(T-gen macro) > +1 pp con IC95 que excluye 0.
- "Empeora": Δ < −1 pp con IC95 que excluye 0.
- "No transfiere": equivalente por TOST ±1 pp.
- Si no se da ninguna de las tres: no concluyente.
- Se reporta además la fracción transferida, Δ(T-gen) / Δ(T-gem), sólo cuando los HN suben T-gem (Δ > 1 pp, IC95 > 0).

*Precisión añadida el 2026-10-01, antes de tener datos reales: la categoría "empeora" no estaba en la primera versión de esta sección.*

### Otros análisis

- **G2b y la receta** (§4).
- **F1b:**
  - todos contra todos en T-mine, el macro de T-gen, T-rand, T-gem y T-mine-50;
  - abstención en T-abst-mine (primaria) y T-gem (secundaria);
  - C1 (§5).

**Estadística:** la de siempre. Bootstrap jerárquico pareado (documentos; semillas dentro de condición), Holm, TOST ±1 pp.

## 7. Qué cuenta como fallo

- Menos de 5 generadores en el macro (§3.6).
- QC manual con > 5 % de falsos negativos en un generador: ese generador sale del macro.
- G2b sin las 2 semillas de cada condición: no hay decisión de receta y F1b no arranca.
- Un cambio de código durante F1b: el análisis lo advierte; las corridas con otro hash se rehacen.

## 8. Costo estimado [SUPUESTO; el notebook mide los tiempos reales]

**GPU:** RTX PRO 6000 Blackwell (96 GB); 18.87 unidades de cómputo por hora, según tu celda 4.17.

| Paso | Tiempo |
|---|---|
| Panel: generar con un modelo de 7–9B (2,000 ítems × 14 respuestas) | ~10–20 min |
| Panel: generar con un modelo de 3–4B | ~5–10 min |
| Panel: gpt-oss-20b | ~30–45 min |
| Panel: Qwen3-30B-A3B (1,000 ítems; MoE lento en transformers) | ~30–60 min |
| Panel: embeber y auditar todo | ~15 min |
| **Panel completo (16)** | **~4–6 h** |
| **Panel sin gated (13)** | **~3.5–5 h** |
| F0.7b, con los datos ya en RAM (F1 piloto: mediana 0.7 min por corrida) | ~1–2 h |
| F1b con la receta actual (como el piloto) | ~3–4 h |
| F1b con la receta de CLM (más muestras por corrida) | ~5–8 h |

**Memoria con la receta de CLM:** en mid, con batch 16,384 y 10 HN por fila, cada grupo de datos ocupa ~13 GB de RAM; el pico ronda ~40 GB (precarga incluida). Tu máquina tiene 190 GB.

## 9. Precisiones hechas antes de correr nada (2026-10-01)

Se agregaron al implementar y tras una revisión independiente del código. Ninguna mira resultados con datos reales.

**Regla del macro (§3.6)**

- Exige generación terminada y cobertura ≥ 80 %. Antes, una generación cortada a mitad podía entrar con los ítems de oro más corto, porque se generan en orden de largo.
- Se agregó la vía de exclusión por QC manual (parámetro `EXCLUIR_POR_QC` en 6.3, registrada en PANEL.json).
- Se agregó el respaldo `tgenall` (todos los primarios, con atajo).

**Lectura de EXP-006 (§6)**

- Se agregó la categoría "empeora".
- La fracción transferida sólo se informa si los HN suben T-gem.

**Receta de CLM (§4)**

- El tope de mid pasó de 4 a 8 épocas (una época de mid a batch 16,384 son 18 pasos).

**Parseo**

- Ya no recorta respuestas que empiezan con números ("2.5 billion", "-40 degrees").
- No descarta respuestas que empiezan con "Sure", "OK" u otras palabras de preámbulo.
- Si el oro empieza con "Answer:", los negativos conservan ese prefijo.

**Generación**

- Se detiene también en el fin de turno de cada familia, aunque su `generation_config` no lo incluya.
- Si una salida se cortó por largo, se descarta la última línea.
- gpt-oss se des-cuantiza a bf16: sus kernels MXFP4 no cubren todas las GPU.

**Integridad**

- Cada set del panel lleva una huella de su contenido. Si los negativos cambian, las auditorías y las evaluaciones viejas se rehacen y el macro viejo no se reutiliza.
- C1 avisa si falta la co-primaria T-gen.
- F3 agrega T-gen a sus corridas antes del análisis.

## 10. Ampliación del panel: 7 modelos grandes (2026-10-01, antes de generar nada)

**Decisión:** el investigador eligió agregar los 7 modelos grandes (opción recomendada) el 2026-10-01. Todavía no se había generado ningún negativo, así que esto sigue siendo diseño pre-especificado. El panel pasa de 16 a 23 generadores; la regla del macro (§3.6) no cambia.

**Por qué [INFERENCE]:**

- **Capacidad.** Los 16 originales tienen 2.6–9.4B (salvo los MoE) y son más débiles que Gemini. Un "no transfiere" podría deberse a que sus negativos son más fáciles, y no a que la ganancia sea propia de Gemini. Con modelos chicos y grandes de varias familias se pueden separar las dos cosas.
- **Control de Gemini.** Gemma 4 (2026) es más cercano en el tiempo a Gemini que Gemma 2 (2024).

### 10.1 Modelos agregados [VERIFIED en la API de HF el 2026-10-01: repo, gated, licencia, arquitectura, parámetros, plantilla de chat]

| Clave | Modelo | Familia | Rol | Acceso | Licencia | Parámetros |
|---|---|---|---|---|---|---|
| phi4_14b | microsoft/phi-4 | Microsoft | primario | abierto | mit | 14.7B |
| mistral_small24b | mistralai/Mistral-Small-24B-Instruct-2501 | Mistral AI | primario | abierto | apache-2.0 | 23.6B |
| olmo31_32b | allenai/Olmo-3.1-32B-Instruct | AI2 | primario | abierto | apache-2.0 | 32.2B |
| glm4_32b | zai-org/GLM-4-32B-0414 | Zhipu / Z.ai | primario | abierto | mit | 32.6B |
| yi15_34b | 01-ai/Yi-1.5-34B-Chat | 01.AI | primario | abierto | apache-2.0 | 34.4B |
| aya32b | CohereLabs/aya-expanse-32b | Cohere | primario | **gated** | cc-by-nc-4.0 | 32.3B |
| gemma4_31b | google/gemma-4-31B-it | Google (≈ Gemini) | control Gemini | abierto | apache-2.0 | 31.3B |

- Todos son densos y caben en la GPU (96 GB) en bf16.
- **Gemma 4:** se usa sin modo de pensamiento (`enable_thinking=False`). Con esa opción, su plantilla deja el bloque de pensamiento vacío en el prompt. Si aun así la salida trae un bloque, se guarda sólo lo que sigue a su cierre; un bloque sin cerrar se descarta.
- **Mistral:** se usa Small 3 (2501), que es sólo texto y trae el tokenizer de transformers. Small 3.2 (2506) no lo trae: sólo tiene el formato de mistral-common.

### 10.2 Considerados y descartados

| Modelo | Motivo |
|---|---|
| Llama 3.3 70B, Apertus 70B, Mistral Small 4 (119B), GLM-5.3-Flash (~321B, FP8), DeepSeek V4 | No caben en 96 GB en bf16. Comprimirlos (4–8 bits) cambia el modelo y no se puede probar aquí. |
| gpt-oss-120b | Des-cuantizado a bf16 ocupa ~235 GB; sus kernels MXFP4 no cubren todas las GPU. |
| Muse Glimmer 30B (Meta, agosto 2026) | Razona en un mensaje aparte (`to=self`) que termina en un fin de mensaje antes de la respuesta, con razonamiento "high" por defecto. Su `generation_config` se detiene en dos tokens distintos; si uno es ese fin de mensaje, la generación se cortaría antes de la respuesta. Necesita un manejo propio que no se puede probar sin el modelo. Se puede agregar después, con su propia prueba. |
| Ministral 3 14B | Pesos en FP8 y arquitectura multimodal. |
| Granite 4 H, Falcon-H1 | Híbridos con Mamba: lentos en transformers sin kernels especiales. |

### 10.3 Qué no cambia

- La regla del macro (§3.6): rol primario, generación terminada, cobertura ≥ 80 %, atajo no alto y sin exclusión por QC. Ahora hay hasta 18 primarios: 12 chicos, 6 grandes y gpt-oss-20b.
- El prompt, el muestreo, los 2,000 ítems, el filtro de falsos negativos y la auditoría.
- **El panel completo (23) se genera antes de mirar EXP-006.** No se agregan ni quitan generadores después de ver resultados.

### 10.4 Análisis secundario pre-especificado: tamaño del generador

- **Grupos:**
  - "grandes": los 6 primarios de 14–34B;
  - "chicos": los 12 primarios densos de 2.6–9.4B.
  - Los MoE (gpt-oss-20b, Qwen3-30B-A3B) no entran en ninguno.
- **Macro por tamaño:** sólo con los generadores que ya entraron al macro (§3.6), cada grupo en la intersección de sus propios ítems. Archivos `tgenbig` y `tgensmall`.
- **En EXP-006, para cada contraste:**
  - el efecto en el macro de grandes y en el de chicos, con la misma lectura de §6;
  - la **interacción**: efecto en grandes menos efecto en chicos, en los ítems comunes a los dos, con bootstrap por documento.
- **Es secundario:** ninguna regla ni decisión depende de él. Responde si la transferencia depende de la capacidad del generador. Se reporta en las dos direcciones, sin hipótesis direccional.

### 10.5 Cambios de código que acompañan la ampliación (v0.3.1)

- **transformers ≥ 5.5** (lo necesita Gemma 4). Se instala en la celda 1.2, antes de importarlo: actualizarlo a mitad de sesión mezcla versiones en memoria.
- **G1 otra vez con el entorno actual (6.2).** Antes de embeber negativos se repite la verificación de G1 con la versión instalada: respuestas y orden de los HN contra los embeddings públicos, con el mismo umbral. Si no se reproduce, no se embebe nada. Motivo: si el entorno cambiara los embeddings, los negativos del panel quedarían en otro espacio que el oro público, y eso sería un atajo.
- **Batch según la memoria libre** de la GPU y, si igual falta memoria, a la mitad. Así los modelos grandes no cortan la generación.
- Cargador de respaldo para arquitecturas registradas sólo como multimodales (se usan sólo con texto).

### 10.6 Costo adicional [SUPUESTO; la 6.1 mide los tiempos reales]

| Paso | Tiempo |
|---|---|
| Un generador de 14–24B (2,000 ítems × 14 respuestas) | ~20–60 min |
| Un generador de 31–34B | ~40–80 min |
| **Los 7 grandes** | **~5–9 h (~95–170 unidades a 18.87/h)** |
| **Panel completo (23)** | **~9–15 h (~170–285 unidades)**, en dos sesiones: `originales` y `grandes` |

Disco: cada modelo grande baja 47–69 GB, que se borran al terminar.

## 11. Panel v2: tres construcciones y un piloto con regla fija (2026-10-02)

**Cuándo se decidió:** después de auditar la v1 del panel (EXP-006a) y **antes de mirar EXP-006** o cualquier T-gen de un modelo entrenado. Nada de esta sección usa resultados de modelos entrenados.

**Quién decidió:** el investigador pidió el 2026-10-02 que "Ejecutar todo" corra el estudio completo, sin opciones. La extensión de §3.6 estaba marcada [AUTHOR DECISION NEEDED si ocurre]; queda cubierta por esa instrucción y por la regla fija del piloto (§11.3).

### 11.1 Qué falló en la v1 (EXP-006a)

- **Los 16 generadores originales quedaron con atajo alto.** El ciego propio acertó 0.77–0.91 y el ciego de Gemini 0.33–0.83 (azar 0.091). La cobertura fue de 27–44 %, porque el filtro de coseno descartó ~55 % de las respuestas. El macro quedó vacío.
- **Causas encontradas en los textos** (15 de los 16 `raw.jsonl` leídos de Drive; el de gpt-oss pesa 22 MB y el conector no lo baja):
  - 416 de los 2,000 ítems no admiten 10 respuestas incorrectas del mismo estilo: 250 de sí/no, 142 de opción múltiple y 24 binarios ("¿A o B?");
  - el oro termina en punto en el 61 % de los ítems restantes y los negativos casi nunca;
  - el filtro de coseno > 0.95 sacaba sobre todo las respuestas que imitaban el formato del oro, y dejaba pasar falsos negativos reales ("6 pages" con oro "6");
  - **el oro es más largo y más específico** que lo que el generador inventa. Pasa incluso con Qwen3-30B-A3B, el modelo que escribió el oro.
- **Sonda de texto** [proxy: clasificador lineal sobre n-gramas de caracteres y rasgos de largo y formato; validación cruzada por ítem; sólo ve las 11 respuestas]. En los Gemini de T-gem (200 ítems del QC) acierta 0.52, frente a 0.58 del ciego con embeddings en todo T-gem, así que aproxima bien al ciego real.

  | Set | Sonda de texto | "Elegir la más larga" |
  |---|---|---|
  | v1 sin el filtro de coseno (15 generadores) | 0.37–0.61 (mediana 0.52) | 0.12–0.44 |
  | Sólo reglas de texto (S, abajo) | 0.30–0.53 (mediana 0.42) | 0.10–0.48 |
  | Qwen3-30B-A3B con S (mismo modelo que el oro) | 0.47 | 0.46 |

  **Lectura [INFERENCE]:** las reglas de texto solas dejarían la mayoría de los generadores en la banda alta. El atajo que queda es sobre todo de **longitud y especificidad**, no de familia del generador.

### 11.2 Construcciones

Las mismas reglas para todos los generadores, aplicadas a lo generado (`raw.jsonl`, `raw_edit.jsonl`, `para.jsonl`).

**Reglas comunes:**

- Sólo ítems de tipo "ok": fuera los de sí/no, opción múltiple y binarios. La cobertura se calcula sobre los ítems elegibles.
- Se descartan las respuestas vacías, las preguntas, los duplicados y las que contienen entero un oro corto (≤ 4 palabras): son falsos negativos probables.
- Se guardan las 10 de largo (en caracteres) más parecido al del positivo.
- El coseno al positivo sólo se reporta; ya no filtra.

| | Positivo | Negativos | Formato |
|---|---|---|---|
| **S** | el oro original (embedding del dataset) | respuestas incorrectas libres (prompt `tgen-v1`, el de §3.3) | el del oro: puntuación final, mayúscula inicial, sin explicación final entre paréntesis |
| **P** | el oro **reescrito por el mismo generador** (prompt `tgen-para-v1`, 3 reescrituras; embebido con nuestro pipeline) | los mismos 10 de S | canónico para los 11: sin puntuación final, mayúscula inicial |
| **E** | el oro original | **ediciones** del oro: cambian sólo el dato clave (prompt `tgen-edit-v1`); se exige que saquen y pongan al menos una palabra de contenido | el del oro |

**Largo asignado en P.** Para cada generador e ítem se sortea t ~ Uniforme{0..10}, con semilla fija: el número de negativos que quedarán **más largos** que el positivo. El largo pedido cae entre los negativos correspondientes, pasado a palabras con los caracteres por palabra de esos negativos. De las 3 reescrituras se usa la que deja exactamente t negativos más largos, o la más cercana.

Así, la longitud no dice cuál es el correcto. Con los datos de la v1, los largos pedidos quedan en 0.5–1.9 veces el del oro (mediana 0.9–1.0).

Nunca se usa una copia del oro. Excepción: si el oro tiene ≤ 3 palabras ("Nature.", "2813") y el generador sólo lo copió, el positivo es el texto del oro en formato canónico, embebido como los negativos.

**Prompt de edición (E):**

```
You are helping build a benchmark of hard incorrect answers for question answering.

Question: {question}
Correct answer: {answer}

Write {k} different INCORRECT answers by editing the correct answer: replace its key fact or facts (a name, number, term, place, date, method, cause, or relation) with different, plausible, but wrong ones, and keep the rest of the wording the same.
Rules:
- Each answer must be clearly wrong for this question, not a synonym or a rewording of the correct answer.
- Keep about the same number of words as the correct answer ({n_words}) and the same format, punctuation, and capitalization.
- Make each replacement as specific as the original (for example, the same number of digits and the same units).
- Do not make an answer wrong just by adding a negation such as "not" or "no".
- Do not say that the answers are incorrect and do not add explanations.
Return exactly {k} lines, one answer per line, with no numbering, bullets, or extra text.
```

**Prompt de reescritura (P):**

```
Rewrite the correct answer to this question in your own words.

Question: {question}
Correct answer: {answer}

Write {k} different rewrites. Each rewrite must:
- keep the same meaning and stay correct, with the same key facts and no new facts;
- be about {n_words} words long (shorten or expand the wording as needed);
- not copy the correct answer word for word.
Do not add explanations. Return exactly {k} lines, one rewrite per line, with no numbering, bullets, or extra text.
```

El muestreo es el de §3.3 (temperature 0.8, top-p 0.95, bf16, hasta 2 intentos). En E se piden k = 14; en P, k = 3.

### 11.3 Piloto y regla de elección

1. **Ítems y generadores.** Los primeros 500 ítems del orden del panel y los primeros 3 primarios con acceso de esta lista: mistral7b, granite33_8b, llama31_8b, olmo2_7b, falcon3_7b, yi15_9b.
2. **Armado y auditoría.** Se arman S, P y E con esos mismos ítems y generadores, y cada set se audita con las sondas de §3.5. El atajo ciego es el máximo de las sondas.
3. **Regla.** Gana la construcción con la **menor mediana del atajo ciego** entre los generadores. Un empate (< 1 pp) lo gana la más barata: S, luego P, luego E. Una construcción con menos de 2 generadores auditados no compite.
4. **Registro.** La elección queda en `data/eval/tgen/BUILD.json`, con la tabla del piloto. Vale para test y confirm, y el PROTOCOL FREEZE la congela.

Después, el panel completo (23 generadores) se arma con la construcción elegida. La regla del macro (§3.6) y las bandas no cambian.

### 11.4 Otros cambios respecto de lo pre-registrado

- **Filtro de coseno (§3.4):** reemplazado por las reglas de texto (§11.2).
- **Cobertura:** sobre los ítems elegibles, no sobre los pedidos.
- **QC manual:** la 6.3 aplica sola el criterio de DISENO_01. Hace falta que un generador tenga ≥ 50 filas marcadas, y queda fuera del macro si > 5 % de sus negativos son correctos o > 5 % de sus positivos reescritos no lo son. Sin marcas no excluye a nadie.
- **Respaldos de §3.6:** se calculan aunque el macro quede vacío:
  - `tgenall`: todos los primarios, "con atajo";
  - `tgenfail` (nuevo): los pares (generador primario, ítem) donde el ciego propio de ese generador falla, con bootstrap por documento.

  EXP-006 los reporta con la misma lectura que el macro, marcados como respaldo.

### 11.5 Riesgos

- **Tamaño del piloto.** Con 500 ítems, el ciego propio se entrena con menos datos y subestima el atajo. La comparación entre construcciones es justa (mismos ítems), pero las bandas finales salen de la auditoría del panel completo.
- **P:** una reescritura puede cambiar el sentido y quedar incorrecta. Eso agrega ruido, no sesgo; lo cubre el QC (filas "positivo").
- **E:** una edición puede ser casi un sinónimo (falso negativo); lo cubre el QC. Es una construcción nueva, no pre-registrada; por eso compite en el piloto con la regla fija.
- **La sonda de texto de §11.1 es un proxy.** Lo que decide es la auditoría con embeddings.

### 11.6 Costo [SUPUESTO; el notebook mide]

| Paso | Tiempo |
|---|---|
| Piloto (500 ítems × 3 generadores × 3 construcciones, embebido y auditoría) | ~20–40 min |
| Panel completo con **S** (sólo faltan los negativos de los 7 grandes) | ~5–9 h |
| Panel completo con **P** (S + 3 reescrituras por ítem en los 23) | ~6–11 h |
| Panel completo con **E** (ediciones en los 23; gpt-oss sola ~4 h) | ~12–16 h |

## 12. Después de la corrida del 2026-10-02/03: correcciones y dos análisis pre-especificados (2026-10-05)

**Cuándo:** después de ver F1b, el panel E, EXP-006 y F2b (notebook rev. 4.2), y **antes del PROTOCOL FREEZE**. EXP-006c (§12.3) todavía no se corrió con datos reales: su regla de lectura se fija aquí, antes de mirar. C2 en F3 (§12.4) se fija antes de abrir el test confirmatorio.

### 12.1 Qué se encontró al revisar la corrida

- **Gemma 4 31B no generó.** Fue un bug de nuestro código, no del modelo. El cálculo del batch leía el número global de cabezas K/V. En Gemma 4 ese número varía por capa, y transformers lanza un error que no es `AttributeError`. Se corrigió en v0.3.3: suma capa por capa y, si no puede leerla, usa el batch pedido y lo baja a la mitad si falta memoria. Gemma 4 es un control de Gemini, no un primario, así que el macro no cambia. La 6.1 lo completa en la próxima corrida.
- **La matriz de transferencia entre ciegos (§3.5) tenía fuga por ítem.**
  - **Qué hacía:** cada ciego (fila) se entrenaba con **todos** los ítems de su set y se evaluaba en **todos** los ítems de los otros sets (columnas). En E (y en S) el oro es el mismo texto en todos los sets, así que el ciego podía reconocer oros que vio al entrenar.
  - **Señal en los datos reales:** en 397 de 462 celdas, el ciego de otro generador acierta más que el ciego propio de la columna. En los 5 generadores del macro, el ciego propio da 0.25–0.30 y el máximo fuera de la diagonal 0.37–0.47.
  - **Prueba sintética:** con el mismo oro en dos sets y negativos de la misma distribución que el oro (sin ningún atajo posible), la versión vieja da 0.17 (azar 0.09) y la corregida 0.09.
  - **Corrección:** pliegues por ítem comunes a todos los generadores; nunca se evalúa en un ítem cuyo oro se vio al entrenar.
  - **Alcance:** la matriz sólo se mostraba. Las bandas y el macro usan el ciego propio (pliegues por ítem dentro de cada set), el ciego de Gemini (entrenado en otro split) y las heurísticas de longitud; nada de eso tiene esta fuga. La pregunta de §3.5 (¿el atajo es el estilo del oro o del generador?) se responde con la matriz corregida.
- **EXP-006 mezcla generador y construcción.** T-gem tiene negativos **libres** de Gemini; el macro de T-gen, negativos **editados** de otros generadores. La "fracción transferida" (0.16–0.24 en los contrastes con HN + in-batch) no separa si la ganancia por HN es propia de Gemini o del tipo de negativo.

### 12.2 Qué no cambia

- **Panel:** la construcción E, la regla del macro, las bandas y los 5 generadores del macro (aya32b, glm4_32b, gptoss20b, mistral_small24b, phi4_14b; 1,125 ítems).
- **Métricas:** la primaria T-mine, la co-primaria T-gen y C1.
- **Análisis:** la estadística y el mecanismo del freeze.

### 12.3 EXP-006c — ¿generador o construcción? (exploratorio; no genera ni entrena nada)

**Datos.** Los sets del piloto (6.0): mistral7b, granite33_8b y llama31_8b, cada uno con S, P y E, en los primeros 500 ítems del panel. Ya están embebidos y auditados.

- Ítems: la intersección de los 9 sets.
- Por ítem, el acierto se promedia sobre los 3 generadores, igual que el macro.
- T-gem se restringe a esos mismos ítems: también son del split de test.

**Corridas.** Las que ya existen: F1b (infonce, infonce_hn, siglip, siglip_hn), F0.7 y F0.7b (receta actual). Sólo se evalúan en esos sets.

**Cantidades,** para cada contraste A − B (Δ = ganancia por HN):

| Cantidad | Fórmula |
|---|---|
| Efecto del generador | Δ(T-gem) − Δ(S): libres de Gemini frente a libres de otros generadores |
| Efecto de la construcción | Δ(S) − Δ(E): libres frente a editados, con los mismos generadores |
| Efecto del oro reescrito | Δ(P) − Δ(S): sólo cambia el positivo |
| Brecha total | Δ(T-gem) − Δ(E) |

Cada Δ se reporta además con estados barajados. El bootstrap es pareado por documento, con semillas dentro de condición.

**Contrastes.** El primario es F1b infonce_hn − infonce. Los demás: F1b siglip_hn − siglip; F0.7b mid_inbatch − pre; F0.7 HN + in-batch − pre; y los de "HN solos" (mecanismo).

**Lectura, fijada ahora.** Para cada componente:

- **explica parte de la brecha:** > +1 pp con IC95 > 0;
- **en sentido contrario:** < −1 pp con IC95 < 0;
- **no la explica:** equivalente por TOST ±1 pp;
- **no concluyente:** cualquier otro caso.

Las proporciones de generador y construcción (sobre la brecha total) son descriptivas. Sólo se informan si la brecha total es > +1 pp con IC95 > 0.

**Límites, escritos antes de mirar:**

- Son 3 generadores chicos (7–8B).
- S y P tienen atajo alto en el piloto (medianas 0.46 y 0.63). Una ganancia en S puede venir en parte de ese atajo; los estados barajados lo dicen.
- Son ~300 ítems comunes, así que los IC serán anchos.
- Los HN de T-gem salieron del pipeline de CLM (prompt propio), así que el "efecto del generador" incluye ese pipeline.

**Ninguna decisión depende de EXP-006c.** Informa la interpretación de H1b y qué se puede afirmar en el paper.

### 12.4 C2 en F3 (secundario pre-especificado; se registra en el freeze)

| | |
|---|---|
| **Pares** | infonce_hn − infonce y siglip_hn − siglip (los cuatro brazos están en F3) |
| **C2** | Δ(T-gem) − Δ(macro de T-gen), en los ítems del macro del test confirmatorio. Bootstrap pareado por documento (semillas dentro de brazo), con Holm entre los dos pares. |
| **Lectura** | Sólo aplica si los HN suben T-gem en ese par (Δ > +1 pp con IC95 > 0). **H1b se sostiene** en un par si C2 > +1 pp con p_Holm < 0.05. **Equivalente** si TOST ±1 pp. **Mayor con otros generadores** si C2 < −1 pp con IC95 < 0. Si no, no concluyente. |
| **Dónde queda** | `PROTOCOL_FREEZE.json` (`secondary_confirmatory.C2`); lo calcula la celda 8.18b. |

- **C1 no cambia:** sigue siendo el contraste confirmatorio primario.
- **Límite:** C2 mezcla generador y construcción (§12.1); EXP-006c es su descomposición exploratoria.

### 12.5 Código v0.3.3 (hash `a31c20d9d8688ea6`)

- `genworker.kv_bytes_per_token`: el cálculo del batch funciona con configs heterogéneas por capa (Gemma 4).
- `generators.transfer_rows` / `blind_transfer_matrix`: pliegues por ítem comunes. La versión anterior se mueve a `history/blind_transfer_v032_fuga_por_item.csv`.
- `pipeline.eval_extra`: evalúa corridas terminadas en sets extra, sin tocar `DONE.json`.
- `analysis.construction_report` (EXP-006c) y `analysis.c2_report` (C2).
- `protocol_freeze` registra C2.
- Notebook rev. 4.3: celdas 6.6b y 8.18b.

## 13. C2b y firma de las generaciones por contenido (2026-10-05, antes del freeze)

**Cuándo:** después de ver EXP-006c y EXP-006d (rev. 4.3), **antes del PROTOCOL FREEZE** y sin haber abierto el test confirmatorio.

### 13.1 Por qué

- EXP-006c (exploratorio) separó la brecha de C2.
  - **Generador:** +15.3 pp [9.4, 21.4] en el contraste primario, ~72 % de la brecha.
  - **Tipo de negativo:** +6.0 pp [2.4, 9.4], ~28 %.
- Cambia cómo se interpreta H1b: lo que no se transfiere es sobre todo propio de los negativos de Gemini y está condicionado al estado.
- Es el hallazgo más informativo del estudio. Confirmarlo con datos independientes cuesta ~1 h de generación.

### 13.2 C2b (secundario pre-especificado; se registra en el freeze)

| | |
|---|---|
| **Datos** | Sets S, P y E de los mismos 3 generadores del piloto (los de `BUILD.json`), generados después del freeze para los primeros 500 ítems del orden del panel del test confirmatorio. Mismos prompts, muestreo, reglas y embebido que en el exploratorio (celda 8.2c). |
| **Corridas** | Los brazos de F3: infonce, infonce_hn, siglip y siglip_hn, con 5 semillas y HP congelados |
| **Cantidades** | Las de §12.3 en los ítems comunes a los 9 sets: generador = Δ(T-gem) − Δ(S); construcción = Δ(S) − Δ(E); oro reescrito = Δ(P) − Δ(S); brecha total = Δ(T-gem) − Δ(E). Bootstrap pareado por documento (semillas dentro de brazo) |
| **Pruebas** | Generador y construcción en los 2 pares de C2 (4 contrastes), con Holm entre los 4 |
| **Lectura** | Sólo aplica si la brecha total es > +1 pp con IC95 > 0. Un componente **se confirma** si es > +1 pp con p_Holm < 0.05; **en sentido contrario** si es < −1 pp con p_Holm < 0.05; **no la explica** si es equivalente por TOST ±1 pp; si no, no concluyente. El oro reescrito y las proporciones son **descriptivos** |
| **Dónde queda** | `PROTOCOL_FREEZE.json` (`secondary_confirmatory.C2b`); lo calcula la celda 8.18c (`results/F3/C2b/`) |

**Límites, fijados antes de mirar:**

- 3 generadores chicos.
- ~280 ítems comunes.
- S y P con atajo ciego alto.
- "Generador" incluye el pipeline de CLM para sus HN.

C1 (primario) y C2 no cambian.

### 13.3 Firma de las generaciones

- **Antes:** `needs_embedding` comparaba la firma de los `raw*.jsonl` como [tamaño, mtime].
- **Problema:** en Drive el mtime cambia entre sesiones, y en la rev. 4.3 eso hizo re-embeber 16 sets idénticos. La huella de su contenido no cambió, así que las evaluaciones y el macro quedaron intactos.
- **Ahora:** la firma es [tamaño, sha1 del contenido]. Con una firma vieja sólo se compara el tamaño: las generaciones sólo se agregan al final del archivo.

### 13.4 Código v0.3.4 (hash `94211b491b3f5d44`)

- `generators.raw_sigs_match`, `build_pilot_sets` y `synthetic_pilot_sets`.
- `analysis.c2b_report` y `c2b_reading`.
- `protocol_freeze` registra C2b.
- Notebook rev. 4.4: celdas 8.2c y 8.18c.

## 14. C2c y freeze confirmado (2026-10-05, antes del freeze)

**Cuándo:** después de la corrida de la rev. 4.4, **antes del PROTOCOL FREEZE** y sin haber abierto el test confirmatorio.

- Los resultados de la rev. 4.4 fueron idénticos a los de la rev. 4.3.
- F3 no corrió porque la casilla de la 8.1 no estaba marcada.
- El investigador confirmó el freeze en la conversación del proyecto.

### 14.1 Por qué

- **El mecanismo de H1b es la parte más mecanística del estudio:** comparar el oro sólo contra sus HN aprende una preferencia que no depende del estado y cuesta discriminación real.
- **Pero toda su evidencia es exploratoria y viene del mismo test:**

  | Fuente | T-gem con estados barajados | T-mine |
  |---|---|---|
  | F0.7, HP por defecto, 2 semillas (EXP-005) | +13.1 pp [12.3, 13.8] | −8.7 [−9.5, −7.9] |
  | F0.7b, receta actual (EXP-006) | +14.1 | −8.4 |
  | F2b, HP de F1b, 3 semillas (EXP-009) | +20.1 | −6.3 |

- **El control apunta en la misma dirección:** con in-batch, los HN no crean esa preferencia. infonce_hn − infonce en T-gem barajado da +0.2 en F1b; HN + in-batch − pre da +0.3 [−0.2, 0.8] en F0.7.
- **Costo de confirmarlo en el test sellado:** ~10 min (5 corridas más).

### 14.2 C2c (secundario pre-especificado; se registra en el freeze)

| | |
|---|---|
| **Corridas** | Variante `infonce_hn_hnonly`: HP congelados de infonce_hn y sus 5 semillas. Sólo cambia el denominador, que pasa a "sólo HN" (`loss_params.hn_denominator = hn_only`). Queda registrada en `PROTOCOL_FREEZE.json` (`F3_variants`) y la entrena la celda 8.17b. La referencia es infonce_hn de F3 ("HN + in-batch") |
| **Pruebas** | (1) Top-1 en T-gem con estados barajados: se predice Δ > +1 pp. (2) Top-1 en T-mine: se predice Δ < −1 pp. Bootstrap pareado por documento (semillas dentro de brazo), con Holm entre las 2 |
| **Lectura** | Una prueba **se confirma** si \|Δ\| > 1 pp en la dirección predicha con p_Holm < 0.05. Va **en sentido contrario** si cumple lo mismo en la otra dirección. **No se confirma (equivalente)** si es equivalente por TOST ±1 pp; en otro caso es no concluyente. El mecanismo se confirma si se confirman las 2 pruebas |
| **Control (fuera de Holm)** | infonce_hn − infonce en T-gem con estados barajados. Se confirma si es equivalente por TOST ±1 pp: con in-batch, los HN no crean la preferencia |
| **Descriptivos** | T-gem, T-rand, T-mine-50 y el macro de T-gen |
| **Dónde queda** | En el freeze, `secondary_confirmatory.C2c` y `F3_variants`. Lo calcula la celda 8.18d (`results/F3/C2c/`) |

**Límites, fijados antes de mirar:**

- La variante usa los HP elegidos para "HN + in-batch", sin tuning propio (igual que en F2b).
- Hay un solo brazo (InfoNCE) y un solo generador de HN de entrenamiento (Gemini).
- La preferencia sin estado se mide con estados barajados, así que no detecta un atajo que combine estado y estilo.

C1, C2 y C2b no cambian. La variante no entra al todos contra todos, a C1, a C2/C2b ni a T-Rex.

### 14.3 Freeze confirmado

- El investigador confirmó el PROTOCOL FREEZE el 2026-10-05, en la conversación del proyecto ("Sí, congela").
- La rev. 4.5 trae marcada la casilla `CONFIRMO_FREEZE` de la 8.1, y la nota del freeze registra la decisión.
- Con "Ejecutar todo", la 8.1 congela y la Sección 8 corre F3 completo.

### 14.4 Código v0.3.5 (hash `3d563f764976e02b`)

- `pipeline.F3_VARIANTS`, `confirm_variant_cfg` y `run_confirm_variant`.
- `analysis.c2c_report`, `c2c_reading` y `c2c_control_reading`.
- `protocol_freeze` registra C2c y `F3_variants`.
- Notebook rev. 4.5: celdas 8.17b y 8.18d, y la 8.1 con la casilla marcada.

## 15. Adenda del freeze y cómo se completa F3 (2026-10-06, después del freeze)

**Cuándo:** después del freeze y de la primera parte de F3 (EXP-010). Esta sección no cambia ninguna regla de análisis. Registra lo que pasó y cómo se completa lo que falta.

### 15.1 Qué pasó

- **El freeze se hizo el 2026-10-06 a las 02:33 UTC desde la rev. 4.4**, con el código v0.3.4 (hash `94211b491b3f5d44`), y no con la rev. 4.5. Esa versión no registraba C2c: el freeze trae C1, C2 y C2b.
- **Los 15 brazos × 5 semillas de F3 se entrenaron con el código congelado**, el 2026-10-06 entre las 10:36 y las 11:43 UTC. En sus `DONE.json`, `code_hash_start` y `code_hash_end` valen `94211b491b3f5d44` [VERIFIED en Drive].
- **El disco local de Colab se llenó** durante el panel del confirmatorio (8.2b) en las dos sesiones posteriores al freeze.
  - Quedaron generados aya32b, glm4_32b, gpt-oss-20b y mistral_small24b.
  - Phi-4 no se pudo bajar ("No space left on device"), y ningún set se embebió.
  - Tampoco se pudieron generar los sets del piloto (8.2c).
  - Por eso faltan la co-primaria de C1 (macro de T-gen), C2 y C2b.

### 15.2 C2c: adenda del freeze

- **Qué es:** `config/PROTOCOL_FREEZE_ADDENDUM.json`. Registra C2c y la variante "HN solos" con la especificación exacta de §14 (la misma función genera el texto del freeze y el de la adenda).
- **Cómo se escribe:** se crea una sola vez, antes de entrenar la variante, y no modifica el freeze.
- **Por qué es legítima:**
  - §14 se escribió antes del freeze y antes de abrir el test confirmatorio.
  - El investigador confirmó el freeze sabiendo que incluía C2c.
  - Ninguna elección de C2c queda abierta: variante, pruebas, umbrales y Holm ya están fijados.
- **Desviación que hay que declarar al reportar C2c:**
  - se registró fuera del JSON del freeze;
  - se ejecuta después de ver los resultados de F3 de los 15 brazos, incluido infonce_hn, que es su referencia.

  Qué etiqueta usar en el manuscrito, "confirmatorio con desviación" o "pre-especificado, ejecutado después", queda para la redacción [AUTHOR DECISION NEEDED].

### 15.3 Cómo se completa F3 (sin cambiar ninguna regla)

- **Panel del confirmatorio:** Phi-4, embebido y auditoría de los 5 generadores congelados. El macro es el del freeze; no se vuelve a elegir.
- **T-gen en las corridas de F3:** se agrega sin re-entrenar (`add_missing_evals`).
- **Sets del piloto en confirm** (para C2b).
- **La variante "HN solos"** (para C2c).
- **C1 se recalcula** con sus dos co-primarias y Holm entre ambas. La parte de T-mine ya no puede cambiar de veredicto: con p = 2.2e-61, Holm entre dos pruebas da a lo sumo 4.5e-61.
- **Luego** C2, C2b y C2c.

### 15.4 Integridad del código después del freeze

- **Qué cambia en v0.3.6:** sólo la gestión del disco en la generación, la adenda y que T-Rex no repita brazos ya hechos.
- **Qué no cambia respecto del código congelado:** el entrenamiento y la evaluación son idénticos byte a byte (`train`, `evaluate`, `losses`, `data`, `heads`, `hpo`, `stats`, `config`, `utils`, `encoder`, `texts` y `genworker`), y también `run_confirm_arm`, `add_missing_evals` y `base_cfg`.
- **La generación** usa los mismos prompts, muestreo, semillas y modelos.
- **La advertencia de hash de las celdas de F3** es esperable y queda explicada aquí.
- **Código:** v0.3.6, hash `e160594adcd46368`; notebook rev. 4.6.

### 15.5 Cómo se completó (2026-10-07)

- **La adenda se creó el 2026-10-07 a las 01:52:08 UTC** (celda 8.1 de la rev. 4.6, código `e160594adcd46368`), antes de entrenar la variante. Su texto es el de §14, sin cambios, y el freeze quedó igual [VERIFIED en Drive: `config/PROTOCOL_FREEZE_ADDENDUM.json`].
- **C2c corrió con `registro: adenda`** y guardó el texto de la desviación en `results/F3/C2c/C2c.json`.
- **Resultados de F3 completos:** EXPERIMENT_LOG, EXP-011.
- **Después del cierre** sólo se agrega un análisis de sensibilidad del panel, post hoc y descriptivo (DP-08). No cambia ninguna regla ni ningún veredicto de C1, C2, C2b o C2c.
- **v0.3.7** cambia sólo la limpieza del disco al generar (DP-08). El entrenamiento y la evaluación siguen idénticos al código congelado.
