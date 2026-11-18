# EXP-006a — análisis de los textos del panel T-gen v1

Scripts del diagnóstico del 2026-10-02 (EXPERIMENT_LOG, EXP-006a; ENMIENDA_03 §11.1). No forman parte del código del
estudio (no van embebidos en el notebook) y no usan ningún modelo entrenado.

| Archivo | Qué es |
|---|---|
| `sonda_texto.py` | Sonda ciega de texto por generador: construcción v1 (sin filtro de coseno) frente a la S de v0.3.2 |
| `sonda_texto_gemini_200.py` | Calibración con 200 ítems de T-gem (HN de Gemini) y comparación a igual tamaño |
| `resultados_sonda_texto.json` | Resultados de `sonda_texto.py` para 15 generadores (`texto` = acierto de la sonda; `mas_largo` = acierto de elegir la respuesta más larga; `punto` = de elegir la que termina en punto) |

Datos de entrada: `data/eval/tgen/test/items_text.jsonl`, `<generador>/raw.jsonl` y `QC_negativos_gemini.csv`, en Drive.
