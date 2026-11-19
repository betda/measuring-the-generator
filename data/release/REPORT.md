# Export report

- Root: `/content/drive/MyDrive/CLM_losses/colab`
- Exported: 26606 files (239.4 MB before compression)
- Skipped: 5508 files
- Study texts used for the overlap check: 545300 strings from 50 files (results/PRE/textos and the T-gen panel)

## Skipped, by reason

- **scores left out (--no-scores)**: 4781 files, e.g. `runs/F0.7/infonce_hn_hnonly/seed_0/eval/scores_pilotE_granite33_8b.npy`, `runs/F0.7/infonce_hn_hnonly/seed_0/eval/scores_pilotE_llama31_8b.npy`, `runs/F0.7/infonce_hn_hnonly/seed_0/eval/scores_pilotE_mistral7b.npy`
- **model checkpoint**: 248 files, e.g. `runs/F0.7/infonce_hn_hnonly/seed_0/best_mid.pt`, `runs/F0.7/infonce_hn_hnonly/seed_1/best_mid.pt`, `runs/F0.7/infonce_hn_inbatch/seed_0/best_mid.pt`
- **embeddings or token IDs**: 140 files, e.g. `data/eval/confirm/c0000_a.npy`, `data/eval/confirm/c0000_n.npy`, `data/eval/confirm/c0000_q.npy`
- **file type not on the allow list**: 114 files, e.g. `results/F1/all_vs_all/cd_diagram.png`, `results/F1/all_vs_all/heatmap_abst_auroc.png`, `results/F1/all_vs_all/heatmap_tgem_top1.png`
- **JSON Lines outside runs/ and config/ (prompts, generations or texts)**: 104 files, e.g. `data/eval/tgen/confirm/items_text.jsonl`, `data/eval/tgen/confirm/aya32b/kept.jsonl`, `data/eval/tgen/confirm/aya32b/raw_edit.jsonl`
- **text field 'prompt'**: 58 files, e.g. `data/eval/tgen/confirm/aya32b/gen_info_edit.json`, `data/eval/tgen/confirm/glm4_32b/gen_info_edit.json`, `data/eval/tgen/confirm/gptoss20b/gen_info_edit.json`
- **text column 'pregunta'**: 24 files, e.g. `data/eval/tgen/test/QC_falsos_negativos_panel.csv`, `data/eval/tgen/test/apertus8b/QC_falsos_negativos.csv`, `data/eval/tgen/test/aya32b/QC_falsos_negativos.csv`
- **marker file**: 18 files, e.g. `data/eval/confirm/c0000.done`, `data/eval/confirm/c0001.done`, `data/eval/confirm/c0002.done`
- **report in Markdown (its numbers are in the JSON and CSV files)**: 10 files, e.g. `results/F1/RESUMEN_F1.md`, `results/F1b/RESUMEN_F1b.md`, `results/F3/RESUMEN_F3.md`
- **labeling sheet with texts (the labels are released in data/labels of the repository)**: 3 files, e.g. `results/PRE/b1_calidad/etiquetas/ETIQUETADAS.xlsx`, `results/PRE/b1_calidad/etiquetas/ETIQUETADAS_amplia.xlsx`, `results/PRE/b1_calidad/etiquetas/ETIQUETAR_*.csv`
- **binary data container**: 2 files, e.g. `data/eval/test/lengths.npz`, `data/eval/val/lengths.npz`
- **text or cache folder**: 2 files, e.g. `results/PRE/textos/confirm.jsonl`, `results/PRE/textos/test.jsonl`
- **names the labeling sheets (they identify the annotators)**: 1 files, e.g. `results/PRE/b1_calidad/etiquetas/acuerdo.json`
- **text field 'pregunta'**: 1 files, e.g. `results/PRE/b1_calidad/fuga/fuga.json`
- **judge's raw output (exported as labels/judge_labels.csv)**: 1 files, e.g. `results/PRE/b1_calidad/juez/juez_raw.jsonl`
- **BM25 vocabulary and document frequencies of the training answers**: 1 files, e.g. `results/PRE/b3_referencias/bm25_corpus.json`

The full list is in `EXCLUDED.csv`. Review the skipped JSON and CSV files: a result file that should be public may have been skipped only because it holds a long note.
