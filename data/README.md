# Data

License: CC BY 4.0 (see `LICENSE-DATA.md`). No text of Nemotron-CC v2.1, of CLM's negatives or of the generated negatives is included.

## False-negative labels (`labels/`)

These are the labels of Section 3.9 and Appendix H of the article. They cover the sealed split.

| File | Columns | Content |
|---|---|---|
| `human_labels.csv` | `item`, `blind_id`, `candidate`, `annotator`, `label` | 3,960 labels of 2,000 candidates (200 questions × 10 negatives). Six annotators, anonymized as A1–A6, labeled them in a ring design. 1,960 candidates have two labels and 40 have one. `label` = 1 if the negative also answers the question correctly |
| `blind_key.json` | `blind_id_to_item` | The blind identifier each annotator saw (e001–e200) and the item it stands for |
| `judge_labels.csv` | `item`, `candidate`, `label`, `attempt` | Labels of OLMo 3.1 32B Instruct (prompt `fn-judge-v1`, Appendix H). Each item has the first valid answer, with its attempt number (0 or 1). 24,727 items: T-gem 9,515, T-mine 9,516, and 1,139–1,140 for each macro generator of T-gen |

`item` is `<set>:<index>`: the evaluation set (`tgem`, `tmine` or `tgen_<generator>`) and the position of the item in that set on the sealed split. `candidate` is 1–10, the position of the negative among the item's candidates.

The strict reference counts a negative as correct if both annotators marked it; the lenient reference, if at least one did. A1–A6 are the codes `alumno1`–`alumno6` of `results/PRE/b1_calidad/etiquetas/etiquetas_individuales.csv` in the data release.

## Data release (`release/`)

`release/` holds the rest of the study's data, without any text:

| File | Content |
|---|---|
| `release_data.tar.xz.001`, `.002`, ... | The archive `release_data.tar.xz`, cut into parts of 9.5 MB to fit GitHub's limits |
| `release_data.SHA256SUMS` | SHA-256 of the archive and of each part |
| `REPORT.md` | Export report: what was exported and what was skipped, and why |

To join the parts, check them and extract the archive (Python standard library only):

```bash
python release/join_release_data.py        # extracts ./release_data/
```

By hand: `cat release_data.tar.xz.0* > release_data.tar.xz && tar -xJf release_data.tar.xz` (Linux, macOS), or open `release_data.tar.xz.001` with 7-Zip (Windows). Inside, `MANIFEST.json` gives the size and SHA-256 of every file and `EXCLUDED.csv` lists every file left out, with the reason.

The archive mirrors the study folder in Google Drive:

| Folder | Content |
|---|---|
| `config/` | Study configuration, protocol freeze and addendum (JSON and Markdown), as the notebook generated them, and the log of the F0 gate decisions |
| `data/` | Document identifiers of each held-out split (`split/*_docs.txt`). For each evaluation set: item metadata (`c*_meta.json`: `doc_id`, `pair_idx`, `row`), subsample indices, and candidate indices and cosines of the random and mined sets (`rand_idx.npy`, `mine_idx.npy`, `mine_sims.npy`). For the T-gen sets, the item indices, the cosines of the negatives, and the description of the panel and of each build |
| `runs/` | Per run: `config.json` (configuration, environment and code hash), `DONE.json`, the training log `log.jsonl`, `eval/metrics.json` and the per-item arrays `correct_*.npy`, `argmax_*.npy` and `items_*.npy`; the logs of the hyperparameter search (`hpo/`) |
| `results/` | JSON and CSV result files of every phase (all versus all and pairwise tables, contrasts C1–C2c, tunability, T-Rex, panel sensitivity, transfer) and of the post hoc analyses (`PRE/`) |
| `logs/` | `pip freeze` of the sessions |
| `labels/` | The judge's labels, as above |

The study's notes and decision rules, which these files carry, are in Spanish.

The archive leaves out, besides all texts, embeddings, token IDs and checkpoints:

- the score arrays (`scores_*.npy`), to keep the release small;
- the summaries of the false negatives and of the leakage across splits, and the judge's settings (`calidad.json`, `fuga.json`, `juez_info.json`), because they quote questions or the judge's prompt. Their numbers are in the article; the labels they come from are in `labels/` and in `data/labels/`, and the prompt is in Appendix H;
- the BM25 vocabulary of the training answers (`bm25_corpus.json`) and the file that names the annotators' labeling sheets;
- figures, server logs of T-Rex and summaries in Markdown, whose numbers are in the JSON and CSV files.

### How it was built

`release/export_release_data.py` builds the release from the study folder in Google Drive, and `release/Exportar_datos.ipynb` runs it in Colab: open it with the study's Google Drive and run all cells. The notebook writes a copy of the script (`release/build_export_notebook.py` regenerates it, and a test checks that the copy matches) and runs:

```bash
python release/export_release_data.py --root /content/drive/MyDrive/CLM_losses/colab --out /content/release_data --no-scores
```

It leaves the parts of the archive, `release_data.SHA256SUMS` and the report in the `CLM_losses` folder of the Drive. Without `--no-scores` the script also exports the score arrays.

Every file is checked before it is copied:

- arrays must be on an allow list and numeric, and must not have the shape of an embedding;
- JSON and CSV files, and the JSON Lines logs of `runs/` and `config/`, must not have fields or columns that hold data text (also under longer names, such as `pregunta_entrenamiento`), sentences that are not Spanish prose, or English text: the data texts are in English and the study's notes are in Spanish;
- every exported file is compared with the study's texts (the questions, answers and candidates of the evaluation sets and the negatives of the T-gen panel), and a file that shares six consecutive words with them is skipped;
- files that name the annotators' labeling sheets are skipped.
