# Measuring the Generator, Not the State: code, registration and data

Code and materials of the article *Measuring the Generator, Not the State: The Validity of LLM-Written Negatives for State–Action Scorers* (Davis Ortiz, Maldonado Robles, Mendoza Zubia, Olvera Gonzalez, Puga Mota, Romero López and Zamora Martínez; Universidad Autónoma de Ciudad Juárez; submitted to the *Journal of Artificial Intelligence Research*).

The study trains the two heads of a state–action scorer on a frozen Qwen3-8B, as in CLM, with 15 training objectives under one protocol. It audits the evaluation sets with probes that do not see the state and with shuffled states, and measures how the gain from hard negatives written by Gemini transfers to negatives written by a panel of 23 generators. The decision rules were fixed before a sealed split was opened.

Code comments, notebooks and internal documents are in Spanish; the article is in English.

## Contents

| Path | What it holds |
|---|---|
| `clm_losses/` | The study package, v0.3.8 (code hash `a466295d82df1b00`): embedding stores and splits, the 15 objectives, training, evaluation, statistics, the generator panel and the post hoc analyses |
| `code_versions/` | Every earlier version of the package used in the study, each with its code hash (see below) |
| `notebooks/` | The Colab notebooks: main study (rev. 4.7, phases F0–F3), sensitivity of the T-gen panel in F3 (rev. A1) and post hoc analyses (rev. P2) |
| `analisis/` | Scripts for the figures and tables (`figuras/figuras_tablas.py`), the human labels (`etiquetas/`), the F3 panel sensitivity and the text probe of EXP-006a |
| `tests/` | Tests on synthetic data |
| `registration/` | The dated design, amendment and freeze documents, in Spanish, as they were written |
| `release/` | `export_release_data.py`, which builds the text-free data release from the study folder, the Colab notebook that runs it, and `join_release_data.py`, which joins and extracts the release |
| `data/` | The false-negative labels (`data/labels/`) and the data release (`data/release/`), described in `data/README.md` |
| `docs/CHANGELOG_es.md` | Development notes of the package, version by version (Spanish) |

## Code versions

Each run stores in its `config.json` the hash of the code it ran with: the first 16 hexadecimal digits of a SHA-256 over the package's `.py` files (`clm_losses.utils.code_hash`). The table maps each version kept in `code_versions/` to that hash.

| Version | Code hash | Used for |
|---|---|---|
| 0.2.0 | `ecaaaf9480ab3a07` | F0.7 and the F1 pilot |
| 0.3.1 | `090269aa0f3950df` | F0.7b |
| 0.3.2 | `de1be58004986114` | F1b and F2b |
| 0.3.3 | `a31c20d9d8688ea6` | Notebook rev. 4.3 |
| 0.3.4 | `94211b491b3f5d44` | F3, 75 runs; the code frozen at the protocol freeze (2026-10-06) |
| 0.3.5 | `3d563f764976e02b` | Notebook rev. 4.5 |
| 0.3.6 | `e160594adcd46368` | F3, the five "HN only" runs of C2c |
| 0.3.7 | `848167f1c158ebed` | Notebook rev. 4.7 and the F3 sensitivity notebook |
| 0.3.8 | `a466295d82df1b00` | Post hoc analyses; the package at the root of this repository |

To check a version, unzip it and run, from the folder that contains `clm_losses/`:

```bash
python -c "from clm_losses.utils import code_hash; print(code_hash())"
```

Version 0.2.0 was recovered from the notebook that embedded it (rev. 3.2); the others are the package archives saved during the study. Every hash in the table was recomputed from these files.

## Reproducing the study

**Data.** The question–answer pairs are the High-Quality-DQA subset of Nemotron-CC v2.1. Its use requires accepting NVIDIA's data agreement on Hugging Face. CLM's embeddings, token IDs and Gemini negatives come from CLM's Hugging Face repositories. Neither is redistributed here.

**Hardware.** All sessions ran in Google Colab Pro. F0.7 ran on a Tesla T4. The later phases and the post hoc analyses ran on an NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB). The judge of the post hoc analyses (OLMo 3.1 32B) needs that GPU, and the post hoc notebook stops without it. The library versions of each phase are in Appendix G of the article and in the `config.json` of each run.

**Notebooks.** Each notebook embeds its version of the package. Its cells write to Google Drive and resume where they stopped.

1. `notebooks/CLM_losses_Colab_rev4_7.ipynb`: phases F0 to F3, including the protocol freeze and the confirmatory test on the sealed split.
2. `notebooks/CLM_losses_Analisis_F3.ipynb`: sensitivity of the T-gen panel in F3. It only reads the runs.
3. `notebooks/CLM_losses_PreEnvio.ipynb`: post hoc analyses: false negatives (judge and human labels), leakage across splits, the boundary condition of the denominator and untrained references.

The gated generators of the panel (Llama 3.1, Gemma and Aya Expanse) need their licenses accepted on Hugging Face and a token in the Colab secrets (`HF_TOKEN`).

**Figures and tables.** Run this on the study folder:

```bash
python analisis/figuras/figuras_tablas.py --raiz /content/drive/MyDrive/CLM_losses/colab --salida <folder> --idioma en
```

**Tests.**

```bash
pip install torch numpy scipy pandas pyarrow huggingface_hub matplotlib pytest
python -m pytest -q tests
```

The test of the data export (`tests/test_export_release_data.py`) needs only numpy and pytest.

## Data release

The data of the study are released without any text, under CC BY 4.0, in `data/release/`: the archive `release_data.tar.xz` (about 13 MB), cut into parts of 9.5 MB to fit GitHub's limits, with its SHA-256 sums and the export report. To join the parts, check them and extract the archive:

```bash
python release/join_release_data.py        # extracts ./release_data/
```

The release contains:

- the configuration, protocol freeze and addendum generated by the notebook, and the log of the F0 gate decisions;
- the identifiers of the documents in each split;
- the item metadata (`doc_id`, `pair_idx`) and candidate indices of each evaluation set;
- the configuration, training log and per-item results of every run;
- the result files of every phase and analysis;
- the judge's false-negative labels.

The human labels and the judge's labels are also in `data/labels/`, with the annotators anonymized.

The release leaves out:

- the texts of Nemotron-CC v2.1, whose agreement forbids redistribution;
- CLM's Gemini negatives, which CLM distributes;
- the negatives written by the panel generators, as text, token IDs or embeddings. Those of construction E are edits of Nemotron-CC gold answers and keep most of their text, and several generators have non-commercial or custom licenses. The prompts, filters and code that produced them are in `clm_losses/generators.py` and in Appendix B of the article;
- trained checkpoints and the score arrays (`scores_*.npy`).

`release/export_release_data.py` built the release from the study folder in Google Drive, and `release/Exportar_datos.ipynb` ran it in Colab. `data/README.md` describes the archive, what it leaves out and the checks every file passed.

## Registration

`registration/` holds the dated files where the plan was fixed before each step was run. They are internal, dated files, not entries in a public registry; the article says so and lists the deviations (Table 5).

| File | Content |
|---|---|
| `DISENO_01_decisiones_y_F0.md` | Design of 2026-09-30, before any code: decisions D1–D6, the 15 arms and the F0 gates with their thresholds |
| `ENMIENDA_03_A1_y_panel_Tgen.md` | Amendment A1 and the design of the T-gen panel (2026-10-01), with later dated sections: panel v2 and its pilot (§11, 2026-10-02), corrections, C2b and C2c before the freeze (§12–§14, 2026-10-05) and the addendum after the freeze (§15, 2026-10-06) |
| `PROTOCOL_FREEZE.md` | Copy of the freeze that the notebook generated (2026-10-06T02:33:41) and of the C2c addendum (2026-10-07T01:52:08). The original files are in `config/` of the data release |

## Licenses

- **Code** (`clm_losses/`, `code_versions/`, `notebooks/`, `analisis/`, `tests/`, `release/`): Apache License 2.0 (`LICENSE`).
- **Data and documents** (`data/`, the data release and `registration/`): CC BY 4.0 (`LICENSE-DATA.md`).

Third-party data and models are not included and keep their own terms.

## Citation

See `CITATION.cff`. Until the article is published, cite it as: Davis Ortiz, A., Maldonado Robles, M. U., Mendoza Zubia, A., Olvera Gonzalez, J. A., Puga Mota, F., Romero López, J. O., and Zamora Martínez, U. E. *Measuring the Generator, Not the State: The Validity of LLM-Written Negatives for State–Action Scorers*. Manuscript submitted to the Journal of Artificial Intelligence Research.
