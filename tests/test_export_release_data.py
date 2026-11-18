"""The text-free data export (release/export_release_data.py) on a synthetic study folder.

The folder follows the layout that clm_losses v0.3.x writes. Texts are planted in every place where the real folder
holds them, together with a sentinel phrase; the test checks that the sentinel never reaches the export and that the
files the article says are released do reach it. Runs without torch."""
import csv
import hashlib
import importlib.util
import json
import os
import tarfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("export_release_data",
                                              os.path.join(HERE, "..", "release", "export_release_data.py"))
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)

SENTINEL = "which harbor lantern guards the violet quartz mosaic near the falcon bridge"
ANSWER = "the violet quartz mosaic is guarded by the third harbor lantern on the falcon bridge"
GENNEG_RUN = "copper sundial western courtyard marks noon eleventh bell"


def w_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    json.dump(obj, open(p, "w", encoding="utf-8"))


def w_npy(p, arr):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.save(p, arr)


def w_text(p, text):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    open(p, "w", encoding="utf-8").write(text)


def build_study(root):
    n = 6
    # config: study, freeze and addendum (the study's own notes are allowed)
    w_json(f"{root}/config/study.json", {"profile": "colab", "prompts": {"tgen-v1": "Write {k} different answers to the question that are plausible and on-topic but factually INCORRECT."}})
    w_json(f"{root}/config/PROTOCOL_FREEZE.json", {"created": "2026-10-06T02:33:00Z", "code_hash": "94211b491b3f5d44",
                                                   "nota": "C2c queda registrado fuera del JSON del freeze y se ejecuta después de ver los resultados de F3 de los 15 brazos"})
    w_text(f"{root}/config/PROTOCOL_FREEZE.md", "# Protocol freeze\n\nPrimary metric: T-mine.\n")
    w_text(f"{root}/config/PROTOCOL_FREEZE_ADDENDUM.md", "# Addendum\n\nC2c.\n")
    # splits and evaluation sets
    w_text(f"{root}/data/split/test_docs.txt", "\n".join(f"0f8c2a6e-{i:04d}-4c1b-9a77-1b2c3d4e5f60" for i in range(n)))
    w_json(f"{root}/data/split/split.json", {"salt": "x", "n_test_docs": n})
    w_npy(f"{root}/data/eval/test/c0000_q.npy", np.random.rand(n, 4096).astype(np.float16))
    w_npy(f"{root}/data/eval/test/c0000_a.npy", np.random.rand(n, 4096).astype(np.float16))
    w_npy(f"{root}/data/eval/test/c0000_n.npy", np.random.rand(n, 10, 4096).astype(np.float16))
    w_json(f"{root}/data/eval/test/c0000_meta.json", {"doc_id": [f"0f8c2a6e-{i:04d}" for i in range(n)],
                                                      "pair_idx": list(range(n)), "row": list(range(n))})
    w_text(f"{root}/data/eval/test/c0000.done", "ok")
    w_npy(f"{root}/data/eval/test/rand_idx.npy", np.zeros((n, 10), np.int64))
    w_npy(f"{root}/data/eval/test/mine_idx.npy", np.zeros((n, 10), np.int64))
    w_npy(f"{root}/data/eval/test/mine_sims.npy", np.zeros((n, 10), np.float32))
    w_npy(f"{root}/data/eval/test/subsample.npy", np.arange(n))
    g = f"{root}/data/eval/tgen/confirm/phi4_14b/E"
    w_npy(f"{g}/items.npy", np.arange(n))
    w_npy(f"{g}/negs.npy", np.random.rand(n, 10, 4096).astype(np.float16))
    w_npy(f"{g}/neg_tokens.npy", np.random.randint(0, 150000, (n, 10, 24)))
    w_npy(f"{g}/neg_cos.npy", np.random.rand(n, 10).astype(np.float32))
    w_text(f"{g}/kept.jsonl", json.dumps({"i": 0, "positive": None, "negatives": [ANSWER.replace("third", "fourth"),
           f"In the old town the {GENNEG_RUN} for the market"]}) + "\n")
    w_npy(f"{root}/data/eval/tgen/confirm/panel_order.npy", np.arange(n))
    w_text(f"{root}/data/eval/tgen/confirm/QC_falsos_negativos.csv",
           f"i,pregunta,candidato,es_falso_negativo\n0,{SENTINEL},{ANSWER},\n")
    # one run
    r = f"{root}/runs/F3/infonce/seed_0"
    w_json(f"{r}/config.json", {"train": {"lr": 0.002}, "env": {"gpu": "NVIDIA RTX PRO 6000", "code_hash": "94211b491b3f5d44"}})
    w_json(f"{r}/DONE.json", {"status": "ok", "seed": 0, "tmine": 0.959})
    w_text(f"{r}/log.jsonl", json.dumps({"step": 1, "loss": 2.3}) + "\n" + json.dumps(
        {"msg": "etapa pre: 1834 pasos, lr=0.00119, batch=2048, filas=1908359, evaluación cada 100 pasos, selección=valmine_top1"}) + "\n")
    w_text(f"{root}/runs/F1/infonce/arm_log.jsonl", json.dumps({"trial": 0, "val": 0.91}) + "\n")
    w_text(f"{root}/config/F0_DECISIONES.jsonl", json.dumps({"fecha": "2026-10-02", "motivo": "Enmienda A1 aprobada por el investigador como respuesta a G2 y G3 (primaria T-mine, T-gen co-primaria, selección por val-mine) y registrada antes de F1b"}) + "\n")
    w_text(f"{r}/ckpt_last.pt", "binary")
    w_text(f"{r}/best_mid.pt", "binary")
    w_json(f"{r}/eval/metrics.json", {"tgem": {"top1": 0.385}, "tmine": {"top1": 0.959}})
    w_npy(f"{r}/eval/correct_tgem.npy", np.ones(n, np.int8))
    w_npy(f"{r}/eval/argmax_tgem.npy", np.zeros(n, np.int8))
    w_npy(f"{r}/eval/scores_tgem.npy", np.random.rand(n, 11).astype(np.float32))
    w_npy(f"{r}/eval/items_tgen_phi4_14b.npy", np.arange(n))
    # results of the phases and of the post hoc block
    w_json(f"{root}/results/F3/all_vs_all/all_vs_all.json", {"tmine": {"InfoNCE": 0.959}})
    w_text(f"{root}/results/F3/all_vs_all/tabla.csv", "brazo,tmine\nInfoNCE,0.959\n")
    w_text(f"{root}/results/F3/RESUMEN_F3.md", f"# F3\n\nExample: {SENTINEL}\n")
    # pairwise tables: columns A and B hold arm names, not answers
    w_text(f"{root}/results/F1/all_vs_all/pairwise_tgem_top1.csv", "A,B,diff,p_holm\ninfonce,infonce_hn,-0.021,0.003\n")
    w_text(f"{root}/results/F1/ejemplos_a.csv", f"id,a\n0,{ANSWER}\n")
    pre = f"{root}/results/PRE"
    w_text(f"{pre}/textos/confirm.jsonl", json.dumps({"i": "tgem:0", "q": SENTINEL, "a": ANSWER, "cands": [ANSWER]}) + "\n")
    w_json(f"{pre}/b1_calidad/etiquetas/clave.json", {"creado": "2026-10-11", "clave": {"e001": "tgem:0"}})
    w_text(f"{pre}/b1_calidad/etiquetas/ETIQUETAR_200_items.csv",
           f"id,pregunta,oro,n,candidato,es_correcta(0/1),notas\ne001,{SENTINEL},{ANSWER},1,{ANSWER},,\n")
    w_text(f"{pre}/b1_calidad/juez/juez_raw.jsonl",
           json.dumps({"i": "tgem:0", "attempt": 0, "labels": None, "raw": f"I think {SENTINEL}"}) + "\n" +
           json.dumps({"i": "tgem:0", "attempt": 1, "labels": [0, 1] + [0] * 8, "raw": "1: 0\n2: 1"}) + "\n" +
           json.dumps({"i": "tgen_phi4_14b:3", "attempt": 0, "labels": [0] * 10, "raw": "1: 0"}) + "\n")
    # a result file whose note quotes a question: only the overlap check can catch it
    w_json(f"{pre}/b1_calidad/calidad.json", {"tasa": 0.3, "comentario": f"see {SENTINEL} for a case"})
    w_json(f"{pre}/b3_referencias/bm25_corpus.json", {"docs": [ANSWER, SENTINEL]})
    # a short quote under a neutral key: only the overlap with the study's texts can catch it
    w_json(f"{root}/results/F3/C2c/C2c.json", {"delta": 20.53, "caso": "harbor lantern guards the violet quartz mosaic"})
    # notes and decision rules in Spanish are kept; English text is not, whatever the key
    w_json(f"{root}/results/F3/C2/C2.json", {"diff": 0.2, "regla": "C2 = Δ(T-gem) − Δ(macro de T-gen) por par; H1b se sostiene si Δ(T-gem) > +ROPE con IC95 > 0 y C2 > +ROPE con p_Holm < 0.05 (Holm entre los pares)"})
    w_json(f"{root}/results/F3/nota_en.json", {"x": 1, "comment": "This result was computed on the sealed split after the freeze and it is reported here for the record of the study"})
    w_json(f"{root}/results/F3/caso_en.json", {"caso": "What is the grant status of the patent?"})
    w_json(f"{pre}/b1_calidad/fuga/fuga.json", {"n": 3, "pares": [{"pregunta_entrenamiento": "What is the grant status of the patent?", "oro_entrenamiento": "Granted"}]})
    # a generated negative quoted in a note: only the overlap with the panel's texts can catch it
    w_json(f"{root}/results/F3/cita_panel.json", {"comentario": f"ver {GENNEG_RUN}"})
    # the labeling sheets are named after the annotators
    w_text(f"{pre}/b1_calidad/etiquetas/ETIQUETAR_Ana2.xlsx", "sheet")
    w_json(f"{pre}/b1_calidad/etiquetas/acuerdo.json", {"anotadores": {"alumno1": {"archivo": "5ef9a4cc-ETIQUETAR_Ana1.xlsx", "kappa": 0.7}}})
    w_json(f"{pre}/b0_extraccion/REGLA.json", {"regla": "si el kappa juez-humano es menor que 0.6 mandan las etiquetas a mano", "creado": "2026-10-11"})
    w_text(f"{root}/logs/pip_freeze.txt", "numpy==2.1.3\ntorch==2.11.0\n")


def all_text_of(folder):
    out = []
    for dirpath, _, files in os.walk(folder):
        for fn in files:
            out.append(open(os.path.join(dirpath, fn), "rb").read().decode("utf-8", "ignore").lower())
    return "\n".join(out)


def test_export_keeps_numbers_and_drops_texts(tmp_path):
    root, out = str(tmp_path / "study"), str(tmp_path / "release")
    build_study(root)
    ex = exp.Exporter(root, out)
    parts = ex.run()
    got = {r for r, _ in ex.included}
    expected = {
        "config/study.json", "config/PROTOCOL_FREEZE.json", "config/PROTOCOL_FREEZE.md", "config/PROTOCOL_FREEZE_ADDENDUM.md",
        "data/split/test_docs.txt", "data/split/split.json", "data/eval/test/c0000_meta.json",
        "data/eval/test/rand_idx.npy", "data/eval/test/mine_idx.npy", "data/eval/test/mine_sims.npy",
        "data/eval/test/subsample.npy", "data/eval/tgen/confirm/phi4_14b/E/items.npy",
        "data/eval/tgen/confirm/phi4_14b/E/neg_cos.npy",
        "runs/F3/infonce/seed_0/config.json", "runs/F3/infonce/seed_0/DONE.json", "runs/F3/infonce/seed_0/log.jsonl",
        "runs/F3/infonce/seed_0/eval/metrics.json", "runs/F3/infonce/seed_0/eval/correct_tgem.npy",
        "runs/F3/infonce/seed_0/eval/argmax_tgem.npy", "runs/F3/infonce/seed_0/eval/scores_tgem.npy",
        "runs/F3/infonce/seed_0/eval/items_tgen_phi4_14b.npy",
        "results/F3/all_vs_all/all_vs_all.json", "results/F3/all_vs_all/tabla.csv",
        "results/F1/all_vs_all/pairwise_tgem_top1.csv", "results/F3/C2/C2.json", "config/F0_DECISIONES.jsonl",
        "runs/F1/infonce/arm_log.jsonl", "data/eval/tgen/confirm/panel_order.npy",
        "results/PRE/b1_calidad/etiquetas/clave.json", "results/PRE/b0_extraccion/REGLA.json",
        "logs/pip_freeze.txt", "labels/judge_labels.csv",
    }
    assert expected <= got, expected - got
    skipped = {r for r, _ in ex.excluded}
    must_skip = {
        "data/eval/test/c0000_q.npy", "data/eval/test/c0000_a.npy", "data/eval/test/c0000_n.npy",
        "data/eval/tgen/confirm/phi4_14b/E/negs.npy", "data/eval/tgen/confirm/phi4_14b/E/neg_tokens.npy",
        "data/eval/tgen/confirm/phi4_14b/E/kept.jsonl", "data/eval/tgen/confirm/QC_falsos_negativos.csv",
        "runs/F3/infonce/seed_0/ckpt_last.pt", "runs/F3/infonce/seed_0/best_mid.pt", "results/F3/RESUMEN_F3.md",
        "results/PRE/textos/confirm.jsonl", "results/PRE/b1_calidad/etiquetas/ETIQUETAR_200_items.csv",
        "results/PRE/b1_calidad/juez/juez_raw.jsonl", "results/PRE/b1_calidad/calidad.json",
        "results/PRE/b3_referencias/bm25_corpus.json", "results/F3/C2c/C2c.json", "results/F1/ejemplos_a.csv",
        "results/F3/nota_en.json", "results/F3/caso_en.json", "results/PRE/b1_calidad/fuga/fuga.json",
        "results/F3/cita_panel.json", "results/PRE/b1_calidad/etiquetas/acuerdo.json",
    }
    assert must_skip <= skipped, must_skip - skipped
    why = dict(ex.excluded)
    assert why["results/F3/C2c/C2c.json"].startswith("shares text")
    assert why["results/F3/cita_panel.json"].startswith("shares text")
    assert why["results/F3/nota_en.json"] == "contains sentences that are not the study's notes in Spanish"
    assert why["results/F3/caso_en.json"] == "contains English text"
    assert why["results/PRE/b1_calidad/fuga/fuga.json"] == "text field 'pregunta_entrenamiento'"
    assert why["results/PRE/b1_calidad/etiquetas/acuerdo.json"].startswith("names the labeling sheets")
    assert why["results/PRE/b3_referencias/bm25_corpus.json"].startswith("BM25 vocabulary")
    # no planted text anywhere in the export or its zip
    text = all_text_of(out)
    for phrase in (SENTINEL, ANSWER, "violet quartz mosaic", "harbor lantern", "copper sundial", "grant status",
                   "etiquetar_ana", "this result was computed"):
        assert phrase not in text, phrase
    with tarfile.open(out + ".tar.xz", "r:xz") as tf:
        blob = b"".join(tf.extractfile(m).read() for m in tf.getmembers() if m.isfile()).decode("utf-8", "ignore")
    assert "violet quartz" not in blob.lower()
    assert len(parts) == 1 and open(parts[0], "rb").read() == open(out + ".tar.xz", "rb").read()
    # judge labels: first valid answer of each item, without the raw answer
    rows = list(csv.DictReader(open(os.path.join(out, "labels", "judge_labels.csv"), encoding="utf-8")))
    assert [r["label"] for r in rows if r["item"] == "tgem:0"] == ["0", "1"] + ["0"] * 8
    assert {r["attempt"] for r in rows if r["item"] == "tgem:0"} == {"1"}
    assert len(rows) == 20
    # manifest hashes match the files
    man = json.load(open(os.path.join(out, "MANIFEST.json"), encoding="utf-8"))
    for f in man["files"]:
        assert hashlib.sha256(open(os.path.join(out, f["path"]), "rb").read()).hexdigest() == f["sha256"]
    assert os.path.exists(os.path.join(out, "REPORT.md")) and os.path.exists(os.path.join(out, "EXCLUDED.csv"))


def test_no_scores_option(tmp_path):
    root, out = str(tmp_path / "study"), str(tmp_path / "release")
    build_study(root)
    ex = exp.Exporter(root, out, with_scores=False)
    ex.run()
    assert not os.path.exists(os.path.join(out, "runs/F3/infonce/seed_0/eval/scores_tgem.npy"))
    assert os.path.exists(os.path.join(out, "runs/F3/infonce/seed_0/eval/correct_tgem.npy"))


def test_notebook_carries_current_script():
    """release/Exportar_datos.ipynb writes a copy of the script in Colab; the copy must match the script."""
    nb = json.load(open(os.path.join(HERE, "..", "release", "Exportar_datos.ipynb"), encoding="utf-8"))
    head = "%%writefile /content/export_release_data.py\n"
    copies = ["".join(c["source"]) for c in nb["cells"]
              if c["cell_type"] == "code" and "".join(c["source"]).startswith(head)]
    assert len(copies) == 1
    script = open(os.path.join(HERE, "..", "release", "export_release_data.py"), encoding="utf-8").read()
    assert copies[0][len(head):] == script


def test_archive_parts_join(tmp_path, monkeypatch):
    """The archive is cut into parts; the join script checks them and extracts the same files."""
    root, out = str(tmp_path / "study"), str(tmp_path / "release_data")
    build_study(root)
    monkeypatch.setattr(exp, "PART_BYTES", 2000)
    parts = exp.Exporter(root, out).run()
    assert len(parts) > 1 and all(os.path.getsize(p) <= 2000 for p in parts)
    assert b"".join(open(p, "rb").read() for p in parts) == open(out + ".tar.xz", "rb").read()
    spec_j = importlib.util.spec_from_file_location("join_release_data",
                                                    os.path.join(HERE, "..", "release", "join_release_data.py"))
    join = importlib.util.module_from_spec(spec_j)
    spec_j.loader.exec_module(join)
    dest = str(tmp_path / "joined")
    top = join.main(["--parts", str(tmp_path), "--out", dest])
    man = json.load(open(os.path.join(top, "MANIFEST.json"), encoding="utf-8"))
    assert len(man["files"]) > 20
    for f in man["files"]:
        assert open(os.path.join(top, f["path"]), "rb").read() == open(os.path.join(out, f["path"]), "rb").read()
    # a damaged part is refused
    open(parts[0], "ab").write(b"x")
    try:
        join.main(["--parts", str(tmp_path), "--out", str(tmp_path / "joined2")])
        raise AssertionError("a damaged part was accepted")
    except SystemExit as e:
        assert "does not match" in str(e)
