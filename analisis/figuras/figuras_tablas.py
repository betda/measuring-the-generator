#!/usr/bin/env python3
"""Figuras y tablas del manuscrito (CLM losses → JAIR), generadas desde los resultados crudos.

Uso (Colab, con Drive montado):
    python analisis/figuras/figuras_tablas.py --raiz /content/drive/MyDrive/CLM_losses/colab \
        --salida /content/drive/MyDrive/CLM_losses/colab/results/FIGURAS [--idioma es|en]

Uso local: el mismo comando con --raiz apuntando a un espejo de Drive con las mismas rutas relativas.

Sólo lee JSON ya guardados: no entrena, no evalúa y no descarga nada. Ninguna cifra está escrita a mano:
todo sale de los archivos de ENTRADAS. Cada salida queda en MANIFIESTO.json con el sha256 de sus
entradas y de sí misma; LEEME.md trae, por salida, la pregunta que responde, su estatus (confirmatorio,
descriptivo o exploratorio), sus fuentes y una leyenda completa.
"""
import argparse
import csv
import datetime
import hashlib
import json
import os
import platform
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

VERSION = "1.4 · 2026-11-16"

# Paleta (skill dataviz): las tres primeras ranuras categóricas, validadas para todos los pares en modo
# claro (CVD ΔE ≥ 9.2; visión normal ≥ 24). El aqua queda bajo 3:1 de contraste: por eso cada serie lleva
# además su propia forma de marcador (sirve también en escala de grises) y cada figura su CSV.
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, GRIS, GRIS_CLARO = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
AZAR = 1 / 11  # 1 oro + 10 negativos

ENTRADAS = {
    "f06": "f0/F0.6_shortcuts/result.json",
    "build": "data/eval/tgen/BUILD.json",
    "panel_test": "data/eval/tgen/test/PANEL.json",
    "panel_conf": "data/eval/tgen/confirm/PANEL.json",
    "avs": "results/F3/all_vs_all/all_vs_all.json",
    "c2": "results/F3/C2/C2.json",
    "c2b": "results/F3/C2b/C2b.json",
    "c2c": "results/F3/C2c/C2c.json",
    "sens_panel": "results/F3/sensibilidad_panel/sensibilidad.json",
    "extr": "results/PRE/b0_extraccion/extraccion.json",
    "cal": "results/PRE/b1_calidad/calidad.json",
    "jvh": "results/PRE/b1_calidad/juez_vs_humanos.json",
    "acu": "results/PRE/b1_calidad/etiquetas/acuerdo.json",
    "borde": "results/PRE/b2_borde/borde.json",
    "refs": "results/PRE/b3_referencias/referencias.json",
    # Fig. de selección (opcional): HPO del F1 piloto y de F1b, y el test exploratorio de cada fase
    "hpo_f1_siglip": "runs/F1/siglip/hpo_summary.json",
    "hpo_f1b_siglip": "runs/F1b/siglip/hpo_summary.json",
    "hpo_f1_infonce_hn": "runs/F1/infonce_hn/hpo_summary.json",
    "avs_f1": "results/F1/all_vs_all/all_vs_all.json",
    "avs_f1b": "results/F1b/all_vs_all/all_vs_all.json",
}
OPCIONALES = {"acu", "hpo_f1_siglip", "hpo_f1b_siglip", "hpo_f1_infonce_hn", "avs_f1", "avs_f1b"}

BRAZOS = ["infonce", "infonce_hn", "siglip", "siglip_hn", "triplet_hn", "simsiam_x", "byol_x", "barlow_x",
          "vicreg_x", "mse_sigreg", "infonce_sigreg", "align_unif", "infonce_lse", "dino_x", "swav_x"]
NOMBRE_BRAZO = {"infonce": "InfoNCE", "infonce_hn": "InfoNCE+HN", "siglip": "SigLIP", "siglip_hn": "SigLIP+HN",
                "triplet_hn": "triplet+HN", "simsiam_x": "SimSiam-x", "byol_x": "BYOL-x", "barlow_x": "Barlow-x",
                "vicreg_x": "VICReg-x", "mse_sigreg": "MSE+SIGReg", "infonce_sigreg": "InfoNCE+SIGReg",
                "align_unif": {"es": "alineación+unif.", "en": "align.+unif."}, "infonce_lse": "InfoNCE+LSE",
                "dino_x": "DINO-x", "swav_x": "SwAV-x"}
# Grupos para el color de la Fig. de brazos (no son las familias de la Tabla 1): qué negativos usa la pérdida.
GRUPO = {**{a: "hn" for a in ("infonce_hn", "siglip_hn", "triplet_hn")},
         **{a: "inbatch" for a in ("infonce", "infonce_lse", "infonce_sigreg", "siglip", "align_unif")},
         **{a: "sinneg" for a in ("vicreg_x", "barlow_x", "mse_sigreg", "simsiam_x", "byol_x", "dino_x", "swav_x")}}
MACRO = ["aya32b", "glm4_32b", "gptoss20b", "mistral_small24b", "phi4_14b"]
NOMBRE_GEN = {"aya32b": "Aya Expanse 32B", "glm4_32b": "GLM-4 32B", "gptoss20b": "gpt-oss-20b",
              "mistral_small24b": "Mistral Small 24B", "phi4_14b": "Phi-4 (14B)", "mistral7b": "Mistral 7B",
              "granite33_8b": "Granite 3.3 8B", "llama31_8b": "Llama 3.1 8B"}
SETS_FN = ["tgem", "tmine"] + [f"tgen_{g}" for g in MACRO]

# ---------------------------------------------------------------------------------------------------------
# Textos (español para el borrador; inglés para la versión final)
# ---------------------------------------------------------------------------------------------------------
TXT = {
    "es": {
        "azar": "azar", "umbral30": "umbral del macro (30 %)", "criterio5": "criterio de F0.5 (5 %)",
        "pp": "pp", "top1": "Top-1 (%)",
        # fig atajos
        "atajos_x": "Acierto sin ver el estado (%)",
        "tgem_row": "T-gem: HN de Gemini (benchmark de CLM)", "tmine_row": "T-mine: oros minados de otros documentos",
        "trand_row": "T-rand: oros de otros documentos al azar",
        "S_row": "Piloto S: respuestas incorrectas libres", "P_row": "Piloto P: oro reescrito",
        "E_row": "Piloto E: edición del dato clave", "panel_test_row": "Panel E, test: 23 generadores",
        "panel_conf_row": "Panel E, sellado: los 5 del macro",
        "leg_ciego": "ciego (sólo ve las acciones)", "leg_larga": "elegir la respuesta más larga (T-gem)",
        "leg_macro": "generador en el macro", "leg_primario": "primario fuera del macro", "leg_control": "control del panel (familias Gemini y Qwen)",
        "leg_atajo": "atajo ciego: la mejor sonda que no ve el estado (en T-gem, T-mine y T-rand, el modelo ciego)",
        "leg_primaria": "set de una métrica primaria (T-mine; los 5 generadores del macro)", "leg_mediana": "mediana del piloto",
        # fig brazos
        "brazos_a": "(a) T-mine (negativos minados)", "brazos_b": "(b) Macro de T-gen (5 generadores)",
        "g_hn": "con HN (Gemini)", "g_inbatch": "con negativos in-batch, sin HN", "g_sinneg": "sin negativos por instancia",
        # fig transferencia
        "transf_a": "(a) Ganancia por HN: negativos de Gemini (T-gem) frente a otros generadores (T-gen)",
        "transf_b": "(b) Descomposición de la brecha (C2b)",
        "dtgem": "Δ en T-gem (sus ítems)", "dtgen": "Δ en T-gen", "macro_row": "Macro (5 generadores)",
        "gan_x": "Ganancia por HN (pp)", "comp_gen": "generador",
        "comp_con": "tipo de negativo (construcción)", "brecha": "brecha total", "comp_x": "Componente de la brecha (pp)",
        # fig denominador
        "den_a": "(a) F3, test sellado: mecanismo (C2c) y controles",
        "den_b": "(b) Condición de borde de B5 (split de test; exploratoria)", "den_b1": "", "equiv": "equivalente, TOST ±1 pp",
        "den_b2": "(c) Misma condición: T-mine",
        "c2c_shuf": "sólo HN − HN + in-batch · T-gem barajado", "c2c_tmine": "sólo HN − HN + in-batch · T-mine",
        "c2c_ctl": "control: InfoNCE+HN − InfoNCE · T-gem barajado",
        "sig_shuf": "SigLIP+HN − SigLIP · T-gem barajado (descriptivo)", "delta_pp": "Diferencia (pp)",
        "conf": "pre-especificado", "desc": "descriptivo",
        "cond_a": "(a) sólo HN", "cond_b": "(b) HN + in-batch", "cond_c": "(c) + HN ajenos (B − 1)",
        "cond_c31": "(c31) + 31 HN ajenos", "cond_d31": "(d31) + 31 oros ajenos", "cond_e": "(e) compartidos, como NegCLIP",
        "tipo_ref": "referencias (a) y (b)", "tipo_hn": "candidatos extra: HN de otros estados",
        "tipo_oro": "candidatos extra: oros de otros estados", "tipo_comp": "HN y oros compartidos",
        "shuf_x": "T-gem con estados barajados (%)", "tmine_x": "T-mine (%)",
        # fig falsos negativos
        "fn_x": "Preguntas con al menos un negativo también correcto (%)",
        "fn_estricta": "humanos, referencia estricta (los dos)", "fn_amplia": "humanos, referencia amplia (al menos uno)",
        "fn_juez": "juez LLM (todo el sellado)",
        # fig selección
        "sel_a": "(a) Validación: 32 trials, dos métricas", "sel_x": "T-gem-val: negativos de Gemini (%)",
        "sel_y": "val-mine: negativos minados (%)", "sel_lab1": "elegido por T-gem-val\n(trial {k})",
        "sel_lab2": "elegido por val-mine\n(trial {k})", "sel_rho": "ρ de Spearman = {rho:.2f} ({n} trials)",
        "sel_b": "(b) Test: los dos modelos finales", "sel_legt": "modelo elegido por", "sel_leg1": "T-gem-val (F1 piloto)",
        "sel_leg2": "val-mine (F1b)", "sel_top1": "Top-1 en test (%)",
    },
    "en": {
        "azar": "chance", "umbral30": "macro threshold (30%)", "criterio5": "F0.5 criterion (5%)",
        "pp": "pp", "top1": "Top-1 (%)",
        "atajos_x": "Accuracy without the state (%)",
        "tgem_row": "T-gem: Gemini HNs (CLM benchmark)", "tmine_row": "T-mine: golds mined from other documents",
        "trand_row": "T-rand: random golds from other documents",
        "S_row": "Pilot S: free wrong answers", "P_row": "Pilot P: rewritten gold",
        "E_row": "Pilot E: key-fact edits", "panel_test_row": "Panel E, test: 23 generators",
        "panel_conf_row": "Panel E, sealed: the 5 macro generators",
        "leg_ciego": "blind model (sees only the actions)", "leg_larga": "pick the longest answer (T-gem)",
        "leg_macro": "generator in the macro", "leg_primario": "primary, not in the macro", "leg_control": "panel control (Gemini and Qwen families)",
        "leg_atajo": "shortcut score: best probe that never sees the state (T-gem, T-mine, T-rand: the blind model)",
        "leg_primaria": "set behind a primary metric (T-mine; the 5 macro generators)", "leg_mediana": "pilot median",
        "brazos_a": "(a) T-mine (mined negatives)", "brazos_b": "(b) T-gen macro (5 generators)",
        "g_hn": "with HNs (Gemini)", "g_inbatch": "in-batch negatives, no HNs", "g_sinneg": "no per-instance negatives",
        "transf_a": "(a) HN gain: Gemini negatives (T-gem) vs. other generators (T-gen)",
        "transf_b": "(b) Decomposition of the gap (C2b)",
        "dtgem": "Δ on T-gem (same items)", "dtgen": "Δ on T-gen", "macro_row": "Macro (5 generators)",
        "gan_x": "HN gain (pp)", "comp_gen": "generator",
        "comp_con": "negative type (construction)", "brecha": "total gap", "comp_x": "Gap component (pp)",
        "den_a": "(a) F3, sealed test: mechanism (C2c) and controls",
        "den_b": "(b) B5 boundary condition (test split; exploratory)", "den_b1": "", "equiv": "equivalent, TOST ±1 pp",
        "den_b2": "(c) Same condition: T-mine",
        "c2c_shuf": "HN only − HN + in-batch · shuffled T-gem", "c2c_tmine": "HN only − HN + in-batch · T-mine",
        "c2c_ctl": "control: InfoNCE+HN − InfoNCE · shuffled T-gem",
        "sig_shuf": "SigLIP+HN − SigLIP · shuffled T-gem (descriptive)", "delta_pp": "Difference (pp)",
        "conf": "pre-specified", "desc": "descriptive",
        "cond_a": "(a) HN only", "cond_b": "(b) HN + in-batch", "cond_c": "(c) + other states' HNs (B − 1)",
        "cond_c31": "(c31) + 31 other HNs", "cond_d31": "(d31) + 31 other golds", "cond_e": "(e) shared, NegCLIP-style",
        "tipo_ref": "references (a) and (b)", "tipo_hn": "extra candidates: other states' HNs",
        "tipo_oro": "extra candidates: other states' golds", "tipo_comp": "shared HNs and golds",
        "shuf_x": "T-gem with shuffled states (%)", "tmine_x": "T-mine (%)",
        "fn_x": "Questions with at least one negative that is also correct (%)",
        "fn_estricta": "humans, strict reference (both)", "fn_amplia": "humans, lenient reference (at least one)",
        "fn_juez": "LLM judge (full sealed set)",
        "sel_a": "(a) Validation: 32 trials, two metrics", "sel_x": "T-gem-val: Gemini negatives (%)",
        "sel_y": "val-mine: mined negatives (%)", "sel_lab1": "chosen by T-gem-val\n(trial {k})",
        "sel_lab2": "chosen by val-mine\n(trial {k})", "sel_rho": "Spearman ρ = {rho:.2f} ({n} trials)",
        "sel_b": "(b) Test: the two final models", "sel_legt": "model chosen by", "sel_leg1": "T-gem-val (pilot F1)",
        "sel_leg2": "val-mine (F1b)", "sel_top1": "Test top-1 (%)",
    },
}
LECTURAS_ES = {"infonce mejor": "InfoNCE mejor",  # versión corta para la tabla; el texto completo va en el CSV
               "H1b se sostiene: la ganancia por HN cae al pasar a otros generadores": "H1b se sostiene",
               "se confirma: explica parte de la brecha": "se confirma"}
LECTURAS_EN = {  # lecturas que escribe el pipeline (en español) → inglés
    "infonce mejor": "InfoNCE better", "se confirma": "confirmed", "se confirma (equivalente)": "confirmed (equivalent)",
    "H1b se sostiene: la ganancia por HN cae al pasar a otros generadores": "H1b holds",
    "se confirma: explica parte de la brecha": "confirmed",
}

# ---------------------------------------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------------------------------------
MANIF = {"entradas": {}, "salidas": {}}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def cargar(raiz):
    d, falt = {}, []
    for k, rel in ENTRADAS.items():
        p = os.path.join(raiz, rel)
        if not os.path.exists(p):
            falt.append(rel)
            continue
        with open(p, encoding="utf-8") as f:
            d[k] = json.load(f)
        MANIF["entradas"][rel] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    return d, falt


def nb(arm, idioma):
    v = NOMBRE_BRAZO[arm]
    return v[idioma] if isinstance(v, dict) else v


def pct(x):
    return 100.0 * x


def menos(s):
    return s.replace("-", "−")


def f_num(x, nd=1, signo=False):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    s = f"{x:+.{nd}f}" if signo else f"{x:.{nd}f}"
    return menos(s)


def f_ci(lo, hi, nd=1, signo=False):
    return f"[{f_num(lo, nd, signo)}, {f_num(hi, nd, signo)}]"


def f_p(p):
    if p is None:
        return "—"
    if p == 0:
        return "< 1e-300"
    if p < 1e-3:
        m, e = f"{p:.1e}".split("e")
        return f"{m}e{int(e)}"
    if p < 0.01:
        return f"{p:.4f}"
    return f"{p:.3f}" if p < 0.1 else f"{p:.2f}"


def a_tex(s):
    """Celda de Markdown → LaTeX (booktabs)."""
    s = str(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    s = re.sub(r"(\d(?:\.\d+)?)e(−?-?\d+)", lambda m: f"${m.group(1)}\\times10^{{{m.group(2).replace('−', '-')}}}$", s)
    for a, b in (("\\", "\\"), ("&", "\\&"), ("%", "\\%"), ("_", "\\_"), ("#", "\\#")):
        if a != "\\":
            s = s.replace(a, b)
    rep = {"−": "$-$", "≤": "$\\leq$", "≥": "$\\geq$", "×": "$\\times$", "κ": "$\\kappa$", "Δ": "$\\Delta$",
           "→": "$\\rightarrow$", "±": "$\\pm$", "≈": "$\\approx$", "–": "--", "< 1e": "$<$ 1e"}
    for a, b in rep.items():
        s = s.replace(a, b)
    s = s.replace("$<$ 1e-300", "$<10^{-300}$").replace("$<$ 1e$-$300", "$<10^{-300}$")
    return s


def escribir_csv(p, filas):
    cols = list(dict.fromkeys(k for f in filas for k in f))  # unión ordenada de columnas
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, restval="")
        w.writeheader()
        w.writerows(filas)


class Salida:
    def __init__(self, raiz_salida, idioma):
        self.raiz, self.idioma = raiz_salida, idioma
        for sub in ("figuras", "tablas", "datos_figuras"):
            os.makedirs(os.path.join(raiz_salida, sub), exist_ok=True)
        self.leeme = []

    def _reg(self, rel, fuentes):
        p = os.path.join(self.raiz, rel)
        MANIF["salidas"][rel] = {"sha256": sha256(p), "fuentes": sorted(ENTRADAS[k] for k in fuentes)}

    def figura(self, fig, nombre, fuentes, pregunta, estatus, leyenda, ubicacion):
        for ext in ("pdf", "png"):
            rel = f"figuras/{nombre}.{ext}"
            fig.savefig(os.path.join(self.raiz, rel), dpi=300, bbox_inches="tight",
                        metadata={"Creator": f"figuras_tablas.py {VERSION}"} if ext == "pdf" else None)
            self._reg(rel, fuentes)
        plt.close(fig)
        self.leeme.append(("figura", nombre, ubicacion, pregunta, estatus, fuentes, leyenda))

    def datos(self, nombre, filas, fuentes):
        rel = f"datos_figuras/{nombre}.csv"
        if filas:
            escribir_csv(os.path.join(self.raiz, rel), filas)
            self._reg(rel, fuentes)

    def tabla(self, nombre, cab, filas, csv_filas, fuentes, pregunta, estatus, leyenda, ubicacion, notas=None):
        md = ["| " + " | ".join(cab) + " |", "|" + "|".join("---" for _ in cab) + "|"]
        md += ["| " + " | ".join(str(c) for c in f) + " |" for f in filas]
        if notas:
            md += [""] + [f"{n}" for n in notas]
        with open(os.path.join(self.raiz, f"tablas/{nombre}.md"), "w", encoding="utf-8") as f:
            f.write(f"**{leyenda}**\n\n" + "\n".join(md) + "\n")
        ancho = len(cab) >= 6 or max(len(" ".join(str(c) for c in f)) for f in filas + [cab]) > 90
        tex = ["\\begin{table}[t]", "\\centering", "\\small",
               f"\\caption{{{a_tex(leyenda)}}}", f"\\label{{tab:{nombre}}}",
               ("\\resizebox{\\linewidth}{!}{%" if ancho else "%"),
               "\\begin{tabular}{" + "l" * len(cab) + "}", "\\toprule",
               " & ".join(a_tex(c) for c in cab) + " \\\\", "\\midrule"]
        tex += [" & ".join(a_tex(c) for c in f) + " \\\\" for f in filas]
        tex += ["\\bottomrule", "\\end{tabular}" + ("}" if ancho else "")]
        if notas:
            tex += ["\\par\\smallskip\\footnotesize " + " ".join(a_tex(n) for n in notas)]
        tex += ["\\end{table}"]
        with open(os.path.join(self.raiz, f"tablas/{nombre}.tex"), "w", encoding="utf-8") as f:
            f.write("\n".join(tex) + "\n")
        escribir_csv(os.path.join(self.raiz, f"tablas/{nombre}.csv"), csv_filas)
        for ext in ("md", "tex", "csv"):
            self._reg(f"tablas/{nombre}.{ext}", fuentes)
        self.leeme.append(("tabla", nombre, ubicacion, pregunta, estatus, fuentes, leyenda))


def estilo():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.8, "axes.linewidth": 0.6,
        "axes.edgecolor": TINTA2, "axes.labelcolor": TINTA, "xtick.color": TINTA2, "ytick.color": TINTA,
        "xtick.major.width": 0.6, "ytick.major.width": 0, "xtick.major.size": 2.5, "ytick.major.size": 0,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRIS_CLARO, "grid.linewidth": 0.5,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.titlelocation": "left",
        "axes.titleweight": "bold", "axes.titlepad": 12, "savefig.pad_inches": 0.03,
    })


def linea_ref(ax, x, texto, estiloL="--", y_texto=1.0, ha="left"):
    ax.axvline(x, color=GRIS, lw=0.7, ls=estiloL, zorder=1)
    ax.text(x, y_texto, f" {texto}" if ha == "left" else f"{texto} ", transform=ax.get_xaxis_transform(),
            color=TINTA2, fontsize=6.3, va="bottom", ha=ha)


def punto_ci(ax, y, m, lo, hi, color, marker, relleno=True, ms=4.2, z=3):
    ax.plot([lo, hi], [y, y], color=color, lw=1.0, solid_capstyle="round", zorder=z)
    ax.plot([m], [y], marker=marker, ms=ms, color=color, mfc=color if relleno else "white", mew=0.9, zorder=z + 1,
            ls="none")


# ---------------------------------------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------------------------------------
def fig_atajos(d, S, t, idioma):
    f06, build, pt, pc = d["f06"], d["build"], d["panel_test"], d["panel_conf"]
    filas_csv = []
    filas = [("tgem_row", None), ("tmine_row", None), ("trand_row", None), ("S_row", "S"), ("P_row", "P"),
             ("E_row", "E"), ("panel_test_row", "test"), ("panel_conf_row", "conf")]
    fig, ax = plt.subplots(figsize=(6.3, 2.9))
    ys = {k: len(filas) - 1 - i for i, (k, _) in enumerate(filas)}
    rng = np.random.default_rng(0)  # sólo para separar puntos superpuestos en vertical (no toca los valores)
    for clave, set_k in (("tgem", "tgem_row"), ("tmine", "tmine_row"), ("trand", "trand_row")):
        v = f06["blind"][clave]
        ax.plot([pct(v)], [ys[set_k]], marker="o", ms=4.2, color=AZUL, mfc=AZUL if clave == "tmine" else "white",
                mew=1.0, ls="none", zorder=4)
        filas_csv.append({"fila": t[set_k], "sonda": "ciego", "generador": "", "acierto_pct": round(pct(v), 3)})
    vl = f06["length_heuristic"]["test"]
    ax.plot([pct(vl)], [ys["tgem_row"]], marker="D", ms=4.0, color=NARANJA, mfc="white", mew=1.0, ls="none", zorder=4)
    filas_csv.append({"fila": t["tgem_row"], "sonda": "la más larga", "generador": "", "acierto_pct": round(pct(vl), 3)})
    ax.annotate(f"{pct(f06['blind']['tgem']):.1f}", (pct(f06["blind"]["tgem"]), ys["tgem_row"]), xytext=(4, 3),
                textcoords="offset points", fontsize=6.3, color=TINTA2)
    ax.annotate(f"{pct(vl):.1f}", (pct(vl), ys["tgem_row"]), xytext=(4, 3), textcoords="offset points",
                fontsize=6.3, color=TINTA2)
    for c in ("S", "P", "E"):
        res = build["pilot"]["results"][c]
        vals = [pct(v) for v in res.values()]
        y = ys[f"{c}_row"]
        ax.plot(vals, y + rng.uniform(-0.12, 0.12, len(vals)), marker="o", ms=3.8, color=AZUL, mfc="white",
                mew=0.9, ls="none", zorder=4)
        ax.plot([pct(build["medianas"][c])], [y], marker="|", ms=9, mew=1.4, color=TINTA, ls="none", zorder=5)
        for g, v in res.items():
            filas_csv.append({"fila": t[f"{c}_row"], "sonda": "atajo ciego", "generador": g, "acierto_pct": round(pct(v), 3)})
    for pan, fila in ((pt, "panel_test_row"), (pc, "panel_conf_row")):
        y = ys[fila]
        claves = [k for k in pan["generators"] if fila == "panel_test_row" or k in pan["macro"]]
        jit = rng.uniform(-0.18, 0.18, len(claves))
        for j, k in zip(jit, claves):
            g = pan["generators"][k]
            if g.get("atajo_ciego") is None:
                continue
            en_macro = k in pan["macro"]
            mk = "s" if str(g.get("role", "")).startswith("control") else "o"
            ax.plot([pct(g["atajo_ciego"])], [y + j], marker=mk, ms=3.8 if mk == "o" else 3.4, color=AZUL,
                    mfc=AZUL if en_macro else "white", mew=0.9, ls="none", zorder=4)
            filas_csv.append({"fila": t[fila], "sonda": "atajo ciego", "generador": k,
                              "acierto_pct": round(pct(g["atajo_ciego"]), 3), "en_macro": en_macro,
                              "rol": g.get("role"), "tamano": g.get("tier"), "banda": g.get("banda")})
    ax.set_yticks([ys[k] for k, _ in filas])
    ax.set_yticklabels([t[k] for k, _ in filas])
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, len(filas) - 0.4)
    ax.set_xlabel(t["atajos_x"])
    linea_ref(ax, pct(AZAR), t["azar"])
    linea_ref(ax, 30, t["umbral30"], ":")
    leg = [Line2D([], [], marker="o", ls="none", color=AZUL, mfc="white", mew=1.0, ms=4.2, label=t["leg_atajo"]),
           Line2D([], [], marker="o", ls="none", color=AZUL, ms=4.2, label=t["leg_primaria"]),
           Line2D([], [], marker="s", ls="none", color=AZUL, mfc="white", mew=0.9, ms=3.4, label=t["leg_control"]),
           Line2D([], [], marker="D", ls="none", color=NARANJA, mfc="white", mew=1.0, ms=4.0, label=t["leg_larga"]),
           Line2D([], [], marker="|", ls="none", color=TINTA, ms=9, mew=1.4, label=t["leg_mediana"])]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.42, -0.2), ncol=2, handletextpad=0.3,
              columnspacing=1.5)
    S.datos("fig_atajos", filas_csv, ["f06", "build", "panel_test", "panel_conf"])
    n_alto = sum(1 for g in pt["generators"].values() if g.get("banda") == "alto")
    if idioma == "es":
        ley = (f"Atajos de cada conjunto de evaluación: acierto top-1 de sondas que no ven el estado (azar = 1/11 = "
               f"{pct(AZAR):.1f} %). En T-gem, el benchmark de CLM con negativos de Gemini, un modelo ciego que sólo "
               f"ve las acciones acierta {pct(f06['blind']['tgem']):.1f} % y elegir la respuesta más larga, "
               f"{pct(vl):.1f} % (F0.6, test). En los negativos minados (T-mine) el ciego queda en "
               f"{pct(f06['blind']['tmine']):.1f} %. Piloto: tres construcciones con los mismos 500 ítems y 3 "
               f"generadores (medianas: S {pct(build['medianas']['S']):.1f}, P {pct(build['medianas']['P']):.1f}, "
               f"E {pct(build['medianas']['E']):.1f} %); la regla fija eligió E. Panel E: atajo ciego por generador "
               f"(máximo de las sondas sin estado); {n_alto} de los 23 quedan en banda alta (≥ 30 %). Relleno: "
               f"sets de las métricas primarias (T-mine y los 5 generadores del macro); cuadrado: control. La dispersión vertical dentro de cada fila sólo separa "
               f"puntos superpuestos.")
    else:
        ley = (f"Shortcuts in each evaluation set: top-1 accuracy of probes that never see the state (chance = 1/11 = "
               f"{pct(AZAR):.1f}%). On T-gem, CLM's benchmark with Gemini negatives, a blind model that sees only the "
               f"actions reaches {pct(f06['blind']['tgem']):.1f}% and picking the longest answer "
               f"{pct(vl):.1f}% (F0.6, test). On mined negatives (T-mine) the blind model stays at "
               f"{pct(f06['blind']['tmine']):.1f}%. Pilot: three constructions on the same 500 items and 3 generators "
               f"(medians: S {pct(build['medianas']['S']):.1f}, P {pct(build['medianas']['P']):.1f}, "
               f"E {pct(build['medianas']['E']):.1f}%); the fixed rule chose E. Panel E: per-generator shortcut score "
               f"(max over state-free probes); {n_alto} of the 23 fall in the high band (≥ 30%). Filled: sets behind "
               f"the primary metrics (T-mine and the 5 macro generators); square: control. Vertical spread within a row only separates overlapping points.")
    S.figura(fig, "fig_atajos", ["f06", "build", "panel_test", "panel_conf"],
             "¿Cuánto de cada conjunto de evaluación se resuelve sin mirar el estado?", "descriptivo (F0, piloto y auditoría del panel)",
             ley, "Resultados I (planeada como Fig. 2)")


def fig_brazos(d, S, t, idioma):
    m = d["avs"]["metrics"]
    tm = {r["arm"]: r for r in m["tmine_top1"]["summary"]}
    tg = {r["arm"]: r for r in m["tgenmacro_top1"]["summary"]}
    orden = sorted(BRAZOS, key=lambda a: tm[a]["mean"])
    fig, axs = plt.subplots(1, 2, figsize=(6.3, 3.3), sharey=True, gridspec_kw={"wspace": 0.08})
    estilo_g = {"hn": (NARANJA, "s"), "inbatch": (AZUL, "o"), "sinneg": (AQUA, "^")}
    filas_csv = []
    for ax, met, tit in ((axs[0], tm, t["brazos_a"]), (axs[1], tg, t["brazos_b"])):
        for y, a in enumerate(orden):
            c, mk = estilo_g[GRUPO[a]]
            r = met[a]
            punto_ci(ax, y, pct(r["mean"]), pct(r["ci95_lo"]), pct(r["ci95_hi"]), c, mk)
        ax.set_title(tit)
        linea_ref(ax, pct(AZAR), t["azar"], y_texto=1.0)
        ax.set_xlabel(t["top1"])
    for a in orden:
        filas_csv.append({"brazo": a, "grupo": GRUPO[a],
                          "tmine_media": tm[a]["mean"], "tmine_lo": tm[a]["ci95_lo"], "tmine_hi": tm[a]["ci95_hi"],
                          "tgenmacro_media": tg[a]["mean"], "tgenmacro_lo": tg[a]["ci95_lo"], "tgenmacro_hi": tg[a]["ci95_hi"],
                          "semillas": tm[a]["n_seeds"]})
    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, max(pct(tg[a]["ci95_hi"]) for a in orden) * 1.12)
    axs[0].set_yticks(range(len(orden)))
    axs[0].set_yticklabels([nb(a, idioma) for a in orden])
    for lab in axs[0].get_yticklabels():
        if lab.get_text() in (nb("infonce", idioma), nb("vicreg_x", idioma)):
            lab.set_fontweight("bold")
    leg = [Line2D([], [], marker=mk, ls="-", lw=1.0, color=c, ms=4.2, label=t[f"g_{g}"])
           for g, (c, mk) in estilo_g.items()]
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.55, 0.02), ncol=3)
    S.datos("fig_brazos", filas_csv, ["avs"])
    c1 = d["avs"]["C1"]["por_metrica"]
    if idioma == "es":
        ley = ("Acierto top-1 de los 15 brazos en el test sellado (F3; media de 5 semillas, IC95 por bootstrap "
               "jerárquico pareado por documento). (a) T-mine, con 10 negativos minados de otros documentos; (b) macro "
               "de T-gen, con negativos editados por 5 generadores. Ordenados por T-mine. Color y forma: qué negativos "
               "usa la pérdida; en T-mine los IC95 son más angostos que el marcador. En negrita, los dos brazos de C1: VICReg-x − InfoNCE = "
               f"{f_num(pct(c1['tmine']['diff']), 2, True)} pp en T-mine y {f_num(pct(c1['tgenmacro']['diff']), 2, True)} pp "
               "en el macro (Tabla de contrastes). SimSiam-x, BYOL-x y DINO-x colapsan con nuestra adaptación.")
    else:
        ley = ("Top-1 accuracy of the 15 arms on the sealed test (F3; mean of 5 seeds, 95% CI from a hierarchical "
               "bootstrap paired by document). (a) T-mine, 10 negatives mined from other documents; (b) T-gen macro, "
               "negatives edited by 5 generators. Sorted by T-mine. Color and shape: which negatives the loss uses; on T-mine "
               "the 95% CIs are narrower than the markers. "
               f"Bold: the two C1 arms; VICReg-x − InfoNCE = {f_num(pct(c1['tmine']['diff']), 2, True)} pp on T-mine and "
               f"{f_num(pct(c1['tgenmacro']['diff']), 2, True)} pp on the macro (contrasts table). SimSiam-x, BYOL-x and "
               "DINO-x collapse under our adaptation.")
    S.figura(fig, "fig_brazos", ["avs"], "¿Qué pérdidas discriminan mejor con negativos minados y con negativos editados?",
             "confirmatorio (C1); el orden completo es descriptivo", ley, "Resultados II (planeada como Fig. 3)")


def fig_transferencia(d, S, t, idioma):
    sens = {f["subconjunto"]: f for f in d["sens_panel"]["filas"]}
    c2 = {p["par"]: p for p in d["c2"]["pares"]}
    pares = ["infonce_hn − infonce", "siglip_hn − siglip"]
    fig = plt.figure(figsize=(6.3, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.55, 0.9], hspace=0.7, wspace=0.08, top=0.86)
    filas_csv = []
    filas = MACRO + ["macro"]
    for j, par in enumerate(pares):
        ax = fig.add_subplot(gs[0, j])
        for i, g in enumerate(filas):
            y = len(filas) - 1 - i
            if g == "macro":
                r = c2[par]
                a, b = r["Δ T-gem"], r["Δ macro T-gen"]
            else:
                r = sens[f"sólo {g}"][par]
                a, b = r["Δ T-gem"], r["Δ T-gen"]
            ax.plot([pct(b["diff"]), pct(a["diff"])], [y, y], color=GRIS_CLARO, lw=2.2, zorder=1, solid_capstyle="round")
            punto_ci(ax, y, pct(a["diff"]), pct(a["ci95_lo"]), pct(a["ci95_hi"]), AZUL, "o")
            punto_ci(ax, y, pct(b["diff"]), pct(b["ci95_lo"]), pct(b["ci95_hi"]), NARANJA, "s")
            filas_csv.append({"par": par, "generador": g, "dtgem": a["diff"], "dtgem_lo": a["ci95_lo"],
                              "dtgem_hi": a["ci95_hi"], "dtgen": b["diff"], "dtgen_lo": b["ci95_lo"], "dtgen_hi": b["ci95_hi"],
                              "fraccion": b["diff"] / a["diff"]})
        ax.set_yticks(range(len(filas)))
        if j == 0:
            ax.set_yticklabels([t["macro_row"] if g == "macro" else NOMBRE_GEN[g] for g in reversed(filas)])
        else:
            ax.set_yticklabels([])
        ax.axhline(0.5, color=GRIS_CLARO, lw=0.6)
        ax.set_xlim(0, 32)
        ax.set_xlabel(t["gan_x"])
        nom = par.replace("infonce_hn", nb("infonce_hn", idioma)).replace("siglip_hn", nb("siglip_hn", idioma))
        nom = nom.replace("− infonce", "− InfoNCE").replace("− siglip", "− SigLIP")
        ax.set_title(nom, loc="left", fontweight="normal", fontsize=7.5)
    fig.text(0.125, 0.985, t["transf_a"], fontweight="bold", fontsize=8, ha="left", va="top")
    leg = [Line2D([], [], marker="o", ls="-", lw=1.0, color=AZUL, ms=4.2, label=t["dtgem"]),
           Line2D([], [], marker="s", ls="-", lw=1.0, color=NARANJA, ms=4.2, label=t["dtgen"])]
    fig.legend(handles=leg, loc="upper left", bbox_to_anchor=(0.12, 0.955), ncol=2)
    # (b) C2b
    ax = fig.add_subplot(gs[1, :])
    pr = d["c2b"]["pruebas"]
    filas_b = []
    for p in pr:
        nom_par = p["par"].replace("F3 · ", "")
        comp = "gen" if p["componente"].startswith("generador") else "con"
        filas_b.append((nom_par, comp, p))
    orden_pares = list(dict.fromkeys(f[0] for f in filas_b))
    yticks, ylabs = [], []
    y = 0
    for par in reversed(orden_pares):
        for comp in ("con", "gen"):
            p = next(f[2] for f in filas_b if f[0] == par and f[1] == comp)
            c = TINTA2 if comp == "gen" else GRIS
            ax.barh(y, pct(p["diff"]), height=0.62, color=c, zorder=2)
            ax.plot([pct(p["ci95"][0]), pct(p["ci95"][1])], [y, y], color=TINTA, lw=1.0, zorder=3)
            ax.text(pct(p["ci95"][1]) + 0.4, y, f"{f_num(pct(p['diff']), 1, True)}", va="center", fontsize=6.3, color=TINTA2)
            nombre = "InfoNCE" if par.startswith("infonce") else "SigLIP"
            ylabs.append(f"{nombre}: {t['comp_gen'] if comp == 'gen' else t['comp_con']}")
            yticks.append(y)
            filas_csv.append({"par": par, "generador": f"C2b:{comp}", "dtgem": "", "dtgem_lo": "", "dtgem_hi": "",
                              "dtgen": "", "dtgen_lo": "", "dtgen_hi": "", "fraccion": "",
                              "componente": p["componente"], "diff": p["diff"], "lo": p["ci95"][0], "hi": p["ci95"][1],
                              "p_holm": p["p_holm"], "brecha_total": p["brecha_total"]})
            y += 1
        ax.text(31.5, y - 1.5, f"{t['brecha']}: {f_num(pct(p['brecha_total']), 1, True)} pp", ha="right", va="center",
                fontsize=6.3, color=TINTA2)
        y += 0.6
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabs)
    ax.set_xlim(0, 32)
    ax.set_xlabel(t["comp_x"])
    ax.set_title(t["transf_b"])
    # CSV con las mismas columnas en todas las filas
    cols = sorted({k for f in filas_csv for k in f})
    S.datos("fig_transferencia", [{k: f.get(k, "") for k in cols} for f in filas_csv], ["sens_panel", "c2", "c2b"])
    fr = [filas_csv[i]["fraccion"] for i in range(len(filas_csv)) if filas_csv[i]["generador"] in MACRO]
    if idioma == "es":
        ley = ("Cuánto de la ganancia por HN de Gemini se transfiere (test sellado, F3; IC95 por bootstrap pareado "
               "por documento). (a) Para cada generador del macro y para el macro congelado: ganancia de entrenar con "
               "HN (Δ top-1) en T-gem sobre los mismos ítems (círculos) y en los negativos de ese generador (cuadrados). "
               "La fila del macro es C2 (pre-especificado); las filas por generador son la sensibilidad post hoc "
               f"(descriptiva). La fracción transferida por generador va de {min(fr):.2f} a {max(fr):.2f}. (b) C2b "
               f"(pre-especificado), con {d['c2b']['n_items']} ítems y 3 generadores chicos del piloto: parte de la "
               "brecha propia del generador (Δ T-gem − Δ S) y parte del tipo de negativo, libre frente a editado "
               "(Δ S − Δ E), con IC95; «InfoNCE» es el par InfoNCE+HN − InfoNCE y «SigLIP», SigLIP+HN − SigLIP.")
    else:
        ley = ("How much of the Gemini HN gain transfers (sealed test, F3; 95% CI from a bootstrap paired by document). "
               "(a) For each macro generator and for the frozen macro: gain from training with HNs (Δ top-1) on T-gem over "
               "the same items (circles) and on that generator's negatives (squares). The macro row is C2 (pre-specified); "
               f"per-generator rows are the post hoc sensitivity (descriptive). The transferred fraction ranges from "
               f"{min(fr):.2f} to {max(fr):.2f}. (b) C2b (pre-specified), {d['c2b']['n_items']} items and 3 small pilot "
               "generators: the part of the gap due to the generator (Δ T-gem − Δ S) and the part due to the negative type, "
               "free vs. edited (Δ S − Δ E), with 95% CIs; 'InfoNCE' is the InfoNCE+HN − InfoNCE pair and 'SigLIP' "
               "the SigLIP+HN − SigLIP pair.")
    S.figura(fig, "fig_transferencia", ["sens_panel", "c2", "c2b"],
             "¿Cuánto de la ganancia por HN se transfiere a negativos de otros generadores, y de dónde viene lo que no?",
             "confirmatorio (C2, C2b); filas por generador post hoc", ley, "Resultados III (planeada como Fig. 4)")


def fig_denominador(d, S, t, idioma):
    c2c, b4, borde = d["c2c"], d["extr"]["b4"]["pares"], d["borde"]
    fig = plt.figure(figsize=(6.3, 4.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[0.8, 1.35], hspace=0.75, wspace=0.08)
    ax = fig.add_subplot(gs[0, :])
    pr = {p["metrica"]: p for p in c2c["pruebas"]}
    sig = next(p for p in b4 if p["par"].startswith("siglip"))["tgem_shuf"]
    filas = [("c2c_shuf", pr["tgem_shuf"], True), ("c2c_tmine", pr["tmine"], True),
             ("c2c_ctl", c2c["control"], True), ("sig_shuf", sig, False)]
    filas_csv = []
    for i, (k, r, pre) in enumerate(filas):
        y = len(filas) - 1 - i
        punto_ci(ax, y, pct(r["diff"]), pct(r["ci95_lo"]), pct(r["ci95_hi"]), TINTA2, "o" if pre else "D", relleno=pre)
        eq = f" ({t['equiv']})" if k == "c2c_ctl" and r.get("tost_equivalent") else ""
        ax.text(pct(r["ci95_hi"]) + 0.5 if r["diff"] >= 0 else pct(r["ci95_lo"]) - 0.5, y,
                f"{f_num(pct(r['diff']), 1, True)}{eq}", va="center", ha="left" if r["diff"] >= 0 else "right",
                fontsize=6.3, color=TINTA2)
        filas_csv.append({"panel": "a", "fila": t[k], "estatus": t["conf"] if pre else t["desc"], "media_o_diff": r["diff"],
                          "lo": r["ci95_lo"], "hi": r["ci95_hi"], "p": r.get("p_holm", r.get("p")), "R": "", "R_lo": "",
                          "R_hi": ""})
    ax.axvline(0, color=TINTA2, lw=0.7)
    ax.set_yticks(range(len(filas)))
    ax.set_yticklabels([t[k] for k, _, _ in reversed(filas)])
    ax.set_xlim(-10, 26)
    ax.set_xlabel(t["delta_pp"])
    ax.set_title(t["den_a"])
    conds = ["a_hn_only", "b_hn_inbatch", "c_hn_otherhn", "c31_hn_otherhn_k31", "d31_hn_inbatch_k31", "e_hn_shared"]
    nom = {"a_hn_only": "cond_a", "b_hn_inbatch": "cond_b", "c_hn_otherhn": "cond_c", "c31_hn_otherhn_k31": "cond_c31",
           "d31_hn_inbatch_k31": "cond_d31", "e_hn_shared": "cond_e"}
    tipo = {"a_hn_only": ("tipo_ref", TINTA2, "o"), "b_hn_inbatch": ("tipo_ref", TINTA2, "o"),
            "c_hn_otherhn": ("tipo_hn", NARANJA, "s"), "c31_hn_otherhn_k31": ("tipo_hn", NARANJA, "s"),
            "d31_hn_inbatch_k31": ("tipo_oro", AZUL, "D"), "e_hn_shared": ("tipo_comp", AQUA, "^")}  # rombo: en gris, los oros ajenos no se confunden con las referencias
    met = borde["metricas"]
    for j, (mk_, tit, xl) in enumerate((("tgem_shuf", "den_b1", "shuf_x"), ("tmine", "den_b2", "tmine_x"))):
        ax = fig.add_subplot(gs[1, j])
        for i, c in enumerate(conds):
            y = len(conds) - 1 - i
            r = met[mk_][c]
            _, col, mk = tipo[c]
            punto_ci(ax, y, pct(r["media"]), pct(r["ic95"][0]), pct(r["ic95"][1]), col, mk)
            if mk_ == "tgem_shuf" and c in borde["R"]:
                R = borde["R"][c]
                ax.text(pct(r["ic95"][1]) + 0.8, y, f"R = {R['R']:.2f}", va="center", fontsize=6.3, color=TINTA2)
            filas_csv.append({"panel": "b" if mk_ == "tgem_shuf" else "c", "fila": t[nom[c]], "estatus": "exploratorio",
                              "media_o_diff": r["media"], "lo": r["ic95"][0], "hi": r["ic95"][1], "p": "",
                              "R": borde["R"].get(c, {}).get("R", "") if mk_ == "tgem_shuf" else "",
                              "R_lo": (borde["R"].get(c, {}).get("ic95") or ["", ""])[0] if mk_ == "tgem_shuf" else "",
                              "R_hi": (borde["R"].get(c, {}).get("ic95") or ["", ""])[1] if mk_ == "tgem_shuf" else ""})
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels([t[nom[c]] for c in reversed(conds)] if j == 0 else [])
        ax.set_xlabel(t[xl])
        if j == 0:
            ax.set_title(t["den_b"], loc="left")
        if mk_ == "tgem_shuf":
            ax.set_xlim(0, 36)
            linea_ref(ax, pct(AZAR), t["azar"])
        else:
            lo = min(pct(met["tmine"][c]["ic95"][0]) for c in conds)
            hi = max(pct(met["tmine"][c]["ic95"][1]) for c in conds)
            ax.set_xlim(np.floor(lo - 1), np.ceil(hi + 1))
    vistos = []
    leg = []
    for c in conds:
        k, col, mk = tipo[c]
        if k not in vistos:
            vistos.append(k)
            leg.append(Line2D([], [], marker=mk, ls="-", lw=1.0, color=col, ms=4.2, label=t[k]))
    fig.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, 0.035), ncol=4)
    S.datos("fig_denominador", filas_csv, ["c2c", "extr", "borde"])
    Pab = pct(borde["P_a_menos_P_b"])
    if idioma == "es":
        ley = ("Qué denominador aprende una preferencia que no mira el estado. (a) Test sellado (F3): C2c compara la "
               "variante \"sólo HN\" con InfoNCE+HN (\"HN + in-batch\"), con los mismos HP y 5 semillas (pre-especificado, "
               "registrado en adenda); el control es InfoNCE+HN − InfoNCE con estados barajados; el rombo es "
               "descriptivo. (b) Condición de borde de B5, exploratoria, en el split de test (InfoNCE+HN con los HP "
               "de F1b, 3 semillas; sólo cambia el denominador estado→acción). R = fracción de la preferencia de (a) "
               f"que conserva cada condición, relativa a (b); P(a) − P(b) = {Pab:.1f} pp. IC95 por bootstrap pareado "
               "por documento.")
    else:
        ley = ("Which denominator learns a preference that ignores the state. (a) Sealed test (F3): C2c compares the "
               "\"HN only\" variant with InfoNCE+HN (\"HN + in-batch\"), same HPs and 5 seeds (pre-specified, registered in "
               "an addendum); the control is InfoNCE+HN − InfoNCE with shuffled states; the diamond is descriptive. "
               "(b) Exploratory B5 boundary condition on the test split (InfoNCE+HN with F1b HPs, 3 seeds; only the "
               "state→action denominator changes). R = fraction of the preference of (a) that each condition keeps, "
               f"relative to (b); P(a) − P(b) = {Pab:.1f} pp. 95% CIs from a bootstrap paired by document.")
    S.figura(fig, "fig_denominador", ["c2c", "extr", "borde"],
             "¿Qué denominador aprende una preferencia sin estado, y por qué los oros de otros estados la quitan?",
             "(a) pre-especificado con desviación declarada (C2c); (b, c) exploratorio", ley,
             "Resultados III (planeada como Fig. 5)")


def fig_falsos_negativos(d, S, t, idioma):
    jvh, cal = d["jvh"], d["cal"]
    fig, ax = plt.subplots(figsize=(6.3, 2.7))
    series = [("estricta", AZUL, "o", 0.22, "fn_estricta"), ("amplia", NARANJA, "s", 0.0, "fn_amplia"),
              ("juez", AQUA, "^", -0.22, "fn_juez")]
    filas_csv = []
    nombres = []
    for i, s_ in enumerate(SETS_FN):
        y = len(SETS_FN) - 1 - i
        for ref, col, mk, dy, _ in series:
            if ref == "juez":
                r = cal["tasas_juez"][s_]
                m, lo, hi, n = r["tasa_item"], r["ic_item"][0], r["ic_item"][1], r["n_items"]
            else:
                r = jvh[ref][s_]
                m, lo, hi, n = r["tasa_item_humano"], r["ic_item_humano"][0], r["ic_item_humano"][1], r["items_completos"]
            punto_ci(ax, y + dy, pct(m), pct(lo), pct(hi), col, mk, ms=3.9)
            filas_csv.append({"set": s_, "referencia": ref, "tasa_pregunta": m, "ic95_lo": lo, "ic95_hi": hi, "n_preguntas": n})
        nh = jvh["estricta"][s_]["items_completos"]
        nj = cal["tasas_juez"][s_]["n_items"]
        etiqueta = {"tgem": "T-gem", "tmine": "T-mine"}.get(s_, f"T-gen · {NOMBRE_GEN.get(s_[5:], s_[5:])}")
        nombres.append(f"{etiqueta} (n = {nh} / {nj:,})")
    ax.set_yticks(range(len(SETS_FN)))
    ax.set_yticklabels(list(reversed(nombres)))
    ax.axhline(len(SETS_FN) - 2.5, color=GRIS_CLARO, lw=0.6)
    ax.set_xlim(-1.5, 100)
    ax.set_xlabel(t["fn_x"])
    linea_ref(ax, 5, t["criterio5"], ":")
    leg = [Line2D([], [], marker=mk, ls="-", lw=1.0, color=col, ms=3.9, label=t[k]) for _, col, mk, _, k in series]
    ax.legend(handles=leg, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    S.datos("fig_falsos_negativos", filas_csv, ["jvh", "cal"])
    kap = jvh["estricta"]["total"]["kappa"], jvh["amplia"]["total"]["kappa"]
    if idioma == "es":
        ley = ("Falsos negativos en el test sellado: porcentaje de preguntas con al menos un negativo que también "
               "responde bien la pregunta según el oro (IC95 de Wilson). Humanos: 200 preguntas (n por set a la "
               "izquierda), cada candidato marcado por dos anotadores por separado; estricta = los dos lo marcaron, "
               "amplia = al menos uno. Juez: OLMo 3.1 32B sobre todas las preguntas del set (segundo número). "
               f"Acuerdo juez–humanos: kappa de Cohen {kap[0]:.2f} (estricta) y {kap[1]:.2f} (amplia); entre "
               f"anotadores, kappa de Fleiss {d['acu']['acuerdo']['kappa_fleiss']:.2f}. "
               "La línea punteada marca el criterio de F0.5 (menos de 5 %).") if "acu" in d else None
    else:
        ley = ("False negatives on the sealed test: percentage of questions with at least one negative that also answers "
               "the question correctly according to the gold (Wilson 95% CI). Humans: 200 questions (n per set on the "
               "left), each candidate labeled by two annotators independently; strict = both marked it, lenient = at "
               "least one. Judge: OLMo 3.1 32B on every question of the set (second number). Judge–human agreement: "
               f"Cohen's kappa {kap[0]:.2f} (strict) and {kap[1]:.2f} (lenient); between annotators, Fleiss' kappa "
               f"{d['acu']['acuerdo']['kappa_fleiss']:.2f}. The dotted line marks the F0.5 criterion (below 5%).") \
            if "acu" in d else None
    S.figura(fig, "fig_falsos_negativos", ["jvh", "cal", "acu"],
             "¿Cuántos negativos de cada conjunto son en realidad respuestas correctas?",
             "descriptivo (B10; resultado descriptivo por decisión del investigador, P22)", ley,
             "Resultados I (nueva; después de la de atajos)")


def _rangos(v):
    """Rangos promedio (empates) para Spearman, sin depender de scipy."""
    v = np.asarray(v, float)
    o = np.argsort(v, kind="mergesort")
    r = np.empty(len(v))
    r[o] = np.arange(1, len(v) + 1)
    for x in np.unique(v):
        m = v == x
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def fig_seleccion(d, S, t, idioma):
    f1, f1b, ihn = d["hpo_f1_siglip"], d["hpo_f1b_siglip"], d["hpo_f1_infonce_hn"]
    a1 = {r["trial"]: r for r in f1["trials"]}
    a2 = {r["trial"]: r for r in f1b["trials"]}
    comunes = sorted(set(a1) & set(a2))
    distintos = [k for k in comunes if a1[k]["overrides"] != a2[k]["overrides"]]
    if distintos:
        raise SystemExit(f"fig_seleccion: los trials {distintos} no tienen los mismos hiperparámetros en F1 y F1b")
    ok = [k for k in comunes if np.isfinite(a1[k]["val_best"]) and np.isfinite(a2[k]["val_best"])]
    b1, b2 = f1["best"]["trial"], f1b["best"]["trial"]
    if b1 not in ok or b2 not in ok:
        raise SystemExit("fig_seleccion: el trial elegido en F1 o en F1b no está en ambas fases")
    x = np.array([a1[k]["val_best"] for k in ok])
    y = np.array([a2[k]["val_best"] for k in ok])
    rho = float(np.corrcoef(_rangos(x), _rangos(y))[0, 1])
    fig = plt.figure(figsize=(6.3, 2.85))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.22, 1.0], wspace=0.34)
    filas_csv = []
    # (a) validación: los mismos trials con las dos métricas ------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    otros = [k for k in ok if k not in (b1, b2)]
    ax.plot([pct(a1[k]["val_best"]) for k in otros], [pct(a2[k]["val_best"]) for k in otros], marker="o", ms=3.6,
            color=GRIS, mfc="white", mew=0.8, ls="none", zorder=2)
    p1 = (pct(a1[b1]["val_best"]), pct(a2[b1]["val_best"]))
    p2 = (pct(a1[b2]["val_best"]), pct(a2[b2]["val_best"]))
    ax.plot(*p1, marker="s", ms=5.4, color=NARANJA, ls="none", zorder=4)
    ax.plot(*p2, marker="o", ms=5.8, color=AZUL, ls="none", zorder=4)
    ax.annotate(t["sel_lab1"].format(k=b1), p1, xytext=(-7, 0), textcoords="offset points", ha="right", va="center",
                fontsize=6.4, color=TINTA)
    ax.annotate(t["sel_lab2"].format(k=b2), p2, xytext=(-2, 6), textcoords="offset points", ha="right", va="bottom",
                fontsize=6.4, color=TINTA)
    # cota: cuánto separa val-mine a los dos trials que T-gem-val casi empata
    xb = max(p1[0], p2[0]) + 1.5
    ax.plot([xb - 0.45, xb, xb, xb - 0.45], [p1[1], p1[1], p2[1], p2[1]], color=TINTA2, lw=0.6, zorder=1,
            solid_joinstyle="miter")
    ax.text(xb + 0.4, (p1[1] + p2[1]) / 2, f"{abs(p2[1] - p1[1]):.0f} pp", rotation=90, ha="left", va="center",
            fontsize=6.3, color=TINTA2)
    ax.text(0.98, 0.03, t["sel_rho"].format(rho=rho, n=len(ok)), transform=ax.transAxes, ha="right", va="bottom",
            fontsize=6.3, color=TINTA2)
    for k in ok:
        filas_csv.append({"panel": "a", "trial": k, "overrides": json.dumps(a1[k]["overrides"], ensure_ascii=False),
                          "tgem_val_F1": a1[k]["val_best"], "valmine_F1b": a2[k]["val_best"],
                          "elegido_F1": k == b1, "elegido_F1b": k == b2})
    ax.set_xlim(10, 36)
    ax.set_xticks(range(10, 36, 5))
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(True, axis="both")
    ax.set_axisbelow(True)
    ax.set_xlabel(t["sel_x"])
    ax.set_ylabel(t["sel_y"])
    ax.set_title(t["sel_a"], fontsize=7.4)
    # (b) test: el modelo final de cada elección -------------------------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    series = (("avs_f1", NARANJA, "s", 5.2, -0.13), ("avs_f1b", AZUL, "o", 5.6, 0.13))
    conjuntos = (("tgem_top1", 0), ("tmine_top1", 1))
    fin = {}
    for k, col, mk, ms, dx in series:
        for met, xc in conjuntos:
            r = {q["arm"]: q for q in d[k]["metrics"][met]["summary"]}["siglip"]
            m, lo, hi = pct(r["mean"]), pct(r["ci95_lo"]), pct(r["ci95_hi"])
            fin[(k, met)] = m
            xv = xc + dx
            ax.plot([xv, xv], [lo, hi], color=col, lw=1.0, solid_capstyle="round", zorder=3)
            ax.plot([xv], [m], marker=mk, ms=ms, color=col, ls="none", zorder=4)
            ax.annotate(f"{m:.1f}", (xv, m), xytext=(-6 if dx < 0 else 6, 0), textcoords="offset points",
                        ha="right" if dx < 0 else "left", va="center", fontsize=6.3, color=TINTA2)
            filas_csv.append({"panel": "b", "trial": "", "overrides": "", "fase": k.replace("avs_", "").upper(),
                              "metrica": met, "media": r["mean"], "ic95_lo": r["ci95_lo"], "ic95_hi": r["ci95_hi"],
                              "semillas": r["n_seeds"]})
    ax.axhline(pct(AZAR), color=GRIS, lw=0.7, ls="--", zorder=1)
    ax.text(1.0, pct(AZAR) + 1.2, f"{t['azar']} ", transform=ax.get_yaxis_transform(), ha="right", va="bottom",
            fontsize=6.3, color=TINTA2)
    ax.set_xlim(-0.6, 1.6)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T-gem", "T-mine"])
    ax.tick_params(axis="x", length=0, labelsize=7.2, labelcolor=TINTA)
    ax.set_ylim(0, 100)
    ax.set_yticks(range(0, 101, 20))
    ax.grid(False, axis="x")
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_ylabel(t["sel_top1"])
    ax.set_title(t["sel_b"], fontsize=7.4)
    leg = [Line2D([], [], marker="s", ls="none", color=NARANJA, ms=5.2, label=t["sel_leg1"]),
           Line2D([], [], marker="o", ls="none", color=AZUL, ms=5.6, label=t["sel_leg2"])]
    ax.legend(handles=leg, loc="upper left", fontsize=6.3, title=t["sel_legt"], title_fontsize=6.3, alignment="left",
              handletextpad=0.3, borderaxespad=0.3, labelspacing=0.45)
    # InfoNCE+HN del F1 piloto: sólo al CSV y a la leyenda (el denominador que eligió T-gem-val) ----------------
    den = {"hn_only": [], "hn_plus_inbatch": []}
    for r in ihn["trials"]:
        if not np.isfinite(r["val_best"]):
            continue
        dd = r["overrides"].get("loss.hn_denominator", "hn_only")  # trial 0 = valores publicados: "sólo HN"
        den.setdefault(dd, []).append((r["trial"], r["val_best"]))
        filas_csv.append({"panel": "c (sólo CSV)", "trial": r["trial"],
                          "overrides": json.dumps(r["overrides"], ensure_ascii=False), "denominador": dd,
                          "tgem_val_F1": r["val_best"], "elegido_F1": r["trial"] == ihn["best"]["trial"]})
    eleg_den = {dd for dd, v in den.items() for tr_, _ in v if tr_ == ihn["best"]["trial"]}
    if eleg_den != {"hn_only"}:
        raise SystemExit(f"fig_seleccion: el trial elegido de InfoNCE+HN no es de 'sólo HN' ({eleg_den}); revisar la leyenda")
    S.datos("fig_seleccion", filas_csv, ["hpo_f1_siglip", "hpo_f1b_siglip", "hpo_f1_infonce_hn", "avs_f1", "avs_f1b"])
    ho = [v for _, v in den["hn_only"]]
    ib = [v for _, v in den["hn_plus_inbatch"]]
    m1, m2 = fin[("avs_f1", "tmine_top1")], fin[("avs_f1b", "tmine_top1")]
    g1, g2 = fin[("avs_f1", "tgem_top1")], fin[("avs_f1b", "tgem_top1")]
    if idioma == "es":
        ley = ("Elegir con un set de validación que tiene atajo premia configuraciones rotas (exploratorio; F1 piloto y F1b, "
               "test exploratorio). (a) SigLIP: los mismos 32 trials (mismos hiperparámetros, 25 % del presupuesto) medidos "
               "con T-gem-val, la métrica de selección del F1 piloto, y con val-mine, la de F1b. T-gem-val da casi lo mismo "
               f"al trial {b1} ({p1[0]:.1f} %), su máximo, y al trial {b2} ({p2[0]:.1f} %), el máximo de val-mine; en "
               f"val-mine los separan {abs(p2[1] - p1[1]):.0f} pp ({p1[1]:.1f} % frente a {p2[1]:.1f} %). Entre los "
               f"{len(ok)} trials, ρ de Spearman = {rho:.2f}. (b) El modelo final entrenado con cada elección (3 semillas; "
               f"barras: IC95 por bootstrap pareado por documento; línea discontinua: azar, 1 de 11): T-mine {m1:.1f} % frente a {m2:.1f} %; tampoco en T-gem gana la "
               f"elección de T-gem-val ({g1:.1f} % frente a {g2:.1f} %). En el F1 piloto de InfoNCE+HN, cuyo espacio de "
               f"búsqueda incluía el denominador, T-gem-val puso a los {len(ho)} trials con \"sólo HN\" "
               f"({pct(min(ho)):.1f}–{pct(max(ho)):.1f} %) por encima de los {len(ib)} con \"HN + in-batch\" "
               f"({pct(min(ib)):.1f}–{pct(max(ib)):.1f} %) y eligió uno de \"sólo HN\", el denominador que aprende la "
               "preferencia sin estado (C2c); esos trials van en el CSV de la figura. Límite: F1 y F1b difieren en la "
               "versión del código y en la métrica de early stopping; en (a) se emparejan trials con hiperparámetros "
               "idénticos, no corridas idénticas.")
    else:
        ley = ("Selecting with a validation set that carries a shortcut rewards broken configurations (exploratory; pilot F1 "
               "and F1b, exploratory test). (a) SigLIP: the same 32 trials (same hyperparameters, 25% budget) scored with "
               "T-gem-val, the selection metric of the pilot F1, and with val-mine, the F1b metric. T-gem-val scores "
               f"trial {b1} ({p1[0]:.1f}%), its maximum, almost the same as trial {b2} ({p2[0]:.1f}%), the val-mine "
               f"maximum; val-mine puts them {abs(p2[1] - p1[1]):.0f} pp apart ({p1[1]:.1f}% vs. {p2[1]:.1f}%). Across "
               f"the {len(ok)} trials, Spearman ρ = {rho:.2f}. (b) The final model trained with each choice (3 seeds; bars: "
               f"95% CI from a bootstrap paired by document; dashed line: chance, 1 in 11): T-mine {m1:.1f}% vs. {m2:.1f}%; the T-gem-val choice does not win "
               f"on T-gem either ({g1:.1f}% vs. {g2:.1f}%). In the pilot F1 of InfoNCE+HN, whose search space included the "
               f"denominator, T-gem-val ranked all {len(ho)} \"HN only\" trials ({pct(min(ho)):.1f}–{pct(max(ho)):.1f}%) "
               f"above all {len(ib)} \"HN + in-batch\" trials ({pct(min(ib)):.1f}–{pct(max(ib)):.1f}%) and picked an "
               "\"HN only\" one, the denominator that learns the state-free preference (C2c); those trials are in the "
               "figure's CSV. Caveat: F1 and F1b differ in code version and early-stopping metric; panel (a) pairs trials "
               "with identical hyperparameters, not identical runs.")
    S.figura(fig, "fig_seleccion", ["hpo_f1_siglip", "hpo_f1b_siglip", "hpo_f1_infonce_hn", "avs_f1", "avs_f1b"],
             "¿Qué pasa al elegir modelos con un set de validación que tiene atajo?", "exploratorio (B7; F1 piloto y F1b)", ley,
             "Resultados III (planeada como Fig. 6, opcional)")


# ---------------------------------------------------------------------------------------------------------
# Tablas
# ---------------------------------------------------------------------------------------------------------
def lect(s, idioma):
    if s is None:
        return "—"
    return LECTURAS_EN.get(s, s) if idioma == "en" else LECTURAS_ES.get(s, s)


def tab_contrastes(d, S, idioma):
    es = idioma == "es"
    c1 = d["avs"]["C1"]["por_metrica"]
    filas, csvf = [], []

    def add(prueba, contraste, r, lectura, lo_k=("ci95_lo", "ci95_hi"), p_k="p_holm"):
        lo = r[lo_k[0]] if lo_k[0] in r else r["ci95"][0]
        hi = r[lo_k[1]] if lo_k[1] in r else r["ci95"][1]
        p = r.get(p_k, r.get("p_holm_c1", r.get("p")))
        filas.append([prueba, contraste, f_num(pct(r["diff"]), 2, True), f_ci(pct(lo), pct(hi), 2, True), f_p(p),
                      lect(lectura, idioma)])
        csvf.append({"prueba": prueba, "contraste": contraste, "diff": r["diff"], "ci95_lo": lo, "ci95_hi": hi, "p_holm": p,
                     "lectura": lectura})

    for met, nom in (("tmine", "T-mine"), ("tgenmacro", "macro T-gen" if not es else "macro de T-gen")):
        r = c1[met]
        add("C1 (H1a)", f"VICReg-x − InfoNCE · {nom}", r, r["verdict"], p_k="p_holm_c1")
    for p in d["c2"]["pares"]:
        par = p["par"].replace("infonce_hn", "InfoNCE+HN").replace("siglip_hn", "SigLIP+HN").replace("− infonce", "− InfoNCE").replace("− siglip", "− SigLIP")
        add("C2 (H1b)", f"{par} · Δ T-gem", p["Δ T-gem"], None)
        add("C2 (H1b)", f"{par} · Δ {'macro T-gen' if not es else 'macro de T-gen'}", p["Δ macro T-gen"], None)
        add("C2 (H1b)", f"{par} · C2 = Δ T-gem − Δ T-gen", p["C2"], p["lectura"])
    for p in d["c2b"]["pruebas"]:
        par = p["par"].replace("F3 · ", "").replace("infonce_hn", "InfoNCE+HN").replace("siglip_hn", "SigLIP+HN").replace("− infonce", "− InfoNCE").replace("− siglip", "− SigLIP")
        comp = p["componente"]
        if not es:
            comp = comp.replace("generador", "generator").replace("construcción", "construction")
        add("C2b (H1b)", f"{par} · {comp}", p, p["lectura"])
    for p in d["c2c"]["pruebas"]:
        nom = {"tgem_shuf": "T-gem barajado" if es else "shuffled T-gem", "tmine": "T-mine"}[p["metrica"]]
        add("C2c (H1b)", f"{'sólo HN − HN + in-batch' if es else 'HN only − HN + in-batch'} · {nom}", p, p["lectura"])
    ctl = d["c2c"]["control"]
    add("C2c (control)", f"InfoNCE+HN − InfoNCE · {'T-gem barajado' if es else 'shuffled T-gem'}", ctl, ctl["lectura"], p_k="p")
    cab = (["Prueba", "Contraste", "Estimación (pp)", "IC95 (pp)", "p (Holm)", "Lectura (regla pre-especificada)"] if es else
           ["Test", "Contrast", "Estimate (pp)", "95% CI (pp)", "p (Holm)", "Reading (pre-specified rule)"])
    notas = (["Test sellado (F3), 5 semillas por brazo; bootstrap jerárquico pareado por documento (B = 4,000). C1: Holm "
              "entre sus dos co-primarias; C2 y C2c: Holm entre sus pruebas; C2b: Holm entre los 4 componentes. El "
              "control de C2c se lee por equivalencia (TOST ±1 pp), sin Holm. C2c se registró en una adenda y se "
              "ejecutó después de ver los 15 brazos (desviación declarada). Valores p por aproximación normal sobre "
              "el bootstrap."] if es else
             ["Sealed test (F3), 5 seeds per arm; hierarchical bootstrap paired by document (B = 4,000). C1: Holm across "
              "its two co-primary metrics; C2 and C2c: Holm across their tests; C2b: Holm across the 4 components. The C2c "
              "control is read by equivalence (TOST ±1 pp), without Holm. C2c was registered in an addendum and run after "
              "seeing the 15 arms (declared deviation). p-values from a normal approximation to the bootstrap."])
    ley = ("Contrastes pre-especificados en el test sellado (F3)." if es else "Pre-specified contrasts on the sealed test (F3).")
    S.tabla("tab_contrastes", cab, filas, csvf, ["avs", "c2", "c2b", "c2c"],
            "¿Qué dicen los contrastes pre-especificados?", "confirmatorio (C2c con desviación declarada)", ley,
            "Resultados II y III (planeada como Tab. 6)", notas)


def tab_panel(d, S, idioma):
    es = idioma == "es"
    pt, pc = d["panel_test"]["generators"], d["panel_conf"]["generators"]
    filas, csvf = [], []
    for g in MACRO:
        a, b = pt[g], pc[g]
        filas.append([NOMBRE_GEN[g], a.get("family", "—"), a.get("tier") or "MoE", f"{a['items']:,}", f"{b['items']:,}",
                      f_num(pct(a["coverage"]), 1), f_num(pct(b["coverage"]), 1), f_num(pct(a["atajo_ciego"]), 2),
                      f_num(pct(b["atajo_ciego"]), 2), b.get("banda", "—")])
        csvf.append({"generador": g, "repo": a.get("repo"), "familia": a.get("family"), "tamano": a.get("tier"),
                     "items_test": a["items"], "items_sellado": b["items"], "cobertura_test": a["coverage"],
                     "cobertura_sellado": b["coverage"], "atajo_test": a["atajo_ciego"], "atajo_sellado": b["atajo_ciego"],
                     "banda_test": a.get("banda"), "banda_sellado": b.get("banda")})
    cab = (["Generador", "Familia", "Tamaño", "Ítems (test)", "Ítems (sellado)", "Cobertura test (%)",
            "Cobertura sellado (%)", "Atajo test (%)", "Atajo sellado (%)", "Banda (sellado)"] if es else
           ["Generator", "Family", "Size", "Items (test)", "Items (sealed)", "Coverage test (%)", "Coverage sealed (%)",
            "Shortcut test (%)", "Shortcut sealed (%)", "Band (sealed)"])
    notas = ([f"Macro congelado: versión {d['panel_conf']['version']} ({d['panel_conf']['n_macro_items']:,} ítems comunes "
              f"en el sellado; {d['panel_test']['n_macro_items']:,} en test). Atajo = máximo de las sondas que no ven el "
              "estado; la regla del macro pide < 30 %. Phi-4 cruzó el umbral en el sellado; el macro congelado se mantuvo."]
             if es else
             [f"Frozen macro: version {d['panel_conf']['version']} ({d['panel_conf']['n_macro_items']:,} common items on the "
              f"sealed split; {d['panel_test']['n_macro_items']:,} on test). Shortcut = max over state-free probes; the macro "
              "rule requires < 30%. Phi-4 crossed the threshold on the sealed split; the frozen macro was kept."])
    S.tabla("tab_panel_macro", cab, filas, csvf, ["panel_test", "panel_conf"], "¿Qué generadores forman el macro de T-gen y cuánto atajo tienen?",
            "descriptivo", "Generadores del macro de T-gen y su auditoría." if es else "T-gen macro generators and their audit.",
            "Resultados I (planeada como Tab. 7)", notas)
    filas, csvf = [], []
    for g, a in sorted(pt.items(), key=lambda kv: (kv[1].get("role") != "primario", -(kv[1].get("atajo_ciego") or 0))):
        filas.append([g, a.get("repo", "—"), a.get("family", "—"), a.get("role", "—"), a.get("tier") or "—",
                      f"{a.get('items', 0):,}", f_num(pct(a["coverage"]), 1) if a.get("coverage") is not None else "—",
                      f_num(pct(a["atajo_ciego"]), 2) if a.get("atajo_ciego") is not None else "—", a.get("banda", "—"),
                      ("sí" if es else "yes") if a.get("in_macro") else ""])
        csvf.append({"clave": g, "repo": a.get("repo"), "revision": a.get("revision"), "familia": a.get("family"),
                     "rol": a.get("role"), "tamano": a.get("tier"), "items": a.get("items"), "cobertura": a.get("coverage"),
                     "atajo": a.get("atajo_ciego"), "banda": a.get("banda"), "en_macro": a.get("in_macro")})
    cab = (["Clave", "Modelo", "Familia", "Rol", "Tamaño", "Ítems", "Cobertura (%)", "Atajo (%)", "Banda", "Macro"] if es else
           ["Key", "Model", "Family", "Role", "Size", "Items", "Coverage (%)", "Shortcut (%)", "Band", "Macro"])
    S.tabla("tab_panel_23", cab, filas, csvf, ["panel_test"], "¿Cómo quedó cada uno de los 23 generadores del panel?",
            "descriptivo", "Panel T-gen (construcción E, split de test): los 23 generadores." if es else
            "T-gen panel (construction E, test split): all 23 generators.", "Apéndice B")


def tab_referencias(d, S, idioma):
    es = idioma == "es"
    tb, ctx = d["refs"]["tablas"], d["refs"]["contexto_F3"]
    cols = ["tgem", "tgem_shuf", "trand", "tmine", "tmine50", "tgenmacro", "tgenmacro_shuf"]
    nom = {"qwen3_emb": ("Qwen3-Embedding-8B, zero-shot (pregunta)", "Qwen3-Embedding-8B, zero-shot (question)"),
           "bm25": ("BM25 (pregunta)", "BM25 (question)"),
           "coseno_crudo": ("Coseno crudo de Qwen3-8B (estado de CLM)", "Raw Qwen3-8B cosine (CLM state)")}
    filas, csvf = [], []
    for split in ("confirm", "test"):
        for k in ("qwen3_emb", "bm25", "coseno_crudo"):
            r = tb[split][k]
            filas.append([nom[k][0 if es else 1], ("sellado" if es else "sealed") if split == "confirm" else "test"]
                         + [f_num(pct(r[c]["media"]), 1) for c in cols])
            for c in cols:
                csvf.append({"referencia": k, "split": split, "set": c, "media": r[c]["media"], "ic95_lo": r[c]["ic95"][0],
                             "ic95_hi": r[c]["ic95"][1], "n": r[c].get("n")})
        if split == "confirm":
            filas.append([("InfoNCE entrenado (F3; estado de CLM)" if es else "Trained InfoNCE (F3; CLM state)"),
                          "sellado" if es else "sealed"] + [f_num(pct(ctx[c]), 1) if c in ctx else "—" for c in cols])
            for c in cols:
                csvf.append({"referencia": "infonce_F3", "split": "confirm", "set": c, "media": ctx.get(c, ""), "ic95_lo": "",
                             "ic95_hi": "", "n": ""})
    cab = (["Referencia", "Split", "T-gem", "T-gem barajado", "T-rand", "T-mine", "T-mine-50", "Macro T-gen",
            "Macro barajado"] if es else
           ["Reference", "Split", "T-gem", "Shuffled T-gem", "T-rand", "T-mine", "T-mine-50", "T-gen macro", "Shuffled macro"])
    notas = (["Top-1 (%); azar 9.1. IC95 en el CSV. No es una comparación pareja: Qwen3-Embedding-8B y BM25 ven sólo la "
              "pregunta, los cabezales y el coseno crudo ven el estado de CLM (contexto + pregunta), y los negativos de "
              "T-mine se minaron con los embeddings crudos de Qwen3-8B."] if es else
             ["Top-1 (%); chance 9.1. 95% CIs in the CSV. Not a like-for-like comparison: Qwen3-Embedding-8B and BM25 see "
              "only the question, the heads and the raw cosine see the CLM state (context + question), and T-mine "
              "negatives were mined with raw Qwen3-8B embeddings."])
    S.tabla("tab_referencias", cab, filas, csvf, ["refs"], "¿Dónde quedan los cabezales frente a referencias sin entrenar?",
            "descriptivo", "Referencias sin entrenar." if es else "Untrained references.", "Resultados II (planeada como Tab. 8)", notas)


def tab_calidad(d, S, idioma):
    es = idioma == "es"
    jvh, cal = d["jvh"], d["cal"]
    filas, csvf = [], []
    for s_ in SETS_FN:
        J = cal["tasas_juez"][s_]
        E, A = jvh["estricta"][s_], jvh["amplia"][s_]
        U = jvh.get("unanimes_mas_una_sola", {}).get(s_, {})
        etiqueta = {"tgem": "T-gem", "tmine": "T-mine"}.get(s_, f"T-gen · {NOMBRE_GEN.get(s_[5:], s_[5:])}")
        filas.append([etiqueta, f"{f_num(pct(J['tasa_item']), 1)} {f_ci(pct(J['ic_item'][0]), pct(J['ic_item'][1]), 1)}",
                      f_num(pct(J["tasa_negativo"]), 2),
                      f"{f_num(pct(E['tasa_item_humano']), 1)} {f_ci(pct(E['ic_item_humano'][0]), pct(E['ic_item_humano'][1]), 1)}",
                      f"{f_num(pct(A['tasa_item_humano']), 1)} {f_ci(pct(A['ic_item_humano'][0]), pct(A['ic_item_humano'][1]), 1)}",
                      (f"{f_num(E['kappa'], 2)} ({f_num(E['sensibilidad'], 2)} / {f_num(E['especificidad'], 2)})"
                       if E["sensibilidad"] is not None else
                       f"— ({'sin 1 humanos' if es else 'no human 1s'}; {'esp' if es else 'spec'} {f_num(E['especificidad'], 2)})"),
                      f"{f_num(A['kappa'], 2)} ({f_num(A['sensibilidad'], 2)} / {f_num(A['especificidad'], 2)})",
                      f"{f_num(U.get('kappa'), 2)}" if U else "—"])
        csvf.append({"set": s_, "juez_tasa_item": J["tasa_item"], "juez_ic_lo": J["ic_item"][0], "juez_ic_hi": J["ic_item"][1],
                     "juez_n_items": J["n_items"], "juez_tasa_negativo": J["tasa_negativo"],
                     "estricta_tasa_item": E["tasa_item_humano"], "estricta_ic_lo": E["ic_item_humano"][0],
                     "estricta_ic_hi": E["ic_item_humano"][1], "humanos_n_items": E["items_completos"],
                     "amplia_tasa_item": A["tasa_item_humano"], "amplia_ic_lo": A["ic_item_humano"][0],
                     "amplia_ic_hi": A["ic_item_humano"][1], "kappa_estricta": E["kappa"], "sens_estricta": E["sensibilidad"],
                     "esp_estricta": E["especificidad"], "kappa_amplia": A["kappa"], "sens_amplia": A["sensibilidad"],
                     "esp_amplia": A["especificidad"], "kappa_sin_desacuerdo": U.get("kappa"), "n_sin_desacuerdo": U.get("n")})
    Et, At, Ut = jvh["estricta"]["total"], jvh["amplia"]["total"], jvh.get("unanimes_mas_una_sola", {}).get("total", {})
    filas.append(["Total" if es else "Overall", "—", "—", "—", "—",
                  f"**{f_num(Et['kappa'], 2)}** ({f_num(Et['sensibilidad'], 2)} / {f_num(Et['especificidad'], 2)})",
                  f"**{f_num(At['kappa'], 2)}** ({f_num(At['sensibilidad'], 2)} / {f_num(At['especificidad'], 2)})",
                  f"**{f_num(Ut.get('kappa'), 2)}**" if Ut else "—"])
    cab = (["Set", "Juez: % preguntas [IC95]", "Juez: % negativos", "Humanos, estricta [IC95]", "Humanos, amplia [IC95]",
            "κ juez–estricta (sens / esp)", "κ juez–amplia (sens / esp)", "κ sin desacuerdo"] if es else
           ["Set", "Judge: % questions [95% CI]", "Judge: % negatives", "Humans, strict [95% CI]", "Humans, lenient [95% CI]",
            "κ judge–strict (sens / spec)", "κ judge–lenient (sens / spec)", "κ without disagreement"])
    acu = d.get("acu", {}).get("acuerdo", {})
    notas = ([f"Juez: OLMo 3.1 32B (bf16, voraz, prompt fn-judge-v1) sobre todas las preguntas del sellado "
              f"({cal['juez']['n_etiquetados']:,} de {cal['juez']['n_items']:,} con etiqueta válida). Humanos: 30 preguntas "
              "de T-gem, 30 de T-mine y 28 por generador, cada candidato con dos etiquetas independientes; estricta = los "
              "dos marcaron 1, amplia = al menos uno. Kappa de Cohen juez–humanos sobre los 2,000 candidatos; \"sin "
              f"desacuerdo\" = candidatos donde los anotadores no discrepan ({Ut.get('n', '—'):,}). Entre anotadores: acuerdo "
              f"{pct(acu.get('pct_acuerdo', float('nan'))):.1f} %, kappa de Fleiss {acu.get('kappa_fleiss', float('nan')):.2f}. "
              "IC95 de Wilson."] if es else
             [f"Judge: OLMo 3.1 32B (bf16, greedy, prompt fn-judge-v1) on every sealed question "
              f"({cal['juez']['n_etiquetados']:,} of {cal['juez']['n_items']:,} with a valid label). Humans: 30 T-gem, 30 "
              "T-mine and 28 questions per generator, each candidate labeled twice independently; strict = both marked 1, "
              "lenient = at least one. Judge–human Cohen's kappa over the 2,000 candidates; \"without disagreement\" = "
              f"candidates where the annotators do not disagree ({Ut.get('n', '—'):,}). Between annotators: "
              f"{pct(acu.get('pct_acuerdo', float('nan'))):.1f}% agreement, Fleiss' kappa {acu.get('kappa_fleiss', float('nan')):.2f}. "
              "Wilson 95% CIs."])
    S.tabla("tab_falsos_negativos", cab, filas, csvf, ["jvh", "cal", "acu"],
            "¿Cuántos falsos negativos hay por set, según el juez y según las personas?",
            "descriptivo (B10, resultado descriptivo por decisión P22)",
            "Falsos negativos en el test sellado." if es else "False negatives on the sealed test.",
            "Resultados I (planeada como Tab. 9a)", notas)
    filas, csvf = [], []
    for sp in ("val", "test", "confirm"):
        r = cal["fuga"][sp]
        nom = {"val": "val", "test": "test", "confirm": "sellado" if es else "sealed"}[sp]
        filas.append([nom, f"{r['n_items']:,}", f"{r['par_exacto']:,}", f"{r['par_exacto_o_casi']:,}",
                      f"{f_num(pct(r['tasa']), 2)} {f_ci(pct(r['ic95'][0]), pct(r['ic95'][1]), 2)}",
                      f_num(pct(r["tasa_pregunta_igual"]), 1)])
        csvf.append({k: v for k, v in r.items() if k != "ejemplos"} | {"split": sp})
    cab = (["Split", "Ítems", "Par exacto", "Exacto o casi", "Tasa (%) [IC95]", "Misma pregunta, cualquier oro (%)"] if es else
           ["Split", "Items", "Exact pair", "Exact or near", "Rate (%) [95% CI]", "Same question, any gold (%)"])
    notas = (["Fuga = el par (pregunta, oro) del ítem aparece entre los pares de entrenamiento (pre y mid): mismo oro "
              "normalizado y pregunta igual o con Jaccard ≥ 0.8 sobre 3-gramas de palabras. El criterio de F0.4 pedía "
              "menos de 0.5 % en test. La última columna es descriptiva."] if es else
             ["Leakage = the item's (question, gold) pair appears among the training pairs (pre and mid): same normalized "
              "gold and an identical question or one with Jaccard ≥ 0.8 over word 3-grams. The F0.4 criterion asked for "
              "less than 0.5% on test. The last column is descriptive."])
    S.tabla("tab_fuga", cab, filas, csvf, ["cal"], "¿Cuántos ítems de evaluación repiten pares de entrenamiento?", "descriptivo",
            "Fuga entre splits." if es else "Leakage across splits.", "Resultados I (planeada como Tab. 9b)", notas)


def tab_sensibilidad(d, S, idioma):
    es = idioma == "es"
    sens = d["cal"]["sensibilidad"]
    esc = list(sens.keys())
    nom_esc = {"sin exclusiones (control)": ("Sin exclusiones", "No exclusions"),
               "sin ítems con falsos negativos (juez)": ("Sin falsos negativos (juez)", "No false negatives (judge)"),
               "sin ítems con fuga": ("Sin fuga", "No leakage"),
               "sin falsos negativos ni fuga": ("Sin ninguno de los dos", "Neither")}
    filas_def = [("C1 · T-mine", lambda s: s["C1"]["tmine"]), ("C1 · macro T-gen", lambda s: s["C1"]["tgenmacro"]),
                 ("C2 · InfoNCE+HN − InfoNCE", lambda s: next(p for p in s["C2"] if p["par"].startswith("infonce"))["C2"]),
                 ("C2 · SigLIP+HN − SigLIP", lambda s: next(p for p in s["C2"] if p["par"].startswith("siglip"))["C2"]),
                 ("C2c · T-gem barajado" if es else "C2c · shuffled T-gem",
                  lambda s: next(p for p in s["C2c"] if p["metrica"] == "tgem_shuf")["c"]),
                 ("C2c · T-mine", lambda s: next(p for p in s["C2c"] if p["metrica"] == "tmine")["c"]),
                 ("C2c · control", lambda s: s["C2c_control"])]
    filas, csvf = [], []
    for nom, fn in filas_def:
        fila = [nom]
        for e in esc:
            r = fn(sens[e])
            fila.append(f"{f_num(pct(r['diff']), 2, True)} {f_ci(pct(r['ci95_lo']), pct(r['ci95_hi']), 2, True)}")
            csvf.append({"contraste": nom, "escenario": e, "diff": r["diff"], "ci95_lo": r["ci95_lo"], "ci95_hi": r["ci95_hi"],
                         "p": r.get("p_holm", r.get("p")), "n_items": r.get("n_items", "")})
        filas.append(fila)
    ex = sens["sin ítems con falsos negativos (juez)"]["exclusiones"]
    cab = ["Contraste" if es else "Contrast"] + [nom_esc.get(e, (e, e))[0 if es else 1] for e in esc]
    notas = ([f"pp [IC95]; descriptivo. Sin exclusiones, el cálculo reproduce exactamente lo confirmatorio. La exclusión "
              f"por juez saca {ex['tgem']:,} ítems de T-gem, {ex['tmine']:,} de T-mine y, en el macro, el generador en ese "
              f"ítem ({', '.join(f'{NOMBRE_GEN[g]} {n:,}' for g, n in ex['tgen'].items())})."] if es else
             [f"pp [95% CI]; descriptive. Without exclusions the computation reproduces the confirmatory results exactly. "
              f"The judge exclusion removes {ex['tgem']:,} T-gem items, {ex['tmine']:,} T-mine items and, in the macro, that "
              f"generator on that item ({', '.join(f'{NOMBRE_GEN[g]} {n:,}' for g, n in ex['tgen'].items())})."])
    S.tabla("tab_sensibilidad", cab, filas, csvf, ["cal"], "¿Cambian los contrastes sin los ítems con falsos negativos o con fuga?",
            "descriptivo", "Sensibilidad de C1, C2 y C2c a falsos negativos y fuga." if es else
            "Sensitivity of C1, C2 and C2c to false negatives and leakage.", "Resultados I–III (planeada como Tab. 10)", notas)


def tab_friedman_y_secundarias(d, S, idioma):
    es = idioma == "es"
    fr = d["extr"]["friedman"]
    orden = sorted(BRAZOS, key=lambda a: fr["rango_7"][a])
    filas = [[nb(a, idioma), f_num(fr["rango_8"][a], 2), f_num(fr["rango_7"][a], 2)] for a in orden]
    csvf = [{"brazo": a, "rango_8_bloques": fr["rango_8"][a], "rango_7_bloques": fr["rango_7"][a]} for a in orden]
    cab = (["Brazo", "Rango medio, 8 bloques (pre-especificado)", "Rango medio, 7 bloques (sin T-gem barajado)"] if es else
           ["Arm", "Mean rank, 8 blocks (pre-specified)", "Mean rank, 7 blocks (without shuffled T-gem)"])
    notas = ([f"Friedman, 8 bloques: p = {f_p(fr['friedman_p_8'])}, diferencia crítica {fr['cd_8']:.2f}; 7 bloques: p = "
              f"{f_p(fr['friedman_p_7'])}, diferencia crítica {fr['cd_7']:.2f}. En el bloque de T-gem barajado, más acierto "
              "significa más preferencia sin estado; por eso se reporta también sin él (sensibilidad declarada)."] if es else
             [f"Friedman, 8 blocks: p = {f_p(fr['friedman_p_8'])}, critical difference {fr['cd_8']:.2f}; 7 blocks: p = "
              f"{f_p(fr['friedman_p_7'])}, critical difference {fr['cd_7']:.2f}. In the shuffled T-gem block, higher accuracy "
              "means more state-free preference; hence the ranking is also reported without it (declared sensitivity)."])
    S.tabla("tab_friedman", cab, filas, csvf, ["extr"], "¿Cambia el ranking global sin el bloque que premia el atajo?",
            "pre-especificado (8 bloques) y sensibilidad declarada (7)", "Ranking de Friedman de F3." if es else
            "F3 Friedman ranking.", "Apéndice C", notas)
    filas, csvf = [], []
    for r in d["extr"]["puntuaciones_secundarias"]:
        dm = r["alt_menos_cos"]
        filas.append([nb(r["brazo"], idioma), f"{f_num(pct(r['alt']), 1)} {f_ci(pct(r['alt_ci'][0]), pct(r['alt_ci'][1]), 1)}",
                      f"{f_num(pct(r['cos']), 1)} {f_ci(pct(r['cos_ci'][0]), pct(r['cos_ci'][1]), 1)}",
                      f"{f_num(pct(dm['diff']), 1, True)} {f_ci(pct(dm['ci95_lo']), pct(dm['ci95_hi']), 1, True)}", f_p(dm["p"])])
        csvf.append({"brazo": r["brazo"], "secundaria": r["alt"], "secundaria_lo": r["alt_ci"][0], "secundaria_hi": r["alt_ci"][1],
                     "coseno": r["cos"], "coseno_lo": r["cos_ci"][0], "coseno_hi": r["cos_ci"][1], "diff": dm["diff"],
                     "diff_lo": dm["ci95_lo"], "diff_hi": dm["ci95_hi"], "p": dm["p"], "n_corridas": r["n_corridas"]})
    cab = (["Brazo", "Puntuación secundaria (%)", "Coseno (%)", "Diferencia (pp)", "p"] if es else
           ["Arm", "Secondary score (%)", "Cosine (%)", "Difference (pp)", "p"])
    notas = (["Top-1 en T-gem del sellado, 5 semillas; IC95 por bootstrap pareado por documento. Puntuaciones "
              "pre-declaradas: el predictor en SimSiam-x y BYOL-x; los prototipos en DINO-x y SwAV-x. Descriptivo."] if es else
             ["Top-1 on the sealed T-gem, 5 seeds; 95% CIs from a bootstrap paired by document. Pre-declared scores: the "
              "predictor for SimSiam-x and BYOL-x; the prototypes for DINO-x and SwAV-x. Descriptive."])
    S.tabla("tab_secundarias", cab, filas, csvf, ["extr"], "¿Cambian las puntuaciones secundarias la lectura de los brazos que colapsan?",
            "descriptivo", "Puntuaciones secundarias pre-declaradas." if es else "Pre-declared secondary scores.", "Apéndice C", notas)


def tab_acuerdo(d, S, idioma):
    if "acu" not in d:
        return
    es = idioma == "es"
    acu = d["acu"]
    ids = {f"alumno{i}": f"A{i}" for i in range(1, 7)}  # anonimizado en la tabla (A1–A6)
    filas, csvf = [], []
    for par, r in acu["por_par"].items():
        a, b = par.split("–")
        solo = [v for k, v in r.items() if k.startswith("solo_1_")]
        filas.append([f"{ids[a]}–{ids[b]}", f"{r['n']:,}", f_num(pct(r["pct_acuerdo"]), 1), f_num(r["kappa_cohen"], 3),
                      f"{solo[0]} / {solo[1]}"])
        csvf.append({"par": f"{ids[a]}-{ids[b]}", "n": r["n"], "pct_acuerdo": r["pct_acuerdo"], "kappa_cohen": r["kappa_cohen"],
                     "solo_1_primero": solo[0], "solo_1_segundo": solo[1]})
    g = acu["acuerdo"]
    filas.append(["**Total**" if es else "**Overall**", f"{g['candidatos_con_2_etiquetas']:,}", f_num(pct(g["pct_acuerdo"]), 1),
                  f"Fleiss {f_num(g['kappa_fleiss'], 3)}", f"{g['desacuerdos']} {'desacuerdos' if es else 'disagreements'}"])
    csvf.append({"par": "total", "n": g["candidatos_con_2_etiquetas"], "pct_acuerdo": g["pct_acuerdo"],
                 "kappa_cohen": g["kappa_fleiss"], "solo_1_primero": g["desacuerdos"], "solo_1_segundo": ""})
    cab = (["Par de anotadores", "Candidatos en común", "Acuerdo (%)", "Kappa", "Sólo uno marcó 1 (primero / segundo)"] if es else
           ["Annotator pair", "Shared candidates", "Agreement (%)", "Kappa", "Only one marked 1 (first / second)"])
    notas = ([f"Diseño en anillo: 6 tramos de 33–34 preguntas; cada anotador marca dos tramos contiguos. Ambos 1: "
              f"{g['ambos_1']}; ambos 0: {g['ambos_0']:,}; prevalencia de 1: {pct(g['prevalencia_1']):.1f} %. Por pregunta "
              f"(¿tiene al menos un falso negativo?): acuerdo {pct(g['pregunta_pct_acuerdo']):.1f} %, kappa de Cohen "
              f"{g['pregunta_kappa_cohen']:.2f}."] if es else
             [f"Ring design: 6 blocks of 33–34 questions; each annotator labels two contiguous blocks. Both 1: {g['ambos_1']}; "
              f"both 0: {g['ambos_0']:,}; prevalence of 1: {pct(g['prevalencia_1']):.1f}%. Per question (does it have at least "
              f"one false negative?): {pct(g['pregunta_pct_acuerdo']):.1f}% agreement, Cohen's kappa {g['pregunta_kappa_cohen']:.2f}."])
    S.tabla("tab_acuerdo", cab, filas, csvf, ["acu"], "¿Cuánto coinciden los anotadores entre sí?", "descriptivo",
            "Acuerdo entre anotadores (etiquetas a mano)." if es else "Inter-annotator agreement (hand labels).", "Apéndice J", notas)


# ---------------------------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------------------------
# Descripciones para \Description (plantilla de JAIR, basada en acmart). Sólo en inglés, desde los CSV de cada
# figura. Reglas (plantilla y guía de ACM "Describing figures"): texto plano sin LaTeX ni listas, < 2000
# caracteres, primera frase (descripción corta) < 125, de lo general a lo particular, sin repetir la leyenda,
# abreviaturas y símbolos escritos completos ("percent", "percentage points", "minus").
# ---------------------------------------------------------------------------------------------------------
MACROS_DESC = {"fig_atajos": "DescAtajos", "fig_falsos_negativos": "DescFalsosNegativos", "fig_brazos": "DescBrazos",
               "fig_transferencia": "DescTransferencia", "fig_denominador": "DescDenominador",
               "fig_seleccion": "DescSeleccion"}


def _leer_csv(raiz, nombre):
    with open(os.path.join(raiz, "datos_figuras", f"{nombre}.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _u(x):
    return f"{x:.1f}"


def _n(k):
    """Cuentas chicas con palabras, como en el texto del paper."""
    return ("zero one two three four five six seven eight nine ten eleven twelve".split()[k] if 0 <= k <= 12 else str(k))


def _rango(v, unidad="percent"):
    return f"{_u(min(v))} to {_u(max(v))} {unidad}".rstrip()


def _desc_atajos(raiz, t):
    A = _leer_csv(raiz, "fig_atajos")
    val = lambda k: [float(r["acierto_pct"]) for r in A if r["fila"] == t[k]]
    S_, P_, E_ = val("S_row"), val("P_row"), val("E_row")
    pt = [r for r in A if r["fila"] == t["panel_test_row"]]
    pc = val("panel_conf_row")
    mac = [float(r["acierto_pct"]) for r in pt if r["en_macro"] == "True"]
    ctl = [float(r["acierto_pct"]) for r in pt if r["rol"].startswith("control")]
    todos = [float(r["acierto_pct"]) for r in pt]
    if not (len(S_) == len(P_) == len(E_) == 3 and len(pt) == 23 and len(mac) == 5 and len(pc) == 5 and ctl):
        raise SystemExit("descripción fig_atajos: el CSV no tiene la forma esperada")
    corto = "Dot plot of accuracy without seeing the state, by evaluation set; only some constructions sit far above chance."
    largo = ("Horizontal dot plot with eight rows. The horizontal axis is top-1 accuracy obtained without seeing the state, "
             "from 0 to 100 percent, with vertical lines at chance and at the 30 percent threshold of the macro. "
             "The first three rows are T-gem, T-mine and T-rand. T-gem has two points far to the right of chance, the blind "
             "model (circle) and the longest-answer rule (diamond), while T-mine and T-rand each have one point close to "
             "chance. The next three rows are the pilot constructions, three generators each, with a vertical tick "
             f"at the median: S spans {_rango(S_)}, P {_rango(P_)} and E {_rango(E_)}. The seventh row is the panel of 23 "
             f"generators on the test split, spread from {_rango(todos)}; its five macro generators (filled circles) lie "
             f"between {_u(min(mac))} and {_u(max(mac))} percent and its {_n(len(ctl))} control generators (squares) between "
             f"{_u(min(ctl))} and {_u(max(ctl))} percent. The last row shows the five macro generators on the sealed split, "
             f"between {_u(min(pc))} and {_u(max(pc))} percent. Appendix B lists the value for each generator.")
    return corto, largo


def _desc_brazos(raiz, t):
    B = {r["brazo"]: r for r in _leer_csv(raiz, "fig_brazos")}
    tm = {a: 100 * float(r["tmine_media"]) for a, r in B.items()}
    tg = {a: 100 * float(r["tgenmacro_media"]) for a, r in B.items()}
    ancho = [100 * (float(r["tgenmacro_hi"]) - float(r["tgenmacro_lo"])) for r in B.values()]
    col = ["simsiam_x", "byol_x", "dino_x"]
    resto = [a for a in B if a not in col]
    top7 = sorted(B, key=lambda a: -tm[a])[:7]
    hn = [a for a in B if GRUPO[a] == "hn"]
    ib = [a for a in B if GRUPO[a] == "inbatch"]
    sn = [a for a in B if GRUPO[a] == "sinneg" and a not in col]
    if len(B) != 15 or max(tm[a] for a in col) >= min(tm[a] for a in resto):
        raise SystemExit("descripción fig_brazos: los tres brazos colapsados ya no son los más bajos en T-mine")
    orden_hn_arriba = min(tg[a] for a in hn) > max(tg[a] for a in B if a not in hn)
    corto = "Two dot plots of top-1 accuracy for the 15 arms: high and tightly grouped on T-mine, low and spread on the T-gen macro."
    largo = ("Two horizontal dot plots side by side share the same 15 rows, one per arm, sorted from highest to lowest "
             "accuracy on T-mine. Marker shape shows which negatives the loss uses: squares for hard negatives, circles for "
             "in-batch negatives only, and triangles for losses without per-instance negatives. In the left panel, T-mine, on "
             f"an axis from 0 to 100 percent, {_n(len(resto))} arms lie between {_u(min(tm[a] for a in resto))} and "
             f"{_u(max(tm[a] for a in resto))} percent, the seven highest between {_u(min(tm[a] for a in top7))} and "
             f"{_u(max(tm[a] for a in top7))} percent, while SimSiam-x, BYOL-x and DINO-x sit close to the chance line, "
             f"between {_u(min(tm[a] for a in col))} and {_u(max(tm[a] for a in col))} percent. In the right panel, the T-gen "
             f"macro, every arm is much lower, from {_u(min(tg.values()))} to {_u(max(tg.values()))} percent. "
             + (f"The three hard-negative arms are the highest ({_u(min(tg[a] for a in hn))} to {_u(max(tg[a] for a in hn))} "
                "percent), " if orden_hn_arriba else
                f"The three hard-negative arms reach {_u(min(tg[a] for a in hn))} to {_u(max(tg[a] for a in hn))} percent, ")
             + f"the in-batch arms lie between {_u(min(tg[a] for a in ib))} and {_u(max(tg[a] for a in ib))} percent, the "
             f"other four arms without per-instance negatives between {_u(min(tg[a] for a in sn))} and "
             f"{_u(max(tg[a] for a in sn))} percent, and the three collapsed arms at chance level ("
             f"{_u(min(tg[a] for a in col))} to {_u(max(tg[a] for a in col))} percent). On the right panel the 95 percent "
             f"confidence intervals are {_u(min(ancho))} to {_u(max(ancho))} percentage points wide.")
    return corto, largo


def _desc_falsos(raiz, t):
    F = _leer_csv(raiz, "fig_falsos_negativos")
    q = {(r["set"], r["referencia"]): (100 * float(r["tasa_pregunta"]), 100 * float(r["ic95_lo"]),
                                       100 * float(r["ic95_hi"]), int(r["n_preguntas"])) for r in F}
    nom = lambda s: {"tgem": "T-gem", "tmine": "T-mine"}.get(s, NOMBRE_GEN.get(s[5:], s[5:]))
    refs = (("estricta", "strict"), ("amplia", "lenient"), ("juez", "judge"))
    hum = [q[(s, r)] for s in SETS_FN for r in ("estricta", "amplia")]
    jue = [q[(s, "juez")] for s in SETS_FN]
    n_h = sorted({x[3] for x in hum})
    mins = {r: min(SETS_FN, key=lambda s: q[(s, r)][0]) for r, _ in refs}
    maxs = {r: max(SETS_FN, key=lambda s: q[(s, r)][0]) for r, _ in refs}
    entre = [s for s in SETS_FN if min(q[(s, "estricta")][0], q[(s, "amplia")][0]) <= q[(s, "juez")][0]
             <= max(q[(s, "estricta")][0], q[(s, "amplia")][0])]
    fuera = [s for s in SETS_FN if s not in entre]
    bajo5 = [s for s in SETS_FN if q[(s, "juez")][0] < 5]
    gens = [s for s in SETS_FN if s.startswith("tgen_")]
    # quién tiene el máximo con cada referencia, dicho en una frase
    por_set = {}
    for r, en in refs:
        por_set.setdefault(maxs[r], []).append((en, q[(maxs[r], r)][0]))
    altos = "; ".join(f"{nom(s)} under " + " and ".join(f"the {en} reference ({_u(v)} percent)" if en != "judge"
                                                        else f"the judge ({_u(v)} percent)" for en, v in lst)
                      for s, lst in por_set.items())
    vm = [_u(q[("tmine", r)][0]) for r, _ in refs]
    bajos = (f"T-mine has the lowest value on all three measures ({vm[0]}, {vm[1]} and {vm[2]} percent)"
             if all(m == "tmine" for m in mins.values())
             else "The lowest values are " + "; ".join(
                 f"{nom(mins[r])} under the {en} reference ({_u(q[(mins[r], r)][0])} percent)" for r, en in refs))
    largo = ("Horizontal dot plot with seven rows: T-gem, T-mine and the five T-gen generators. The horizontal axis is the "
             "share of questions in which some negative is also a correct answer, from 0 to 100 percent, with a dotted "
             "vertical line at 5 percent. Each row has three points with 95 percent confidence intervals: human annotators under the "
             "strict reference (circle), human annotators under the lenient reference (square) and the large language model "
             "judge (triangle). "
             f"The human intervals are {_u(min(x[2] - x[1] for x in hum))} to {_u(max(x[2] - x[1] for x in hum))} "
             f"percentage points wide, because each row has only {' or '.join(str(n) for n in n_h)} questions; the judge intervals "
             f"are at most {_u(max(x[2] - x[1] for x in jue))} points wide. {bajos}"
             + (f", and its judge value is the only one below the dotted line. " if bajo5 == ["tmine"] else ". ")
             + f"The highest values are for {altos}. The five T-gen generators range from "
             f"{_rango([q[(s, 'estricta')][0] for s in gens])} under the strict reference, "
             f"{_rango([q[(s, 'amplia')][0] for s in gens], '')} under the lenient one and "
             f"{_rango([q[(s, 'juez')][0] for s in gens], '')} under the judge. In {_n(len(entre))} of the "
             f"{_n(len(SETS_FN))} rows the judge "
             "falls between the two human references"
             + ("" if not fuera else "; " + "; ".join(
                 f"for {nom(s)} it is {'above' if q[(s, 'juez')][0] > max(q[(s, 'estricta')][0], q[(s, 'amplia')][0]) else 'below'}"
                 f" both ({_u(q[(s, 'juez')][0])} versus {_u(q[(s, 'estricta')][0])} and {_u(q[(s, 'amplia')][0])})"
                 for s in fuera)) + ".")
    corto = "Dot plot of the share of questions with a correct negative, per set, under two human references and a model judge."
    return corto, largo


def _desc_transferencia(raiz, t):
    T = _leer_csv(raiz, "fig_transferencia")
    pares = (("infonce_hn − infonce", "InfoNCE"), ("siglip_hn − siglip", "SigLIP"))
    g = {(r["par"], r["generador"]): (100 * float(r["dtgem"]), 100 * float(r["dtgen"])) for r in T
         if r["generador"] and not r["generador"].startswith("C2b")}
    c = {(r["par"], r["generador"]): (100 * float(r["diff"]), 100 * float(r["brecha_total"])) for r in T
         if r["generador"].startswith("C2b")}
    gemas = [v[0] for k, v in g.items() if k[1] != "macro"]
    gens_ = [v[1] for k, v in g.items() if k[1] != "macro"]
    sig_mayor = all(g[(pares[1][0], m)][0] > g[(pares[0][0], m)][0] and g[(pares[1][0], m)][1] > g[(pares[0][0], m)][1]
                    for m in MACRO)
    razon = [c[(p, "C2b:gen")][0] / c[(p, "C2b:con")][0] for p, _ in pares]
    corto = "Hard-negative gains on T-gem versus on other generators' negatives, and a split of the gap into two components."
    largo = ("Panel (a) has two dot plots side by side, one per pair of losses (InfoNCE with and without hard negatives, and "
             "SigLIP with and without), each with six rows: the five macro generators and the macro. In every row a light "
             "band joins two points: on the right, the gain on T-gem over the same items (circle), between "
             f"{_u(min(gemas))} and {_u(max(gemas))} percentage points; on the left, the gain on that generator's own "
             f"negatives (square), between {_u(min(gens_))} and {_u(max(gens_))} percentage points. The macro row shows "
             f"{_u(g[(pares[0][0], 'macro')][0])} versus {_u(g[(pares[0][0], 'macro')][1])} points for InfoNCE and "
             f"{_u(g[(pares[1][0], 'macro')][0])} versus {_u(g[(pares[1][0], 'macro')][1])} for SigLIP. The same pattern "
             "repeats for every generator" + (", and the SigLIP gains are larger than the InfoNCE gains on both measures for "
                                              "every generator. " if sig_mayor else ". ")
             + "Panel (b) is a horizontal bar chart with two pairs of bars, one pair per loss pair: the generator component "
             f"({_u(c[(pares[0][0], 'C2b:gen')][0])} and {_u(c[(pares[1][0], 'C2b:gen')][0])} percentage points) and the "
             f"construction component ({_u(c[(pares[0][0], 'C2b:con')][0])} and {_u(c[(pares[1][0], 'C2b:con')][0])}), each "
             "with its 95 percent confidence interval drawn as a line; text at the right gives the total gap, "
             f"{_u(c[(pares[0][0], 'C2b:gen')][1])} and {_u(c[(pares[1][0], 'C2b:gen')][1])} points. The generator bars are "
             + (f"about {min(razon):.1f} times" if f"{min(razon):.1f}" == f"{max(razon):.1f}"
                else f"about {min(razon):.1f} to {max(razon):.1f} times") + " as long as the construction bars.")
    return corto, largo


def _desc_denominador(raiz, t):
    Dn = _leer_csv(raiz, "fig_denominador")
    a = [r for r in Dn if r["panel"] == "a"]
    b = [r for r in Dn if r["panel"] == "b"]
    cc = [r for r in Dn if r["panel"] == "c"]
    if not (len(a) == 4 and len(b) == len(cc) == 6):
        raise SystemExit("descripción fig_denominador: el CSV no tiene la forma esperada")
    v = lambda r: 100 * float(r["media_o_diff"])
    signo = lambda x: (f"minus {_u(-x)}" if x < 0 else _u(x))
    azar = 100 / 11
    lo_c = np.floor(min(100 * float(r["lo"]) for r in cc) - 1)
    hi_c = np.ceil(max(100 * float(r["hi"]) for r in cc) + 1)
    medio = b[2:]
    largo = ("Panel (a) is a horizontal dot plot of four differences in percentage points, with a vertical line at zero. "
             "The difference between the denominator with hard negatives (HN) only and the one with HN plus in-batch "
             f"negatives is {signo(v(a[0]))} on T-gem with shuffled states, far to the right, and {signo(v(a[1]))} on T-mine, "
             "left of zero; the control, InfoNCE+HN minus InfoNCE on shuffled T-gem, "
             f"sits at {signo(v(a[2]))}, on the zero line; SigLIP+HN minus SigLIP on shuffled T-gem is {signo(v(a[3]))} "
             "(hollow diamond, descriptive). Panel (b) has two dot plots that share six rows, one per condition: (a) HN only, "
             "(b) HN plus in-batch, (c) plus the other states' hard negatives, (c31) plus 31 other hard negatives, (d31) plus "
             "31 other golds, and (e) shared hard negatives and golds. On the left, accuracy on T-gem with shuffled states: "
             f"(a) reaches {_u(v(b[0]))} percent and (b) {_u(v(b[1]))} percent, just "
             f"{'below' if v(b[1]) < azar else 'above'} chance; the four conditions with extra candidates fall in between, "
             f"from {_u(min(v(r) for r in medio))} to {_u(max(v(r) for r in medio))} percent, each labeled with its R value ("
             + ", ".join(f"{float(r['R']):.2f}" for r in medio[:-1]) + f" and {float(medio[-1]['R']):.2f}, in row order)."
             + " On the right, accuracy on T-mine on an axis from "
             f"{lo_c:.0f} to {hi_c:.0f} percent: (a) is the lowest at {_u(v(cc[0]))} percent, (b) the highest at "
             f"{_u(v(cc[1]))} percent, and the others between {_u(min(v(r) for r in cc[2:]))} and "
             f"{_u(max(v(r) for r in cc[2:]))} percent. Marker shape shows the type of extra candidate: circles for the two "
             "reference conditions, squares for other states' hard negatives, a diamond for other states' golds, and a "
             "triangle for shared ones.")
    if not (v(cc[0]) == min(v(r) for r in cc) and v(cc[1]) == max(v(r) for r in cc)):
        raise SystemExit("descripción fig_denominador: (a) ya no es el mínimo o (b) el máximo en T-mine")
    corto = "Panel (a): differences between denominators and a control; panel (b): accuracy under six denominator conditions."
    return corto, largo


def _desc_seleccion(raiz, t):
    Sx = _leer_csv(raiz, "fig_seleccion")
    a = [r for r in Sx if r["panel"] == "a"]
    b = {(r["fase"], r["metrica"]): (100 * float(r["media"]), 100 * float(r["ic95_lo"]), 100 * float(r["ic95_hi"]))
         for r in Sx if r["panel"] == "b"}
    xs = [100 * float(r["tgem_val_F1"]) for r in a]
    ys = [100 * float(r["valmine_F1b"]) for r in a]
    oy = sorted(ys)
    k = max(range(len(oy) - 1), key=lambda i: oy[i + 1] - oy[i])  # el hueco más grande en val-mine
    corte = (oy[k] + oy[k + 1]) / 2
    bajos = [(x, y) for x, y in zip(xs, ys) if y < corte]
    altos = [(x, y) for x, y in zip(xs, ys) if y >= corte]
    tg1, tm1 = b[("F1", "tgem_top1")], b[("F1", "tmine_top1")]
    tg2, tm2 = b[("F1B", "tgem_top1")], b[("F1B", "tmine_top1")]
    if not (max(tg1[0], tg2[0]) < 40 and tm2[0] > 90 and abs(tm1[0] - tg1[0]) < 2):
        raise SystemExit("descripción fig_seleccion: el patrón del panel (b) cambió; revisar el texto")
    anchos = {k_: v_[2] - v_[1] for k_, v_ in b.items()}
    ancho_max = max(anchos, key=anchos.get)
    corto = "Scatter of 32 trials under two validation metrics, and the test accuracy of the two final models."
    largo = ("Panel (a) is a scatter plot with one hollow circle per trial. The horizontal axis is T-gem-val accuracy, with "
             f"points from {_u(min(xs))} to {_u(max(xs))} percent, and the vertical axis is val-mine accuracy, from 0 to 100 "
             f"percent. The points form two groups separated by an empty band between {_u(oy[k])} and {_u(oy[k + 1])} "
             f"percent on val-mine: {len(bajos)} trials below it, spread across the whole T-gem-val range, and {len(altos)} "
             f"trials above it, between {_u(min(y for _, y in altos))} and {_u(max(y for _, y in altos))} percent on val-mine "
             f"and from {_u(min(x for x, _ in altos))} to {_u(max(x for x, _ in altos))} percent on T-gem-val. The trial "
             "chosen by T-gem-val is a square at the lower right of the cloud; the trial chosen by val-mine is a filled "
             "circle at the top right, almost directly above it, and a vertical bracket marks the distance between them. "
             "Panel (b) is a dot plot with two columns, T-gem and T-mine, showing the final model trained from each choice: "
             "a square for the T-gem-val choice and a filled circle for the val-mine choice, with 95 percent confidence "
             "intervals as vertical bars and a dashed line at chance. On T-gem both models stay below 40 percent. On T-mine "
             "the val-mine model is near the top of the axis, while the T-gem-val model stays at about the same level as on "
             "T-gem" + (f" and has the widest interval, {_u(anchos[ancho_max])} points."
                        if ancho_max == ("F1", "tmine_top1") else "."))
    return corto, largo


def _mayor_tramo_comun(a, b):
    """Largo (en palabras) del tramo común más largo entre dos textos: detecta frases copiadas de la leyenda."""
    pa, pb = re.findall(r"[a-z0-9.\-+]+", a.lower()), re.findall(r"[a-z0-9.\-+]+", b.lower())
    mejor, prev = 0, [0] * (len(pb) + 1)
    for i in range(1, len(pa) + 1):
        cur = [0] * (len(pb) + 1)
        for j in range(1, len(pb) + 1):
            if pa[i - 1] == pb[j - 1]:
                cur[j] = prev[j - 1] + 1
                mejor = max(mejor, cur[j])
        prev = cur
    return mejor


def descripciones_en(S, t):
    """Escribe descripciones/descriptions_en.tex (macros \\DescX con \\Description[corta]{larga}) y .md; valida."""
    leyendas = {n: ley for tipo, n, _, _, _, _, ley in S.leeme if tipo == "figura"}
    fn = {"fig_atajos": _desc_atajos, "fig_falsos_negativos": _desc_falsos, "fig_brazos": _desc_brazos,
          "fig_transferencia": _desc_transferencia, "fig_denominador": _desc_denominador, "fig_seleccion": _desc_seleccion}
    out, problemas = {}, []
    for nombre, f in fn.items():
        if nombre not in leyendas:
            continue
        corto, largo = f(S.raiz, t)
        for etiqueta, txt, lim in (("corta", corto, 125), ("larga", largo, 2000)):
            if len(txt) >= lim:
                problemas.append(f"{nombre}: descripción {etiqueta} de {len(txt)} caracteres (límite {lim})")
            if not txt.isascii() or any(ch in txt for ch in "\\{}$&#_%^~"):
                problemas.append(f"{nombre}: descripción {etiqueta} con caracteres que no son texto plano")
        tramo = _mayor_tramo_comun(largo, leyendas[nombre])
        if tramo >= 8:
            problemas.append(f"{nombre}: la descripción copia {tramo} palabras seguidas de la leyenda")
        out[nombre] = (corto, largo, tramo)
    if problemas:
        raise SystemExit("descripciones:\n" + "\n".join(problemas))
    os.makedirs(os.path.join(S.raiz, "descripciones"), exist_ok=True)
    tex = [f"% Figure descriptions for \\Description (JAIR template, acmart). Generated by figuras_tablas.py {VERSION}",
           "% from the CSV of each figure; do not edit by hand. Use: \\input{descriptions_en} in the preamble, then put",
           "% the macro (e.g. \\DescBrazos) inside each figure environment, after \\caption.", ""]
    md = ["# Descripciones de las figuras (`\\Description`, inglés)", "",
          f"Generado por `figuras_tablas.py` {VERSION} desde los CSV de cada figura. Validado: texto plano ASCII, sin "
          "LaTeX; corta < 125 caracteres; larga < 2000; ningún tramo de 8 palabras o más copiado de la leyenda. "
          "Verificar la letra del apéndice (\"Appendix B\") al armar el paper.", ""]
    for nombre, (corto, largo, tramo) in out.items():
        tex.append(f"\\newcommand{{\\{MACROS_DESC[nombre]}}}{{\\Description[{corto}]{{{largo}}}}}")
        md += [f"## {nombre} (`\\{MACROS_DESC[nombre]}`)", "", f"**Corta** ({len(corto)} caracteres): {corto}", "",
               f"**Larga** ({len(largo)} caracteres; tramo común más largo con la leyenda: {tramo} palabras): {largo}", ""]
    for ext, lineas in (("tex", tex), ("md", md)):
        rel = f"descripciones/descriptions_en.{ext}"
        with open(os.path.join(S.raiz, rel), "w", encoding="utf-8") as f:
            f.write("\n".join(lineas) + "\n")
        S._reg(rel, [])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raiz", required=True, help="carpeta del estudio en Drive (o su espejo), p. ej. .../CLM_losses/colab")
    ap.add_argument("--salida", required=True)
    ap.add_argument("--idioma", default="es", choices=["es", "en"])
    a = ap.parse_args()
    d, falt = cargar(a.raiz)
    obligatorias = [k for k in ENTRADAS if k not in OPCIONALES]
    faltan_obl = [ENTRADAS[k] for k in obligatorias if k not in d]
    if faltan_obl:
        raise SystemExit("Faltan entradas: " + ", ".join(faltan_obl))
    estilo()
    S = Salida(a.salida, a.idioma)
    t = TXT[a.idioma]
    for fn in (fig_atajos, fig_falsos_negativos, fig_brazos, fig_transferencia, fig_denominador):
        fn(d, S, t, a.idioma)
    if all(k in d for k in ("hpo_f1_siglip", "hpo_f1b_siglip", "hpo_f1_infonce_hn", "avs_f1", "avs_f1b")):
        fig_seleccion(d, S, t, a.idioma)
    else:
        print("Aviso: sin los hpo_summary.json de F1/F1b o sus all_vs_all.json no se genera fig_seleccion")
    for fn in (tab_contrastes, tab_panel, tab_referencias, tab_calidad, tab_sensibilidad, tab_friedman_y_secundarias,
               tab_acuerdo):
        fn(d, S, a.idioma)
    desc = descripciones_en(S, t) if a.idioma == "en" else {}
    yo = os.path.abspath(__file__)
    MANIF.update({"script": os.path.relpath(yo, os.path.dirname(os.path.dirname(os.path.dirname(yo)))),
                  "script_sha256": sha256(yo), "version": VERSION, "idioma": a.idioma,
                  "creado_utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                  "python": platform.python_version(), "matplotlib": matplotlib.__version__, "numpy": np.__version__,
                  "entradas_faltantes": falt})
    with open(os.path.join(a.salida, "MANIFIESTO.json"), "w", encoding="utf-8") as f:
        json.dump(MANIF, f, ensure_ascii=False, indent=1)
    lin = [f"# Figuras y tablas del manuscrito ({'español' if a.idioma == 'es' else 'English'})", "",
           f"Generado por `{MANIF['script']}` ({VERSION}) el {MANIF['creado_utc']}. Cada salida está en `MANIFIESTO.json` "
           "con el sha256 de sus entradas. Ninguna cifra se escribió a mano.", ""]
    for tipo, nombre, ubic, preg, est, fuentes, ley in S.leeme:
        lin += [f"## {nombre} ({tipo})", "", f"- **Dónde va:** {ubic}", f"- **Pregunta que responde:** {preg}",
                f"- **Estatus:** {est}", f"- **Fuentes:** {', '.join('`' + ENTRADAS[k] + '`' for k in fuentes if k in d)}",
                f"- **Leyenda:** {ley}", ""]
    if desc:
        lin += ["## Descripciones para `\\Description` (plantilla de JAIR)", "",
                f"{len(desc)} figuras: `descripciones/descriptions_en.tex` (una macro por figura, para `\\input`) y "
                "`descripciones/descriptions_en.md` (texto, largo y chequeos). La Fig. 1 (esquema) tiene la suya en "
                "`FIGURA_1.zip`.", ""]
    with open(os.path.join(a.salida, "LEEME.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lin))
    print(f"ok: {len(MANIF['salidas'])} archivos en {a.salida}")
    if falt:
        print("Aviso, entradas opcionales que faltan:", ", ".join(falt))


if __name__ == "__main__":
    sys.exit(main())
