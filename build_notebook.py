"""Genera CLM_losses_Colab.ipynb: el paquete clm_losses va embebido (zip en base64) y cada paso
de F0, cada brazo de F1/F3 y cada experimento de F2 tiene su propia celda.

Uso:  python build_notebook.py [--out CLM_losses_Colab.ipynb]
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from clm_losses import __version__  # noqa: E402
from clm_losses.hpo import space_for  # noqa: E402
from clm_losses.losses import ARMS, NONCONTRASTIVE_POOL  # noqa: E402
from clm_losses.utils import code_hash  # noqa: E402
from clm_losses.generators import PANEL_MIN_TRANSFORMERS  # noqa: E402


# --------------------------------------------------------------------------------------
def package_zip() -> bytes:
    """Zip determinista del paquete (mismas entradas y fechas => mismos bytes)."""
    buf = io.BytesIO()
    files = []
    for base, _, names in os.walk(os.path.join(ROOT, "clm_losses")):
        if "__pycache__" in base:
            continue
        files += [os.path.join(base, n) for n in names if n.endswith(".py")]
    for extra in ("README_CODIGO.md",):
        if os.path.exists(os.path.join(ROOT, extra)):
            files.append(os.path.join(ROOT, extra))
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for f in sorted(files):
            zi = zipfile.ZipInfo(os.path.relpath(f, ROOT), date_time=(2026, 1, 1, 0, 0, 0))
            zi.compress_type = zipfile.ZIP_DEFLATED
            zi.external_attr = 0o644 << 16
            with open(f, "rb") as fh:
                z.writestr(zi, fh.read())
    return buf.getvalue()


def md(text: str) -> dict:
    text = text.strip("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


FORM_KEYS = ("1.2 ·", "1.3 ·", "1.4 ·", "Descarga", "F0.5b", "F0.7b", "2.1 ·", "4.0 ·", "4.17 ·", "5.1 ·", "5.4 ·",
             "5.5 ·", "5.6 ·", "6.0 ·", "6.1 ·", "6.2 ·", "6.3 ·", "6.5 ·", "6.6 ·", "6.6b ·", "6.7 ·", "7.6 ·", "8.1 ·", "8.2b", "8.2c",
             "8.17b", "8.18b", "8.18c", "8.18d")


GUARD = ('if "study" not in globals():   # la sesión se reinició: rehace sola la Sección 1 (o pide correrla)\n'
         '    exec(open("/content/clm_boot.py").read() if __import__("os").path.exists("/content/clm_boot.py") '
         'else "raise SystemExit(\'Corre primero la Sección 1 (1.1 → 1.4).\')")\n')


def code(text: str, form: bool = False, guard: bool = True) -> dict:
    """form=True oculta el código detrás del formulario de Colab; sólo se usa en celdas con
    parámetros o con código largo. El resto muestra su código (1–5 líneas) para que sea auditable.
    guard=True (todas menos la Sección 1): si la sesión se reinició, la celda rehace sola la Sección 1."""
    text = text.strip("\n")
    if guard and text.startswith("#@title"):
        head, _, rest = text.partition("\n")
        text = head + "\n" + GUARD + rest
    first = text.splitlines()[0] if text else ""
    form = form and any(k in first for k in FORM_KEYS)
    meta = {"cellView": "form"} if form else {}
    return {"cell_type": "code", "execution_count": None, "metadata": meta, "outputs": [],
            "source": text.splitlines(keepends=True)}


# --------------------------------------------------------------------------------------
ARM_DOCS = {
    "infonce": ("InfoNCE bidireccional (receta CLM, sin HN)",
                "Softmax sobre el batch en ambas direcciones (s→a y a→s), temperatura aprendida. Es la referencia de C1."),
    "infonce_hn": ("InfoNCE + HN (receta CLM completa)",
                   "Pre-training in-batch y luego mid-training con los 10 HN de Gemini en s→a. **En F1b el denominador "
                   "es \"HN + in-batch\" por diseño (A1.7)**: el oro compite contra sus HN y contra las respuestas de los "
                   "otros ítems del batch. En el F1 piloto el HPO eligió \"HN solos\" (fórmula del README), que aprende "
                   "un atajo sin estado (EXP-004/005); esa variante queda como condición explícita de B1 (7.7)."),
    "siglip": ("SigLIP (sigmoide por pares)", "−Σ log σ(y_ij·(t·cos_ij + b)), con t y b aprendidos. Candidato de H2 (abstención)."),
    "siglip_hn": ("SigLIP + HN", "Igual que SigLIP, sumando los pares (s_i, HN_ik) con etiqueta −1."),
    "triplet_hn": ("Triplet con margen + HN", "Margen sobre distancia coseno: negativo semi-hard in-batch y todos los HN válidos."),
    "simsiam_x": ("SimSiam cross-modal", "½[D(p_s(z_s), sg(z_a)) + D(p_a(z_a), sg(z_s))]: predictor por torre y stop-gradient; sin negativos."),
    "byol_x": ("BYOL cross-modal", "Predictor por torre contra un target EMA de la otra torre; sin negativos."),
    "barlow_x": ("Barlow Twins cross-modal", "Correlación cruzada entre los expanders de s y de a, empujada a la identidad."),
    "vicreg_x": ("VICReg cross-modal", "Invarianza s↔a + varianza + covarianza por torre, sobre expanders."),
    "mse_sigreg": ("MSE + SIGReg", "(1−λ)‖z_s − z_a‖² + λ·½[SIGReg(z_s) + SIGReg(z_a)] (LeJEPA); sin negativos."),
    "infonce_sigreg": ("InfoNCE + SIGReg", "InfoNCE sobre z normalizado + λ·SIGReg sobre z sin normalizar."),
    "align_unif": ("Alignment + uniformity", "‖ẑ_s − ẑ_a‖² + γ/2·[U(ẑ_s) + U(ẑ_a)]. Puente: repulsión desacoplada (fuera del pool de C1)."),
    "infonce_lse": ("InfoNCE + logsumexp", "L_InfoNCE + λ·mean_i(logsumexp_j ℓ_ij)². Control simple de H4 y candidato justo de H2."),
    "dino_x": ("DINO cross-modal", "Autodestilación con prototipos compartidos, centering y maestro EMA; el par s–a son las dos vistas."),
    "swav_x": ("SwAV cross-modal", "Predicción intercambiada de códigos de Sinkhorn sobre prototipos compartidos."),
}


def hpo_space_text(arm: str, phase: str | None = None) -> str:
    sp = space_for(arm, phase)
    parts = []
    for k, v in sp.items():
        k = k.replace("loss.", "")
        if v[0] == "choice":
            parts.append(f"`{k}` ∈ {{{', '.join(map(str, v[1]))}}}")
        else:
            parts.append(f"`{k}` ~ {v[0]}[{v[1]}, {v[2]}]")
    return "; ".join(parts)


# --------------------------------------------------------------------------------------
NB_REV = "rev. 4.7 · 2026-10-07"

INTRO = f"""
# CLM — comparación controlada de 15 funciones de pérdida (todos contra todos)

Notebook de Colab del proyecto **"cosas como jev"** · código `clm_losses` v{__version__} · hash `{{HASH}}` · notebook {NB_REV}

> **Rev. 4.7 (código v0.3.7) — el estudio terminó: no hace falta volver a correrlo.**
>
> - **F3 está completo** (2026-10-07): C1 con sus dos co-primarias, C2, C2b y C2c. Todo está en tu Drive.
> - **Único cambio: la limpieza del disco al generar.** El disco se llenaba por la caché compartida de huggingface_hub: los pesos reales quedan en `hub/blobs` y lo que se borraba eran sólo enlaces. v0.3.7 la desactiva para los generadores y borra también esos archivos. Entrenamiento, evaluación y análisis no cambian.
> - **Para qué sirve:** para repetir el estudio desde cero, por ejemplo al publicar el código. Si lo corres con lo ya hecho, todo lo terminado se salta.
> - **Análisis de sensibilidad del panel:** está en otro notebook, `CLM_losses_Analisis_F3.ipynb` (sin GPU, ~5 min).
>
> **Rev. 4.6 (código v0.3.6) — completó F3.**
>
> - **Ya está hecho:** el freeze (2026-10-06 02:33, desde la rev. 4.4), los 15 brazos × 5 semillas en el test confirmatorio, C1 en T-mine y T-Rex. Todo eso se salta.
> - **Falta lo que no entró por disco lleno.** En la corrida anterior el disco local se llenó en el panel del confirmatorio, así que faltan:
>   - Phi-4 y el embebido de los 5 generadores del macro (8.2b), que dan la co-primaria de C1, y C2;
>   - los sets del piloto (8.2c), que necesita C2b.
> - **Disco:** antes de bajar cada generador se borran los pesos de los anteriores y la caché de Xet. Si igual queda poco disco, el notebook muestra qué lo ocupa.
> - **C2c:** el freeze se hizo con la rev. 4.4, que no lo registraba. La 8.1 lo registra en una **adenda** sin tocar el freeze, tal como lo fija ENMIENDA_03 §14, y la 8.17b entrena la variante "HN solos".
> - **Duración:** ~1.5–2 h (datos ~20 min, Phi-4 y el embebido ~40 min, sets del piloto ~30–60 min, variante y análisis ~20 min). Si Colab se corta, vuelve a darle Ejecutar todo.
>
> **Rev. 4.5 (código v0.3.5):** freeze confirmado por el investigador (2026-10-05) y C2c pre-especificado (ENMIENDA_03 §14).
>
> - **Confirmaste el PROTOCOL FREEZE el 2026-10-05**, en la conversación del proyecto. Por eso la casilla `CONFIRMO_FREEZE` de la **8.1** ya viene marcada, y la nota del freeze registra tu decisión.
> - **Una sola acción:** Entorno de ejecución → **Ejecutar todo**. Lo exploratorio ya terminado se salta en minutos. Después la 8.1 congela el protocolo y la Sección 8 corre F3 completo.
> - **Duración:** ~10–11 h. El panel del confirmatorio toma ~8.6 h (casi todo gpt-oss), los sets del piloto ~1 h y el entrenamiento ~1 h.
> - **Si Colab se corta:** vuelve a darle Ejecutar todo. El freeze ya hecho no se repite y cada celda retoma donde quedó.
> - **Nuevo en el freeze: C2c** (8.17b y 8.18d), sólo ~10 min más. Confirma en el test sellado el mecanismo de H1b: si comparar el oro sólo contra sus HN ("HN solos") aprende una preferencia que no depende del estado y cuesta discriminación real (ENMIENDA_03 §14). C1, C2 y C2b no cambian.
>
> **Rev. 4.4 (código v0.3.4):** C2b (8.2c y 8.18c). Además, la 6.2 ya no vuelve a embeber sets idénticos: compara su contenido, no la fecha de modificación de los archivos en Drive.
>
> **Rev. 4.3 (código v0.3.3):** Gemma 4 corregido, matriz de ciegos sin fuga por ítem, EXP-006c (6.6b) y C2 en F3 (8.18b).
>
> **Rev. 4.2 (código v0.3.2) — "Ejecutar todo" corre el estudio completo, sin opciones.**
>
> - **Una sola acción:** Entorno de ejecución → **Ejecutar todo**. Si Colab se corta, vuelve a Ejecutar todo: lo terminado se salta solo.
> - **Ningún formulario que llenar.** La autorización para entrenar F1b queda registrada sola en la 4.0, citando tu decisión (enmienda A1 aprobada el 2026-10-01; pedido del 2026-10-02 de correr todo).
> - **La única pausa es el PROTOCOL FREEZE (8.1):** congelar es irreversible y tu protocolo exige decidirlo después de ver F1b. Cuando quieras congelar, marca la casilla y vuelve a Ejecutar todo.
> - **Panel T-gen corregido (ENMIENDA_03 §11).** La v1 salió inutilizable: en los 16 generadores un modelo que no ve la pregunta encontraba el oro el 77–91 % de las veces. Ahora un **piloto automático (6.0)** prueba tres construcciones y se queda con la de menor atajo, por regla fija: **S** (reglas de texto), **P** (el oro reescrito por el mismo generador, con largo asignado al azar) y **E** (negativos que sólo cambian el dato clave del oro).
> - **Elección automática de los brazos de B1-R2 (7.6)** con una regla fijada antes de F1b.
> - Si la sesión se reinicia, cualquier celda rehace sola la Sección 1.
>
> **Rev. 4–4.1 (código v0.3–0.3.1):** enmienda A1 (primaria T-mine, T-gen co-primaria, selección por val-mine, T-abst-mine, estados barajados, T-mine-50), F0.7b, F1b y el panel de 23 generadores (16 de 3–9B y 7 de 14–34B).
>
> **Lo ya corrido** (F0, F1 piloto, F0.7, F0.7b, las generaciones del panel v1) se conserva: el código nuevo reutiliza lo generado y sólo agrega lo que falta.

| | |
|---|---|
| **Qué implementa** | Los cabezales de CLM (4096→1536→1536→512 con LayerNorm, confirmado en F0.1) sobre embeddings congelados de Qwen3-8B, entrenados desde cero con cada una de las 15 pérdidas bajo el mismo protocolo. Mismos datos y splits por documento, misma arquitectura, mismo presupuesto de tuning (32 trials al 25 %), selección en validación (val-mine en F1b) y 3 semillas por brazo. |
| **Por qué** | Capa A (benchmark controlado; RQ1, RQ4) y capa B (B1 atajos de HN, B2 abstención, B3 mecanismo de SIGReg) de `DISENO_01`, con la enmienda A1 y el panel T-gen de `ENMIENDA_03`. |
| **Qué prueba** | H1a/C1 (mejor no contrastiva vs InfoNCE, en T-mine y T-gen), H1b/C2 (¿la ganancia por HN es del generador? transferencia a 23 generadores y estados barajados), H2/C3 (abstención en T-abst-mine), H4/C4 (SIGReg con HN desde el inicio). Además, las **105 comparaciones por pares** en cada métrica, con Holm. |
| **Qué produce** | Todo en tu Drive: splits, sets de evaluación, negativos de cada generador con su auditoría, checkpoints, logs, métricas por ítem, tablas por pares, mapas de calor y reportes. |
| **Qué cuenta como fallo** | Una compuerta que no pasa, un colapso sin explicar, menos de 5 generadores sin atajo alto en el panel, o un resultado que depende de un bug. Ningún umbral ni regla se ajusta después de ver el resultado. |

**Qué hace "Ejecutar todo", en orden** (todo se reanuda solo):

**Con lo ya corrido:** no queda nada pendiente. "Ejecutar todo" hace la prueba de humo del código nuevo y se salta todo lo terminado, F3 incluido; no re-entrena nada.

| Paso | Celdas | Tiempo aprox. **[SUPUESTO; el notebook mide]** |
|---|---|---|
| Entorno y prueba de humo (una vez por versión del código) | 1, 2.1 | ~2 min + ~20 min |
| F0 (ya terminado: sólo muestra) | 3 | minutos |
| Autorización y datos de entrenamiento | 4.0 | ~10–20 min de descarga |
| **F1b**: los 15 brazos | 4.1–4.15 | **~3 h** (medido en el piloto) |
| Análisis de F1b | 5 | minutos |
| **Piloto del panel** | 6.0 | ~20–40 min |
| **Panel completo** | 6.1–6.3 | **~5–16 h** según la construcción que gane (S ~5–9 h, P ~6–11 h, E ~12–16 h) |
| T-gen en lo ya entrenado, EXP-006 y EXP-006c | 6.5–6.6b | ~30–60 min |
| F2b (B3, B2, B1) | 7 | ~3–5 h |
| **PROTOCOL FREEZE** | 8.1 | **hecho el 2026-10-06; adenda de C2c el 2026-10-07** |
| F3 confirmatorio (después del freeze) | 8.2–8.19 | **hecho** (2026-10-06/07): los 15 brazos, el panel y los sets del piloto del confirmatorio, la variante de C2c, C1, C2, C2b, C2c y T-Rex |
"""

RESUME = """
## Si Colab se corta (o cierras la pestaña)

1. Reconecta. Si la máquina cambió, el disco local se borró; **Drive no**.
2. Ejecuta la **Sección 1** completa (1.1 → 1.4). Tarda 1–2 minutos.
3. Vuelve a **Ejecutar todo**. La 4.0 vuelve a bajar los datos públicos que falten en el disco local y cada celda retoma donde quedó.
4. Si sólo reiniciaste la sesión (misma máquina), puedes correr directamente cualquier celda: si falta la Sección 1, la celda la rehace sola.

Cada corrida guarda `ckpt_last.pt` en Drive cada 200 pasos o cada 10 minutos (lo que ocurra primero) y continúa **en el mismo batch**, con el mismo estado de optimizador, scheduler, semillas y posición en los datos. Lo terminado tiene `DONE.json` y se salta. Los pasos de F0 guardan `result.json` y tampoco se repiten. La generación de negativos guarda cada lote en `raw.jsonl` y retoma donde quedó.

**Carpetas en Drive** (`MyDrive/CLM_losses/<perfil>/`):

```
config/study.json                configuración congelada del estudio (no cambia entre sesiones)
config/F0_DECISIONES.jsonl       decisiones registradas en la 4.0 (con el hash del código)
config/RECETA_F1b.json           receta de F1b que salió de la regla de F0.7b
config/PROTOCOL_FREEZE.json|md   se crea en la Sección 8
data/splits/                     documentos de test, val y confirm (sellado) + máscaras de entrenamiento
data/eval/{val,test,confirm}/    sets de evaluación (1 oro + 10 HN); pools de T-rand, T-mine, T-mine-50 y val-mine
data/eval/tgen/BUILD.json        construcción del panel elegida por el piloto (S, P o E) y su tabla
data/eval/tgen/test/<generador>/ generaciones (raw*.jsonl, para.jsonl), set construido (negs.npy, pos.npy en P),
                                 auditoría, QC; pilot_<S|P|E>/ = sets del piloto; history/ = versiones anteriores
data/eval/tgen/test/PANEL.json   qué generadores entran al macro de T-gen (regla fija)
f0/F0.x_*/                       resultado de cada paso de F0 (incluye F0.5b y F0.7b) + F0_REPORT.md
runs/F0.7b/<receta>/<condición>/seed_N/   corridas de F0.7b
runs/F1b/<brazo>/hpo/trial_XX/   trials de la búsqueda de hiperparámetros (F1 = piloto)
runs/F1b/<brazo>/seed_N/         config.json, log.jsonl, stage_*.json, best_*.pt, eval/, DONE.json
results/F1b/all_vs_all/          pairwise_*.csv, summary_*.csv, heatmap_*.png, cd_diagram.png, all_vs_all.json
results/transfer/EXP006/         transferencia entre generadores
code/<hash>/                     copia legible del código con el que se corrió
logs/                            entorno y pip freeze de cada sesión
```

Los embeddings públicos (decenas de GB) van al disco local de la máquina (`/content/cache`) y se vuelven a bajar solos. Si tu Drive tiene espacio de sobra, puedes poner `CACHE_ROOT` dentro de Drive para no re-descargar (las lecturas serán más lentas).
"""

RESOURCES = """
## GPU, disco y tiempos

**Medido (2026-10-01, perfil `colab`, NVIDIA RTX PRO 6000 Blackwell de 96 GB):**

- F1 piloto completo (15 brazos × 32 trials + 3 semillas) tardó ≈ 3 h 10 min, con una mediana de 0.7 min por corrida completa.
- En F0.7, con los datos todavía fuera de la RAM, una corrida pre→mid tardó ~11 min.
- La primera pasada por los datos lee del disco; las siguientes, de la RAM.

| Paso (rev. 4) | Tiempo aprox. **[SUPUESTO; el notebook mide]** |
|---|---|
| 3.6b · F0.5b (pools) | minutos |
| 3.8b · F0.7b (2 recetas × 4 condiciones × 2 semillas) | ~1–2 h con los datos en RAM; el peor caso (HN desde el inicio con la receta de CLM) puede sumar ~1 h |
| 4 · F1b, receta actual | ~3–4 h (como el piloto) |
| 4 · F1b, receta de CLM (más muestras por corrida) | ~5–8 h |
| 6.0 · Piloto del panel (500 ítems × 3 generadores × 3 construcciones) | ~20–40 min |
| 6.1 · Un generador de 3–9B, negativos (medido 2026-10-02: 2,000 ítems) | 3–14 min |
| 6.1 · gpt-oss-20b, negativos (medido) | ~4 h |
| 6.1 · Qwen3-30B-A3B, 1,000 ítems (medido) | ~25 min |
| 6.1 · Un generador de 14–34B, negativos | ~20–80 min |
| 6.1 · Reescrituras del oro (construcción P), por generador | ~2–10 min |
| 6.1 · Panel completo: S / P / E | ~5–9 h / ~6–11 h / ~12–16 h |
| 6.2–6.6 · Embeber, auditar, evaluar lo ya entrenado y EXP-006 | ~30–60 min |

**Unidades de cómputo:** pon tu consumo por hora en la 4.17. Con 18.87 unidades/h, cada hora de GPU son ~19 unidades.

**Disco local:**

- El perfil `colab` usa ~66 GB de datos de entrenamiento.
- Cada generador baja 5–70 GB de pesos, que se **borran** al terminar. Los grandes necesitan ~70 GB libres mientras corren.
- Qwen3-8B (embebido) ocupa ~16 GB.

**Usa siempre la misma GPU** en todas las corridas de entrenamiento (F0.7b, F1b, F2b, F3), para que el tipo de GPU no sea un confusor.

**Licencias:**

- Nemotron-CC-v2.1 se rige por un acuerdo de NVIDIA (acceso restringido).
- Los generadores tienen sus propias licencias (tabla de la Sección 6).
- No redistribuyas datos derivados sin revisarlo **[AUTHOR DECISION NEEDED]**.
"""

S1 = "# 1 · Entorno — ejecutar SIEMPRE al conectar o reconectar"

C11 = """
#@title 1.1 · Parámetros de la sesión (fijos: no hay nada que cambiar)
PROFILE = "colab"
DRIVE_BASE = "/content/drive/MyDrive/CLM_losses"
CACHE_ROOT = "/content/cache"
USE_DRIVE_CODE = False                            # True: usa DRIVE_BASE/code/clm_losses_pkg.zip en vez del embebido
HPO_TRIALS, F1_SEMILLAS, F3_SEMILLAS = 32, 3, 5   # congelados con el estudio (P9); cambiarlos detiene el código
print(f"perfil={PROFILE}  drive={DRIVE_BASE}  cache={CACHE_ROOT}  trials={HPO_TRIALS}  "
      f"semillas F1/F3={F1_SEMILLAS}/{F3_SEMILLAS}")
"""

C12 = """
#@title 1.2 · Montar Drive, ver la GPU e instalar dependencias
import os, shutil, subprocess, sys
from google.colab import drive
drive.mount("/content/drive")
os.makedirs(DRIVE_BASE, exist_ok=True)
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else ""
print("GPU:", gpu or "ninguna (sólo sirve para el smoke y el análisis)")
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)
pip("huggingface_hub>=0.24", "pyarrow>=14", "scipy", "pandas", "matplotlib", "transformers>={TF_MIN}", "accelerate>=0.33")
# transformers se instala aquí, antes de importarlo: actualizarlo a mitad de sesión mezcla versiones en memoria
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN cargado desde los secretos de Colab")
except Exception:
    print("Sin HF_TOKEN (opcional): el panel T-gen saltará los 4 generadores gated (Llama 3.1 8B, Aya Expanse 8B y 32B, Gemma 2)")
"""

C13 = """
#@title 1.3 · Instalar el código del estudio (embebido en el notebook)
import base64, hashlib, io, shutil, zipfile
CODE_ZIP_B64 = "{B64}"
CODE_ZIP_SHA256 = "{SHA}"
PKG_ROOT = "/content/clm_losses_pkg"
drive_zip = os.path.join(DRIVE_BASE, "code", "clm_losses_pkg.zip")
if USE_DRIVE_CODE and os.path.exists(drive_zip):
    data, origen = open(drive_zip, "rb").read(), drive_zip
else:
    data, origen = base64.b64decode(CODE_ZIP_B64), "embebido"
    assert hashlib.sha256(data).hexdigest() == CODE_ZIP_SHA256, "el código embebido está corrupto"
shutil.rmtree(PKG_ROOT, ignore_errors=True)
zipfile.ZipFile(io.BytesIO(data)).extractall(PKG_ROOT)
open("/content/clm_code.zip", "wb").write(data)       # para rehacer la Sección 1 sola si la sesión se reinicia
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
os.environ["PYTHONPATH"] = PKG_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
for m in [m for m in list(sys.modules) if m == "clm_losses" or m.startswith("clm_losses.")]:
    del sys.modules[m]
import clm_losses
from clm_losses.utils import code_hash
CODE_HASH = code_hash()
snap = os.path.join(DRIVE_BASE, "code", CODE_HASH)
if not os.path.exists(snap):
    shutil.copytree(os.path.join(PKG_ROOT, "clm_losses"), snap)
print(f"clm_losses {clm_losses.__version__} ({origen}) · hash {CODE_HASH} · copia legible: {snap}")
"""

C13_BOOT = """
import io, zipfile
PKG_ROOT = "/content/clm_losses_pkg"
if not os.path.isdir(os.path.join(PKG_ROOT, "clm_losses")):
    zipfile.ZipFile(io.BytesIO(open("/content/clm_code.zip", "rb").read())).extractall(PKG_ROOT)
if PKG_ROOT not in sys.path:
    sys.path.insert(0, PKG_ROOT)
os.environ["PYTHONPATH"] = PKG_ROOT + os.pathsep + os.environ.get("PYTHONPATH", "")
import clm_losses
from clm_losses.utils import code_hash
CODE_HASH = code_hash()
print(f"clm_losses {clm_losses.__version__} (sesión reiniciada: Sección 1 rehecha sola) · hash {CODE_HASH}")
"""

C14_CORE = """
#@title 1.4 · Configuración del estudio y chequeo de recursos
import json
import numpy as np
import pandas as pd
import torch
from IPython.display import Image, Markdown, display
from clm_losses import f0
from clm_losses import generators as tgen
from clm_losses.analysis import (c2_report, c2b_report, c2c_report, construction_report, f07b_table,
                                 state_use_table, transfer_report)
from clm_losses.losses import ARMS
from clm_losses.pipeline import (setup, resources, load_bundle, run_arm, summarize_arm, progress, run_all_vs_all,
                                 run_variant, compare_variants, plot_val_curves, protocol_freeze, run_confirm_arm,
                                 run_confirm_variant, freeze_addendum, frozen_variants,
                                 add_missing_evals, trex_for_arms, export_summary_md, profile_disk_table,
                                 estimate_budget, tunability_report, freeze_path, recipe_decision, default_testsets,
                                 top_arms_rule)
from clm_losses.utils import append_jsonl, load_json, makedirs, now_iso, read_done, read_jsonl, save_json
import warnings
warnings.filterwarnings("ignore", message="To exit")   # las paradas controladas (SystemExit) no son errores
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 40)

# --- comprobaciones que usan las celdas siguientes (avisan qué falta en vez de fallar) ---
def datos_listos():
    ok = all(os.path.exists(study.paths.d(*p)) for p in (("data", "splits", "split.json"),
                                                        ("data", "eval", "test", "manifest.json"),
                                                        ("data", "eval", "val", "manifest.json")))
    if not ok:
        print("Faltan los splits o los sets de evaluación: corre primero F0.4 y F0.5 (Sección 3). No se hace nada.")
    return ok

def get_bundle():
    global bundle
    b = globals().get("bundle")
    if b is not None and b.val_mine_idx is None and os.path.exists(study.paths.d("data", "eval", "val", "mine_idx.npy")):
        b = None                      # F0.5b corrió después de cargar los datos: se recarga con val-mine
    if b is None:
        bundle = load_bundle(study) if datos_listos() else None
    return bundle

def f07b_completo():
    r = load_json(study.paths.d("f0", "F0.7b_recipes", "result.json")) or {}
    claves = [f"{rec}/{c}/seed_{s}" for rec in f0.F07B_RECIPES for c in f0.F07B_CONDITIONS for s in (0, 1)]
    return all((r.get("runs", {}).get(k) or {}).get("status") == "ok" for k in claves)

def f1b_listo(avisar=True):
    falta = []
    if not os.path.exists(study.paths.d("data", "eval", "val", "mine_idx.npy")):
        falta.append("F0.5b (pool de val-mine, celda 3.6b)")
    if recipe_decision(study) is None:
        falta.append("la decisión de receta de F0.7b (celda 3.8b)")
    if falta and avisar:
        print("F1b todavía no puede empezar: falta " + " y ".join(falta) + ". Esta celda no entrena nada.")
    return not falta

def brazos_terminados(fase="F1"):
    out = []
    for a in ARMS:
        base = study.paths.d("runs", fase, a)
        if os.path.isdir(base) and any((read_done(os.path.join(base, d)) or {}).get("status") == "ok"
                                       for d in os.listdir(base) if d.startswith("seed_")):
            out.append(a)
    return out

def brazo_listo(arm, fase="F1"):
    ok = os.path.exists(study.paths.d("runs", fase, arm, "hpo_summary.json")) and arm in brazos_terminados(fase)
    if not ok:
        print(f"Falta {fase} del brazo {arm}: corre primero su celda de la Sección 4. No se hace nada.")
    return ok

def freeze_hecho():
    if not os.path.exists(freeze_path(study)):
        print("El test confirmatorio sigue sellado: falta el PROTOCOL FREEZE (celda 8.1). No se hace nada.")
        return False
    return True

def compuertas():
    return f0.f0_report(study, log=lambda *a, **k: None)["gates"]

def brazo_completo(arm, fase="F1"):
    base = study.paths.d("runs", fase, arm)
    semillas = study.f3_seeds if fase == "F3" else study.f1_seeds
    return os.path.exists(os.path.join(base, "hpo_summary.json")) and all(
        (read_done(os.path.join(base, f"seed_{s}")) or {}).get("status") == "ok" for s in semillas)

NB_REV = "{NB_REV}"
# Decisión del investigador que autoriza entrenar con G2/G3 fallidas (se registra sola en la 4.0, con el hash)
DECISION_A1 = ("Enmienda A1 aprobada por el investigador el 2026-10-01 como respuesta a G2 y G3 (primaria T-mine, "
               "T-gen co-primaria, selección por val-mine, receta decidida en F0.7b). El 2026-10-02 pidió que "
               "'Ejecutar todo' corra el estudio completo sin opciones; la única decisión pendiente es el freeze.")

def entorno_panel_ok():
    # Para generar y embeber textos nuevos: transformers sin mezclar versiones y G1 bueno en respuestas y negativos.
    import importlib.metadata as _md, transformers as _tf
    if _tf.__version__ != _md.version("transformers"):
        print(f"transformers cambió después de importarlo ({_tf.__version__} en memoria, {_md.version('transformers')} "
              "instalado): reinicia la sesión (Entorno de ejecución → Reiniciar sesión) y vuelve a Ejecutar todo.")
        return False
    r03 = load_json(study.paths.d("f0", "F0.3_embeddings", "result.json"))
    ok = lambda m: m["median"] >= 0.999 and m["p01"] >= 0.995
    if r03 is None:
        print("Falta F0.3 (G1): sin esa verificación no se embeben textos nuevos.")
        return False
    if not (ok(r03["a"]) and ((r03.get("neg_order") or {}).get("diag_cos_mean") or 0) >= 0.999):
        print("G1 falla en las respuestas o en los negativos: no se embeben textos nuevos (mira F0.3).")
        return False
    return True

def encoder_verificado():
    # Qwen3-8B para embeber textos nuevos, sólo si G1 se reproduce con el entorno de ESTA sesión.
    from clm_losses.encoder import make_encoder
    enc = make_encoder("hf")
    if not f0.g1_recheck(study, enc)["ok"]:
        del enc
        raise RuntimeError("G1 no se reproduce con el entorno actual: no se embebe nada (los textos quedarían en otro "
                           "espacio que el oro). Detalle en f0/F0.3_embeddings/recheck_*.json.")
    return enc

def liberar_gpu():
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def entrenamiento_autorizado(avisar=True):
    # Las celdas que ENTRENAN comparaciones (Secciones 4 y 7) sólo corren si G2–G4 pasan o si registraste una
    # decisión en la 4.0 CON ESTE CÓDIGO (hash). F0.7b es parte de F0 y no necesita esta autorización.
    if PROFILE == "smoke":
        return True
    malas = [k.split()[0] for k, v in compuertas().items() if k.split()[0] in ("G2", "G3", "G4") and v is not True]
    if not malas:
        return True
    if any(d.get("code_hash") == CODE_HASH and str(d.get("motivo", "")).strip()
           for d in read_jsonl(study.paths.d("config", "F0_DECISIONES.jsonl"))):
        return True
    if avisar:
        print(f"Entrenamiento no autorizado: {malas} no pasaron y no hay una decisión registrada para este código "
              f"({CODE_HASH}, celda 4.0). Esta celda no entrena nada.")
    return False

tabla = profile_disk_table(CACHE_ROOT)
print(f"Disco local libre en esta máquina: {tabla.attrs['disco_libre_gb']} GB")
display(tabla)
fila = tabla[tabla.perfil == PROFILE]
if PROFILE != "smoke" and not bool(fila["cabe_para_entrenar"].iloc[0]):
    raise SystemExit(f"El perfil {PROFILE} no cabe en el disco local de esta máquina: elige uno que diga True en "
                     "'cabe_para_entrenar' (o pon CACHE_ROOT en Drive). No se creó nada.")
if PROFILE != "smoke" and not bool(fila["cabe_con_todo"].iloc[0]):
    print("Aviso: con este perfil, los pasos que corren Qwen3-8B (F0.3, F0.9, T-gen2) conviene hacerlos en una "
          "sesión aparte (la máquina nueva empieza con el disco vacío).")
study = setup(PROFILE, drive_base=DRIVE_BASE, cache_root=CACHE_ROOT, hpo_overrides={"n_trials": HPO_TRIALS},
              study_overrides={"f1_seeds": list(range(F1_SEMILLAS)), "f3_seeds": list(range(F3_SEMILLAS))})
print("Carpeta del estudio en Drive:", study.paths.drive_root)
print(json.dumps(resources(study), indent=1))
"""

C14_WRITER = """
import base64 as _b64
open("/content/clm_boot.py", "w").write(_b64.b64decode("{BOOT_B64}").decode())   # Sección 1, por si la sesión se reinicia
"""

S2 = """
# 2 · Prueba de humo (una vez por versión del código; datos sintéticos, no descarga nada)

Recorre en pequeño **todo** el flujo de v0.3.4:

- F0.4–F0.8, F0.5b y F0.7b, con su decisión de receta;
- el **piloto del panel** y un panel T-gen **sintéticos**, con su auditoría y su macro;
- F1 piloto (4 brazos) y F1b (15 brazos con HPO);
- todos contra todos con T-mine y el macro de T-gen;
- la transferencia (EXP-006), su descomposición con el piloto (EXP-006c) y F2b;
- el PROTOCOL FREEZE y un F3 reducido sobre el test sellado, con C1, C2 y C2b.

Usa su propia carpeta (`CLM_losses/smoke`), así que no toca el estudio real. Si ya pasó con este mismo código, se salta.

**Qué mirar:**

- **Piloto sintético.** S trae un atajo de longitud y E uno en el embedding, así que debe ganar **P**.
- **Panel sintético.** El generador de artefacto fuerte (`syn_fuerte`) debe quedar **fuera** del macro, por atajo ciego alto, y el limpio **dentro**.

Si eso aparece, la auditoría y la regla del piloto saben detectar atajos. El smoke **no** dice nada sobre las pérdidas ni sobre generadores reales.
"""

C21 = """
#@title 2.1 · Prueba de humo (una vez por versión del código)
from clm_losses.smoke import run_smoke
smoke_dir = os.path.join(DRIVE_BASE, "smoke")
smoke = load_json(os.path.join(smoke_dir, "results", "SMOKE_SUMMARY.json")) or {}
if smoke.get("code_hash") == CODE_HASH:
    print(f"La prueba de humo ya pasó con este código ({CODE_HASH}; tardó {smoke.get('minutes')} min): se salta.")
else:
    if smoke:            # humo de otra versión del código: se rehace desde cero (sólo la carpeta smoke)
        shutil.rmtree(smoke_dir, ignore_errors=True)
        shutil.rmtree(os.path.join(CACHE_ROOT, "_smoke"), ignore_errors=True)
    smoke = run_smoke(DRIVE_BASE, cache_root=os.path.join(CACHE_ROOT, "_smoke"), mount=False)
    print(f"listo en {smoke['minutes']} min · receta de F1b en el smoke: {smoke['receta_F1b']}")
print("Modelo ciego (top-1):", smoke["F0.6"]["blind"], "->", smoke["F0.6"]["G3"])
if smoke.get("piloto"):
    print("Piloto sintético del panel -> construcción", smoke["piloto"]["build"], "· medianas", smoke["piloto"]["medianas"])
print("Panel sintético, macro:", smoke["panel"]["macro"])
display(pd.DataFrame(smoke["panel"]["tabla"]))
"""

S3 = """
# 3 · F0 — compuertas antes de cualquier comparación

Nada de F1 debería arrancar sin pasar F0. Cada paso guarda su resultado en `f0/F0.x_*/` y no se repite.

**Umbrales (P4), fijados antes de correr:**

- **G1:** mediana ≥ 0.999 y p01 ≥ 0.995.
- **G2:** orden mid > pico con HN desde el inicio > pre. Con datos completos, además 52.1 ± 3 y 69.2 ± 3 pp.
- **G3** (modelo ciego en T-gem): < 15 % despreciable; 15–30 % moderado; > 30 % cambia la métrica primaria.
- **G4:** colapso = RankMe < 5 % de la dimensión, o T-rand a ≤ 2 pp del azar.
- **G5:** acuerdo 0.658 ± 0.05 con shield, 5/5 sobrevive y sin *stall*.
- **G6:** exactamente 31/38.

**Enmienda A1:**

- **F0.5b** (3.6b) construye los pools de val-mine y T-mine-50.
- **F0.7b** (3.8b) mide la G2 en T-mine con las dos recetas (**G2b**) y decide la receta de F1b con una regla fijada de antemano.
"""

F0_STEPS = [
    ("3.1 · F0.1 — Auditoría de los cabezales públicos de CLM",
     "Carga los checkpoints pre, mid, post y DeepSWE: `cfg`, parámetros, `logit_scale` y SHA-256. **Por qué:** fijar la arquitectura común (D6). **Criterio:** si `matches_our_arch` es False, se adopta la arquitectura real para todos los brazos **antes** de entrenar.",
     """
#@title F0.1 · Checkpoints públicos
r01 = f0.f0_1_checkpoints(study)
for k, v in r01["heads"].items():
    print(k, "->", v.get("cfg_effective"), "params:", v.get("n_params"), "escala:", v.get("scale"), v.get("error", ""))
print("¿Coincide con nuestra arquitectura?", r01["matches_our_arch"])
if not r01["matches_our_arch"]:
    print("NO SIGAS: la arquitectura común debe ajustarse a la real antes de entrenar (D6).")
"""),
    ("3.2 · F0.2 — Manifiesto de datos",
     "Tamaños, SHA-256 (según Hugging Face) y licencias de cada archivo que usa el perfil. **Por qué:** los repos personales pueden cambiar o desaparecer.",
     """
#@title F0.2 · Manifiesto
r02 = f0.f0_2_manifest(study)
print("GB listados:", r02["total_gb_listed"]); print(json.dumps(r02["licenses_known"], indent=1, ensure_ascii=False))
"""),
    ("3.3 · Descarga de los embeddings públicos al disco local",
     "Baja los chunks del perfil (pares q/a y HN). Se puede interrumpir: los archivos completos no se vuelven a bajar.",
     """
#@title Descarga (reanudable)
#@markdown Baja los datos públicos de entrenamiento al disco local (≈ 65 GB con el perfil `colab`). Hace falta para F0.7b y para entrenar (F1b, F2b). Si F0 está completo, la baja la 4.0 (después de registrar la autorización).
f0_hecho = all(os.path.exists(study.paths.d("f0", d, "result.json"))
               for d in ("F0.4_splits", "F0.5_eval_sets", "F0.6_shortcuts", "F0.7_reproduction", "F0.8_collapse"))
if f0_hecho and f07b_completo() and not entrenamiento_autorizado(avisar=False):
    print("F0 (incluido F0.7b) ya está completo: los datos de entrenamiento los baja la 4.0.")
else:
    r_dl = f0.f0_download(study)
    print(json.dumps(r_dl, indent=1))
"""),
    ("3.4 · F0.3 — Equivalencia de embeddings (G1) · requiere L4 o A100",
     "Re-embebe 2,000 pares desde sus token IDs exactos con Qwen3-8B (last-token, L2) y compara con los embeddings públicos; también verifica el orden de los HN. **Por qué:** T-gen2 y T-mine necesitan embeber textos nuevos con el mismo pipeline.",
     """
#@title F0.3 · G1 (Qwen3-8B; necesita ≥ 24 GB de GPU)
import torch
mem = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0
if mem < 22:
    print(f"GPU con {mem:.0f} GB: cambia el entorno a L4 o A100 para este paso (el resto de F0 corre en T4).")
else:
    pip("transformers>=4.51", "accelerate")
    r03 = f0.f0_3_embedding_equivalence(study, backend="hf")
    print("G1:", "PASA" if r03["G1_pass"] else "NO PASA", "| q:", r03["q"], "| a:", r03["a"], "| HN:", r03["neg_order"])
    ok = lambda m: m["median"] >= 0.999 and m["p01"] >= 0.995
    n_ok = (r03.get("neg_order") or {}).get("diag_cos_mean") or 0
    print(f"estados {'OK' if ok(r03['q']) else 'NO'} · respuestas {'OK' if ok(r03['a']) else 'NO'} · "
          f"negativos {'OK' if n_ok >= 0.999 else 'NO'}")
    if ok(r03["a"]) and n_ok >= 0.999 and not ok(r03["q"]):
        print("Sólo fallan los estados: los embeddings públicos de las preguntas no salen sólo de sus tokens "
              "(hipótesis: incluyen el documento como contexto). No afecta a F1 ni a T-gen2 (sólo se embeben "
              "negativos); sí afecta si algún día se embeben estados nuevos.")
"""),
    ("3.5 · F0.4 — Splits por documento",
     "Asignación determinista por hash del `doc_id`: test (exploratorio), val y **confirm (sellado)** salen sólo de documentos con HN; ningún documento retenido aparece en entrenamiento. **Criterio:** 0 filas con fuga y alineación por (`doc_id`, `pair_idx`) entre pares y HN.",
     """
#@title F0.4 · Splits
r04 = f0.f0_4_splits(study)
print(json.dumps(r04, indent=1))
"""),
    ("3.6 · F0.5 — Sets de evaluación",
     "T-gem (1 oro + 10 HN, ≤ 2 ítems por documento), val, el test confirmatorio sellado, índices de T-rand y T-mine, longitudes (para la sonda de atajo) y un CSV de 200 ítems para revisar a mano falsos negativos.",
     """
#@title F0.5 · Sets de evaluación
r05 = f0.f0_5_eval_sets(study)
print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "counts"}) for k, v in r05.items()}, indent=1, ensure_ascii=False))
"""),
    ("3.6b · F0.5b — Pools de val-mine (selección, A1.2) y T-mine-50 · enmienda A1",
     "Construye **val-mine**: a cada ítem de val, 10 respuestas minadas de otros documentos de val (espacio crudo del encoder, coseno ≤ 0.95), con el mismo procedimiento que T-mine. **Por qué:** T-gem-val premiaba puntuar sin mirar el estado (EXP-001/002); desde F1b, el HPO, el early stopping y el brazo de C1 se eligen con val-mine. También arma **T-mine-50** (50 minadas; secundaria, para el techo de T-mine) en test y confirm, sin evaluar ningún modelo. Tarda minutos.",
     """
#@title F0.5b · Pools de val-mine y T-mine-50
if datos_listos():
    r05b = f0.f0_5b_pools(study)
    print(json.dumps(r05b, indent=1, ensure_ascii=False))
    print("Línea base sin entrenamiento (coseno crudo): val-mine", round(r05b["val_mine"]["raw_cosine_top1"], 4),
          "· T-mine-50", round((r05b.get("test_mine50") or {}).get("raw_cosine_top1", float("nan")), 4))
"""),
    ("3.7 · F0.6 — Sondas de atajo (G3)",
     "Un modelo **ciego** (ve sólo las acciones, nunca el estado) y la heurística de longitud, en T-gem, T-rand, T-mine y T-gen2. **Por qué:** medir cuánto se puede acertar sin mirar el estado. Incluye la línea base sin entrenamiento (coseno crudo del encoder).",
     """
#@title F0.6 · G3
if get_bundle() is not None:
    r06 = f0.f0_6_shortcuts(study, bundle)
    print("azar:", round(r06["chance"], 4), "| ciego:", r06["blind"], "| longitud:", r06.get("length_heuristic"), "->", r06["G3"])
    print("línea base sin entrenamiento (coseno crudo del encoder):", r06.get("raw_cosine"))
    if not r06["G3_pass"]:
        lh = (r06.get("length_heuristic") or {}).get("test")
        print(f"Un modelo que nunca ve el estado acierta {100 * r06['blind']['tgem']:.1f} % en T-gem"
              + (f" y elegir la respuesta más larga, {100 * lh:.1f} %" if lh is not None else "")
              + ". Regla pre-registrada: la métrica primaria pasa a T-mine + T-gen2. Mira también la celda 5.4.")
"""),
    ("3.8 · F0.7 — Reproducción de la receta de CLM (G2)",
     "InfoNCE pre / pre→mid (HN sólo en el denominador, y HN + in-batch) / HN desde el inicio, 2 semillas cada uno. Los cabezales públicos de CLM se evalúan aparte como referencia **contaminada** (vieron estos pares). Con un subconjunto de datos, G2 es cualitativo.",
     """
#@title F0.7 · G2
if get_bundle() is not None:
    r07 = f0.f0_7_reproduction(study, bundle)
    print(json.dumps(r07["G2"], indent=1)); print("contaminados:", r07.get("clm_public_contaminated"))
    g2 = r07["G2"]
    if g2.get("ordering_ok") is False and None not in (g2.get("start_peak_val"), g2.get("mid_hnonly"), g2.get("pre")):
        print(f"El orden de CLM no se reprodujo en T-gem: HN desde el inicio (pico {g2['start_peak_val']:.3f}) vs "
              f"pre→mid ({max(g2['mid_hnonly'] or 0, g2['mid_inbatch'] or 0):.3f}) vs pre ({g2['pre']:.3f}). "
              "EXP-005 mostró que en T-gem lo decide el atajo; la enmienda A1 la vuelve a medir en T-mine (F0.7b, 3.8b).")
"""),
    ("3.8b · F0.7b — Receta de CLM frente a la actual (A1.5) · decide la receta de F1b",
     """Compara la receta **actual** (la del F1 piloto: batch 2,048/512, OneCycle) con la **real de CLM**, leída de sus checkpoints [VERIFIED]: batch 16,384, LR constante (6.53e-3 en pre, 1e-3 en mid), weight decay 0. La receta de CLM usa además un tope de épocas (pre 4, mid 8; una época de mid a batch 16,384 son sólo 18 pasos) y 20 evaluaciones por etapa [SUPUESTO: CLM no publica cuántas épocas usó].

**Condiciones.** Todas con **selección por val-mine** y semillas 0 y 1:

- **pre**: sólo pre-training.
- **mid_inbatch**: HN + in-batch, desde el pre de la misma semilla.
- **start_inbatch**: HN + in-batch desde el inicio, con el mismo número de muestras.
- **mid_hnonly**: HN solos. Es referencia de mecanismo y **no entra en la regla**.

**Regla (pre-especificada, ENMIENDA_03 §4).** F1b usa la receta de CLM si y sólo si, con ella, el top-1 en **T-mine** cumple mid_inbatch > start_inbatch > pre (media de las 2 semillas). Si no, usa la actual y la desviación queda documentada. La decisión queda en `config/RECETA_F1b.json` y no se sobrescribe.

Es parte de F0: no necesita la autorización de la 4.0, pero sí los datos de entrenamiento (3.3) y F0.5b.""",
     """
#@title F0.7b · Receta de CLM vs actual (A1.5)
if get_bundle() is not None:
    if bundle.val_mine_idx is None:
        print("Falta el pool de val-mine: corre F0.5b (celda 3.6b). No se hace nada.")
    else:
        r07b = f0.f0_7b_recipes(study, bundle)
        tabla07b, g2b = f07b_table(study)
        display(tabla07b)
        for rec, g in (g2b or {}).items():
            estado = {True: "SÍ se cumple", False: "NO se cumple", None: "pendiente"}[g.get("ordering_ok")]
            print(f"G2b · receta {rec}: T-mine medio {g.get('tmine_mean')} -> orden de CLM {estado}")
            for k, c in (g.get("contrasts_tmine") or {}).items():
                print(f"    {k}: {100 * c['diff']:+.1f} pp, IC95 [{100 * c['ci95_lo']:+.1f}, {100 * c['ci95_hi']:+.1f}]")
        dec = recipe_decision(study)
        print("\\nReceta de F1b (regla de A1.5):", dec["receta"] if dec else "sin decisión todavía (faltan corridas)")
"""),
    ("3.9 · F0.8 — Colapso de los brazos sin negativos (G4)",
     "Corridas cortas de los brazos 6–10, 12, 14 y 15, con y sin aumentación en espacio de embedding. El colapso se mide en el estado **final** (RankMe y top-1 en T-rand). **Criterio:** un colapso se reproduce en 2 semillas y se explica antes de seguir.",
     """
#@title F0.8 · G4
if get_bundle() is not None:
    r08 = f0.f0_8_collapse(study, bundle)
    display(pd.DataFrame(r08).T)
"""),
    ("3.10 · F0.9 — Arnés de T-Rex (G5) · requiere L4 o A100",
     "Cabezal público de CLM con y sin shield, 5 cursos × 60 s. Levanta en segundo plano un servidor de embeddings (Qwen3-8B) y `clm-serve` del repo de CLM (commit auditado). **Advertencia:** aun sin shield, T-Rex mide seguir etiquetas del planner; es métrica secundaria.",
     """
#@title F0.9 · G5
pip("transformers>=4.51", "accelerate", "fastapi", "uvicorn", "httpx", "requests")
r09 = f0.f0_9_trex(study)
print(json.dumps(r09, indent=1, ensure_ascii=False))
"""),
    ("3.11 · F0.10 — Identidad del set de DeepSWE (G6)",
     "`bon_eval.py` original con el cabezal público de DeepSWE sobre `tarsur385/deepswe-prm-embeddings-8k`. Si reproduce exactamente 31/38, se habilita la validación cruzada sobre 113 tareas; si no, DeepSWE queda exploratorio.",
     """
#@title F0.10 · G6
r10 = f0.f0_10_deepswe(study)
print(json.dumps(r10, indent=1, ensure_ascii=False))
"""),
    ("3.12 · Reporte de F0",
     "Tabla de compuertas a partir de lo guardado (`f0/F0_REPORT.md`). Una compuerta que no pasa se registra y se diagnostica; no se cambia el umbral.",
     """
#@title Reporte F0
rep0 = f0.f0_report(study)
display(Markdown(open(study.paths.d("f0", "F0_REPORT.md")).read()))
"""),
]

S4 = """
# 4 · F1b — un brazo por celda (exploratorio; enmienda A1)

Cada celda hace tres cosas:

- corre la búsqueda de hiperparámetros del brazo: 32 trials al 25 % del presupuesto; el trial 0 son los valores por defecto publicados;
- elige el mejor **por val-mine** (A1.2);
- entrena 3 semillas con el presupuesto completo y la **receta que decidió F0.7b**.

**Evaluación de cada semilla:**

- **T-mine** (primaria);
- T-gen (cada generador del panel y el macro);
- T-rand, T-gem y T-mine-50;
- T-abst-mine y T-abst de Gemini;
- estados barajados en todos;
- calibración y geometría.

**InfoNCE+HN** usa "HN + in-batch" por diseño (A1.7); "HN solos" queda para B1 (7.7).

**F1 piloto:** queda terminado y se conserva (4.P); sus resultados se reportan aparte.

El test de esta fase es exploratorio; el confirmatorio queda sellado hasta la Sección 8.

**Para entrenar F1b hacen falta tres cosas, y "Ejecutar todo" las deja listas solo:**

1. F0.5b (3.6b);
2. la decisión de receta de F0.7b (3.8b);
3. la autorización con el código actual, que la 4.0 registra sola (G2 y G3 fallaron en F0 y la enmienda A1, que aprobaste, es la respuesta).

Las celdas de brazos ya terminados sólo muestran su resumen.
"""

C40 = """
#@title 4.0 · Compuertas de F0, autorización para ENTRENAR y datos (automático)
#@markdown No hay nada que marcar. G2 y G3 no pasaron en F0 y la enmienda A1, que aprobaste el 2026-10-01, es la respuesta a ambas. Por eso la autorización para entrenar con este código queda registrada sola en `config/F0_DECISIONES.jsonl`, con el hash del código y citando esa decisión. Si fallara otra compuerta (G4), no se autoriza nada y la celda lo dice.
autorizado = False
if datos_listos():
    if PROFILE != "smoke":
        g = compuertas()
        print("Compuertas de F0:", {k: ("PASA" if v else "PENDIENTE" if v is None else "NO PASA") for k, v in g.items()})
        clave = {k.split()[0]: v for k, v in g.items()}
        if clave.get("G3") is False:
            r6 = load_json(study.paths.d("f0", "F0.6_shortcuts", "result.json")) or {}
            print(f"· G3: un modelo que no ve el estado acierta {100 * (r6.get('blind') or {}).get('tgem', float('nan')):.1f} % "
                  "en T-gem (azar 9.1 %). Regla pre-registrada: la métrica primaria pasa a T-mine (+ T-gen); F1b selecciona "
                  "con val-mine.")
        if clave.get("G2") is False:
            print("· G2: el orden de CLM no se reprodujo en T-gem (EXP-005: lo decide el atajo). F0.7b lo mide en T-mine "
                  f"(G2b) y decide la receta de F1b: {(recipe_decision(study) or {}).get('receta', 'pendiente')}.")
        if clave.get("G4") is False:
            print("· G4: todas las configuraciones sin negativos colapsaron en F0.8.")
        malas = [k for k in ("G2", "G3", "G4") if clave.get(k) is not True]
        if malas and not entrenamiento_autorizado(avisar=False):
            if set(malas) <= {"G2", "G3"}:
                append_jsonl(study.paths.d("config", "F0_DECISIONES.jsonl"),
                             {"fecha": now_iso(), "compuertas": g, "code_hash": CODE_HASH, "motivo": DECISION_A1,
                              "decidido_por": "investigador (conversación del proyecto, 2026-10-01 y 2026-10-02)",
                              "registrado_por": f"notebook {NB_REV}"})
                print(f"Autorización registrada en config/F0_DECISIONES.jsonl para el código {CODE_HASH}: {DECISION_A1}")
            else:
                print(f"No se autoriza entrenar: {malas} no pasaron y la enmienda A1 sólo responde a G2 y G3. "
                      "Hace falta una decisión tuya; avisa en la conversación del proyecto.")
    autorizado = entrenamiento_autorizado(avisar=False)
    f1b_listo(avisar=True)
    print("Receta de F1b:", (recipe_decision(study) or {}).get("receta", "pendiente (F0.7b, celda 3.8b)"),
          "· pool de val-mine:", "listo" if os.path.exists(study.paths.d("data", "eval", "val", "mine_idx.npy")) else "falta (3.6b)")
    if autorizado:
        r_dl = f0.f0_download(study)          # no baja nada si ya está en el disco local
        bundle = load_bundle(study)
        print(f"val={bundle.val.n} ítems · test={bundle.test.n} ítems · dim={bundle.hidden} · "
              f"filas pre={sum(int(m.sum()) for m in bundle.masks_pre.values())} · filas mid={sum(int(m.sum()) for m in bundle.masks_mid.values())}")
"""

C4PILOT = """
#@title 4.P · F1 piloto (terminado; v0.2, selección con T-gem-val): avance por brazo
display(progress(study, "F1"))
"""

C4P = """
#@title 4.16 · Avance de F1b
display(progress(study, "F1b"))
"""

C4B = """
#@title 4.17 · ¿Cuánto falta de F1b? Horas y unidades de cómputo, con tiempos MEDIDOS
CU_POR_HORA = 18.87     # el consumo por hora que reportaste para tu GPU (RTX PRO 6000)
b = estimate_budget(study, cu_per_hour=CU_POR_HORA, phase="F1b")
if "tabla" in b:
    print({k: v for k, v in b.items() if k != "tabla"}); display(b["tabla"])
else:
    print(b["nota"])
"""

C53 = """
#@title 5.3 · Control de tunability: ¿cambia el ranking con 8, 16 o 32 trials?
t = tunability_report(study, globals().get("FASE", "F1b"), ns=(1, 2, 4, 8, 16, 32))
if "evp" in t:
    display(t["evp"].round(4)); print("Kendall τ del ranking con n trials vs n =", t["n_referencia"], ":", t["kendall_tau_vs_ref"])
    display(Image(t["png"]))
else:
    print(t["nota"])
"""

C54 = """
#@title 5.4 · Atajos en F1 (B1, exploratorio): ¿cuánto depende cada brazo del estado?
#@markdown Para cada brazo: top-1 en T-gem completo, en los ítems donde el modelo ciego falla, en los ítems donde el oro **no** es la respuesta más larga, y con los **estados barajados** (cada candidato se puntúa con el estado de otro ítem). Si el acierto casi no baja al barajar, el brazo no está usando el estado. Exploratorio: no reemplaza a T-gen2. Tarda unos minutos; lo calculado se guarda junto a cada corrida.
INCLUIR_CLM_PUBLICO = True
listos = brazos_terminados("F1")
blind_dir = study.paths.d("f0", "F0.6_shortcuts")
if not listos or not os.path.exists(os.path.join(blind_dir, "blind.pt")):
    print("Hace falta F0.6 (modelo ciego) y al menos un brazo terminado de F1. No se hace nada.")
elif get_bundle() is not None:
    from clm_losses.config import train_config_from_dict
    from clm_losses.evaluate import project, rank_metrics
    from clm_losses.heads import Heads, heads_from_clm_checkpoint, l2n, load_torch
    from clm_losses.stats import clusters_from_docs, collect_runs, contrast
    from clm_losses.texts import length_heuristic
    from clm_losses.utils import atomic_np_save
    test, dev = bundle.test, ("cuda" if torch.cuda.is_available() else "cpu")
    out = makedirs(study.paths.d("results", "F1", "atajos"))
    Q, A = test.q_all(), test.a_all()
    perm = np.random.default_rng(0).permutation(test.n)          # misma permutación para todos los brazos

    @torch.no_grad()
    def puntuar(h, estados):
        filas = []
        for s0 in range(0, test.n, 2048):
            sel = np.arange(s0, min(test.n, s0 + 2048))
            zs = l2n(project(h.state_head, estados[sel], dev))
            c = np.concatenate([A[sel][:, None, :], test.negatives(sel)], 1)
            zc = l2n(project(h.action_head, c.reshape(-1, c.shape[-1]), dev)).view(len(sel), c.shape[1], -1)
            filas.append(torch.einsum("bd,bkd->bk", zs, zc).cpu().numpy())
        return np.concatenate(filas)

    def cargar(rd):
        cfg = train_config_from_dict(load_json(os.path.join(rd, "config.json"))["train"])
        h = Heads(hidden=bundle.hidden, width=cfg.width, depth=cfg.depth, proj=cfg.proj, activation=cfg.activation,
                  layernorm=cfg.layernorm, residual=cfg.residual, tau_init=cfg.tau_init).to(dev)
        etapa = "mid" if os.path.exists(os.path.join(rd, "best_mid.pt")) else "pre"
        h.load_state_dict(torch.load(os.path.join(rd, f"best_{etapa}.pt"), map_location=dev, weights_only=False)["heads"])
        return h.eval()

    blind = f0.train_blind(bundle, blind_dir, log=lambda *a, **k: None)
    ciego = f0.blind_top1(blind, test)
    subconj = {"T-gem completo": np.ones(test.n, bool), "ciego falla": ciego < 0.5}
    lp = study.paths.d("data", "eval", "test", "lengths.npz")
    largo = None
    if os.path.exists(lp):
        z = np.load(lp)
        largo = length_heuristic(z["gold"], z["neg"], "longest")
        subconj["oro no es el más largo"] = np.nan_to_num(largo, nan=1.0) == 0.0   # algún HN es estrictamente más largo
    print("ítems por subconjunto:", {k: int(m.sum()) for k, m in subconj.items()})

    items, barajado = {}, {}
    for arm, lst in collect_runs(study.paths.runs, "F1", listos).items():
        C, S = [], []
        for seed, rd in lst:
            C.append(np.load(os.path.join(rd, "eval", "correct_tgem.npy")))
            p = os.path.join(rd, "eval", "correct_tgem_barajado.npy")
            if not os.path.exists(p):
                atomic_np_save(rank_metrics(puntuar(cargar(rd), Q[perm]))[1], p)
            S.append(np.load(p))
        items[arm], barajado[arm] = np.vstack(C), np.vstack(S)
    if INCLUIR_CLM_PUBLICO:
        try:
            info = f0.f0_1_checkpoints(study)["heads"]["Contrastive-LM/CLM-v0.1-8B/CLM_v0.1-8B.pt"]
            if not os.path.exists(info["path"]):
                from huggingface_hub import hf_hub_download
                hf_hub_download("Contrastive-LM/CLM-v0.1-8B", "CLM_v0.1-8B.pt", local_dir=os.path.dirname(info["path"]))
            h = heads_from_clm_checkpoint(load_torch(info["path"]), dev).eval()
            items["CLM público (contaminado)"] = rank_metrics(puntuar(h, Q))[1][None]
            barajado["CLM público (contaminado)"] = rank_metrics(puntuar(h, Q[perm]))[1][None]
        except Exception as e:
            print("CLM público omitido:", e)

    filas = []
    for arm in items:
        C, S = items[arm], barajado[arm]
        fila = {"brazo": arm, "semillas": C.shape[0]}
        fila.update({k: round(float(C[:, m].mean()), 4) for k, m in subconj.items()})
        fila["estados barajados"] = round(float(S.mean()), 4)
        fila["uso del estado (pp)"] = round(100 * float(C.mean() - S.mean()), 1)
        filas.append(fila)
    ref = {"brazo": "modelo ciego (no ve el estado)", "semillas": 1}
    ref.update({k: round(float(ciego[m].mean()), 4) for k, m in subconj.items()})
    ref["estados barajados"] = ref["T-gem completo"]; ref["uso del estado (pp)"] = 0.0
    filas.append(ref)
    if largo is not None:
        ref2 = {"brazo": "elegir la respuesta más larga", "semillas": 1}
        ref2.update({k: round(float(np.nanmean(largo[m])), 4) for k, m in subconj.items()})
        ref2["estados barajados"] = ref2["T-gem completo"]; ref2["uso del estado (pp)"] = 0.0
        filas.append(ref2)
    tabla = pd.DataFrame(filas).sort_values("T-gem completo", ascending=False)
    tabla.to_csv(os.path.join(out, "atajos_por_brazo.csv"), index=False)
    display(tabla)

    # efecto de entrenar con HN, dentro y fuera del atajo (bootstrap pareado por documento)
    efectos = []
    for con_hn, sin_hn in (("infonce_hn", "infonce"), ("siglip_hn", "siglip")):
        if con_hn in items and sin_hn in items:
            for k, m in subconj.items():
                c = contrast(items[con_hn][:, m], items[sin_hn][:, m], clusters_from_docs(test.doc_ids[m]),
                             study.rope_pp / 100, B=2000)
                efectos.append({"contraste": f"{con_hn} − {sin_hn}", "subconjunto": k, "ítems": int(m.sum()),
                                "Δ (pp)": round(100 * c["diff"], 1), "IC95 (pp)": f"[{100 * c['ci95_lo']:.1f}, {100 * c['ci95_hi']:.1f}]",
                                "p": c["p"]})
    if efectos:
        display(pd.DataFrame(efectos))
    save_json({"subconjuntos": {k: int(m.sum()) for k, m in subconj.items()}, "tabla": filas, "efectos_hn": efectos,
               "permutacion_semilla": 0, "nota": "exploratorio; el modelo ciego se entrenó con HN de Gemini del split de entrenamiento"},
              os.path.join(out, "atajos.json"))
    print("Guardado en", out)
"""

C55 = """
#@title 5.5 · Atajos según cómo entran los HN al entrenamiento (F0.7 y brazos de F1 con HN; exploratorio)
INCLUIR_F1 = True
#@markdown Las corridas de F0.7 comparten el pre-training, los hiperparámetros por defecto y el presupuesto; sólo cambia cómo entran los HN de Gemini al mid-training: **pre** (sin HN), **HN + in-batch** (el oro compite contra los HN y contra las respuestas de los otros ítems del batch), **HN solos** (sólo contra los HN: la fórmula del README de CLM, y la que eligió el HPO de F1) y **HN desde el inicio**. Para cada una: top-1 en T-gem completo, donde el ciego falla y donde el oro no es el más largo; con los estados barajados (misma permutación que la 5.4); y cuántas veces elige **el candidato más largo**, con los estados reales y con los barajados. Sin preferencia por la longitud, con los estados barajados eso daría ≈ 1/11 = 9 %. El orden de las longitudes coincide con el de los candidatos (verificado en F0.3: `neg_order`). Tarda unos minutos; lo calculado se guarda junto a cada corrida.
blind_dir = study.paths.d("f0", "F0.6_shortcuts")
lp = study.paths.d("data", "eval", "test", "lengths.npz")


def _semillas(base):
    if not os.path.isdir(base):
        return []
    return [os.path.join(base, d) for d in sorted(os.listdir(base))
            if d.startswith("seed_") and (load_json(os.path.join(base, d, "DONE.json")) or {}).get("status") == "ok"]


grupos = {}
for etiqueta, carpeta in (("F0.7 · pre (sin HN)", "infonce_pre"), ("F0.7 · HN + in-batch", "infonce_hn_inbatch"),
                          ("F0.7 · HN solos", "infonce_hn_hnonly"), ("F0.7 · HN desde el inicio", "infonce_hn_start")):
    rds = _semillas(study.paths.d("runs", "F0.7", carpeta))
    if rds:
        grupos[etiqueta] = rds
if INCLUIR_F1:
    for arm in ("infonce", "infonce_hn", "siglip_hn", "triplet_hn"):
        rds = _semillas(study.paths.d("runs", "F1", arm))
        if rds:
            grupos[f"F1 · {arm}"] = rds
if not any(k.startswith("F0.7") for k in grupos) or not os.path.exists(os.path.join(blind_dir, "blind.pt")):
    print("Hace falta F0.6 (modelo ciego) y las corridas de F0.7. No se hace nada.")
elif get_bundle() is not None:
    from clm_losses.config import train_config_from_dict
    from clm_losses.evaluate import project, rank_metrics
    from clm_losses.heads import Heads, l2n
    from clm_losses.stats import clusters_from_docs, contrast
    from clm_losses.texts import length_heuristic
    from clm_losses.utils import atomic_np_save
    test, dev = bundle.test, ("cuda" if torch.cuda.is_available() else "cpu")
    out = makedirs(study.paths.d("results", "F0", "atajos_por_denominador"))
    Q, A = test.q_all(), test.a_all()
    perm = np.random.default_rng(0).permutation(test.n)          # la misma de la 5.4

    @torch.no_grad()
    def _puntuar(h, estados):
        filas = []
        for s0 in range(0, test.n, 2048):
            sel = np.arange(s0, min(test.n, s0 + 2048))
            zs = l2n(project(h.state_head, estados[sel], dev))
            c = np.concatenate([A[sel][:, None, :], test.negatives(sel)], 1)
            zc = l2n(project(h.action_head, c.reshape(-1, c.shape[-1]), dev)).view(len(sel), c.shape[1], -1)
            filas.append(torch.einsum("bd,bkd->bk", zs, zc).cpu().numpy())
        return np.concatenate(filas)

    def _cargar(rd):
        cfg = train_config_from_dict(load_json(os.path.join(rd, "config.json"))["train"])
        h = Heads(hidden=bundle.hidden, width=cfg.width, depth=cfg.depth, proj=cfg.proj, activation=cfg.activation,
                  layernorm=cfg.layernorm, residual=cfg.residual, tau_init=cfg.tau_init).to(dev)
        etapa = "mid" if os.path.exists(os.path.join(rd, "best_mid.pt")) else "pre"
        h.load_state_dict(torch.load(os.path.join(rd, f"best_{etapa}.pt"), map_location=dev, weights_only=False)["heads"])
        return h.eval()

    blind = f0.train_blind(bundle, blind_dir, log=lambda *a, **k: None)
    ciego = f0.blind_top1(blind, test)
    subconj = {"T-gem completo": np.ones(test.n, bool), "ciego falla": ciego < 0.5}
    mas_largo = None
    if os.path.exists(lp):
        z = np.load(lp)
        L = np.concatenate([z["gold"][:, None], z["neg"]], 1).astype(np.int64)
        unico = (L >= 0).all(1) & ((L == L.max(1, keepdims=True)).sum(1) == 1)
        mas_largo = np.where(unico, L.argmax(1), -1)            # -1: ítem sin longitudes o con empate en el máximo
        largo = length_heuristic(z["gold"], z["neg"], "longest")
        subconj["oro no es el más largo"] = np.nan_to_num(largo, nan=1.0) == 0.0
    print("ítems por subconjunto:", {k: int(m.sum()) for k, m in subconj.items()},
          "| ítems con un único candidato más largo:", None if mas_largo is None else int((mas_largo >= 0).sum()))

    def _elige_largo(argmax):
        ok = mas_largo >= 0
        return float((argmax[ok] == mas_largo[ok]).mean())

    por_item, filas = {}, []
    for g, rds in grupos.items():
        C, S, Tm, Tr, real, bar, acuerdo, tr, tm = [], [], [], [], [], [], [], [], []
        for rd in rds:
            ev = os.path.join(rd, "eval")
            p_r, p_b = os.path.join(ev, "argmax_tgem.npy"), os.path.join(ev, "argmax_tgem_barajado.npy")
            p_c = os.path.join(ev, "correct_tgem_barajado.npy")
            if not all(os.path.exists(p) for p in (p_r, p_b, p_c)):
                h = _cargar(rd)
                s_r, s_b = _puntuar(h, Q), _puntuar(h, Q[perm])
                atomic_np_save(s_r.argmax(1).astype(np.int8), p_r)
                atomic_np_save(s_b.argmax(1).astype(np.int8), p_b)
                if not os.path.exists(p_c):
                    atomic_np_save(rank_metrics(s_b)[1], p_c)
                del h
            c_guardado = np.load(os.path.join(ev, "correct_tgem.npy"))
            a_r = np.load(p_r)
            acuerdo.append(float(((a_r == 0) == (c_guardado >= 0.5)).mean()))
            C.append(c_guardado)
            S.append(np.load(p_c))
            for nombre, lista in (("tmine", Tm), ("trand", Tr)):
                p = os.path.join(ev, f"correct_{nombre}.npy")
                if os.path.exists(p):
                    lista.append(np.load(p))
            if mas_largo is not None:
                real.append(_elige_largo(a_r)); bar.append(_elige_largo(np.load(p_b)))
            t = (load_json(os.path.join(rd, "DONE.json")) or {}).get("test", {})
            tr.append((t.get("trand") or {}).get("top1", np.nan)); tm.append((t.get("tmine") or {}).get("top1", np.nan))
        C, S = np.vstack(C), np.vstack(S)
        por_item[g] = {"T-gem": C, "barajados": S}
        if len(Tm) == len(rds): por_item[g]["T-mine"] = np.vstack(Tm)
        if len(Tr) == len(rds): por_item[g]["T-rand"] = np.vstack(Tr)
        fila = {"grupo": g, "semillas": len(rds)}
        fila.update({k: round(float(C[:, m].mean()), 4) for k, m in subconj.items()})
        fila["estados barajados"] = round(float(S.mean()), 4)
        fila["uso del estado (pp)"] = round(100 * float(C.mean() - S.mean()), 1)
        if mas_largo is not None:
            fila["elige el más largo"] = round(float(np.mean(real)), 4)
            fila["elige el más largo (barajados)"] = round(float(np.mean(bar)), 4)
        fila["T-rand"], fila["T-mine"] = round(float(np.nanmean(tr)), 4), round(float(np.nanmean(tm)), 4)
        fila["acuerdo con la evaluación guardada"] = round(float(np.mean(acuerdo)), 4)
        filas.append(fila)
    ref = {"grupo": "modelo ciego (no ve el estado)", "semillas": 1}
    ref.update({k: round(float(ciego[m].mean()), 4) for k, m in subconj.items()})
    ref["estados barajados"], ref["uso del estado (pp)"] = ref["T-gem completo"], 0.0
    if mas_largo is not None:
        from clm_losses.f0 import _cands
        with torch.no_grad():
            s_c = np.concatenate([blind(torch.from_numpy(_cands(q, a, n)).float().to(dev)).cpu().numpy()
                                  for _, q, a, n in test.iter_batches(1024)])
        ref["elige el más largo"] = ref["elige el más largo (barajados)"] = round(_elige_largo(s_c.argmax(1)), 4)
    filas.append(ref)
    tabla = pd.DataFrame(filas)
    tabla.to_csv(os.path.join(out, "atajos_por_denominador.csv"), index=False)
    display(tabla)
    bajo = [f["grupo"] for f in filas if f.get("acuerdo con la evaluación guardada", 1.0) < 0.99]
    if bajo:
        print("OJO: la re-evaluación no coincide con la guardada en", bajo, "(revisar antes de interpretar).")

    efectos = []
    cl = clusters_from_docs(test.doc_ids)
    for g_a, g_b in (("F0.7 · HN solos", "F0.7 · HN + in-batch"), ("F0.7 · HN + in-batch", "F0.7 · pre (sin HN)"),
                     ("F0.7 · HN solos", "F0.7 · pre (sin HN)"), ("F1 · infonce_hn", "F1 · infonce")):
        if g_a in por_item and g_b in por_item:
            for met in ("T-gem", "barajados", "T-mine", "T-rand"):
                if met in por_item[g_a] and met in por_item[g_b]:
                    c = contrast(por_item[g_a][met], por_item[g_b][met], cl, study.rope_pp / 100, B=2000)
                    efectos.append({"contraste": f"{g_a} − {g_b}", "métrica": met, "Δ (pp)": round(100 * c["diff"], 1),
                                    "IC95 (pp)": f"[{100 * c['ci95_lo']:.1f}, {100 * c['ci95_hi']:.1f}]", "p": c["p"]})
    if efectos:
        display(pd.DataFrame(efectos))
    save_json({"subconjuntos": {k: int(m.sum()) for k, m in subconj.items()}, "tabla": filas, "efectos": efectos,
               "permutacion_semilla": 0, "nota": "exploratorio; 'barajados' = cada candidato se puntúa con el estado de otro ítem"},
              os.path.join(out, "atajos_por_denominador.json"))
    print("Guardado en", out)
"""

S5 = """
# 5 · Todos contra todos (F1b; también el F1 piloto)

**Para cada métrica:**

- las 105 diferencias por pares, con bootstrap jerárquico pareado (documentos; semillas dentro de cada brazo);
- p-valores con corrección de Holm;
- equivalencia por TOST (±1 pp);
- probabilidades de que A sea mejor, equivalente o peor.

**Además:** el ranking multi-tarea (Friedman + Nemenyi, un bloque por tarea) y **C1**. En F1b, C1 va en T-mine y en el macro de T-gen, con Holm entre las dos.

**Orden de métricas en F1b:** T-mine (primaria) → macro de T-gen → T-rand → T-gem → T-gem barajado → T-mine-50. Abstención: T-abst-mine (primaria) y T-abst de Gemini.

Se puede correr con los brazos que ya terminaron; lo que falte se omite. Las celdas 5.4 y 5.5 son los diagnósticos de atajos del F1 piloto y de F0.7 (rev. 3–3.1); la 5.6 es su versión para cualquier fase, con los estados barajados que ahora trae cada evaluación.
"""

C51 = """
#@title 5.1 · Análisis todos contra todos
FASE = "F1b"
#@markdown F1b = el estudio con la enmienda A1 (métrica primaria T-mine + macro de T-gen). F1 = el piloto (v0.2).
listos = brazos_terminados(FASE)
if len(listos) < 2:
    print(f"Hacen falta al menos 2 brazos terminados en {FASE} (hay {len(listos)}: {listos}). No se hace nada.")
elif get_bundle() is not None:
    rep = run_all_vs_all(study, bundle, FASE)
    out = study.paths.d("results", FASE, "all_vs_all")
    for name, m in rep["metrics"].items():
        if "summary" in m:
            print(f"{name}: {m['n_significant']}/{m['n_pairs']} pares significativos (Holm), {m['n_equivalent']} equivalentes")
    if FASE == "F1":
        print("\\nF1 piloto: la selección usó T-gem-val (tiene el atajo, EXP-001/002). Métrica primaria pre-registrada: T-mine.")
    print("Panel T-gen usado en el macro:", rep.get("panel_version") or "todavía no existe (Sección 6)")
    orden = [f"{t}_top1" for t in default_testsets(FASE)]
    for name in orden:
        f = os.path.join(out, f"summary_{name}.csv")
        if os.path.exists(f):
            print(name); display(pd.read_csv(f))
    if "C1" in rep:
        c = rep["C1"]
        for met, cc in (c.get("por_metrica") or {c.get("metrica", "tgem"): c}).items():
            ph = f" p_Holm(C1)={cc['p_holm_c1']:.3g}" if "p_holm_c1" in cc and FASE != "F1" else ""
            print(f"C1 en {met}: {c['A']} − infonce = {cc['diff']:+.4f} IC95 [{cc['ci95_lo']:+.4f}, {cc['ci95_hi']:+.4f}] "
                  f"p={cc['p']:.3g}{ph} -> {cc['verdict']}")
        print(f"(brazo no contrastivo elegido por validación en {c.get('selection_phase', FASE)}; exploratorio hasta F3)")
    for f in ("heatmap_tmine_top1.png", "heatmap_tgenmacro_top1.png", "heatmap_tgem_top1.png", "heatmap_abst_auroc_tmine.png",
              "heatmap_abst_auroc.png", "cd_diagram.png"):
        if os.path.exists(os.path.join(out, f)):
            display(Image(os.path.join(out, f)))
"""

C52 = """
#@title 5.2 · Resumen en Markdown (para el EXPERIMENT_LOG del proyecto)
fase = globals().get("FASE", "F1b")
if os.path.exists(study.paths.d("results", fase, "all_vs_all", "all_vs_all.json")):
    p = export_summary_md(study, fase); print(p); display(Markdown(open(p).read()))
else:
    print(f"Todavía no hay análisis de {fase} (celda 5.1). No se hace nada.")
"""

C56 = """
#@title 5.6 · Uso del estado por brazo (A1.4): estados reales vs barajados en T-mine, T-gem y T-gen
FASE_ESTADO = "F1b"
#@markdown "Uso del estado" = acierto con los estados reales − acierto con los estados barajados (la misma permutación de 5.4/5.5). Si casi no cambia al barajar, el brazo acierta sin mirar el estado. En F1 (piloto) hace falta antes la 6.5 (agrega estas evaluaciones a lo ya entrenado).
t = state_use_table(study, FASE_ESTADO)
if len(t):
    display(t)
    t.to_csv(makedirs(study.paths.d("results", FASE_ESTADO, "uso_del_estado")) + "/uso_del_estado.csv", index=False)
else:
    print(f"Todavía no hay corridas terminadas de {FASE_ESTADO}. No se hace nada.")
"""

def panel_md() -> str:
    from clm_losses.generators import PANEL, ROLE_TXT
    rows = ["| Clave | Modelo | Familia | Rol | Tamaño | Acceso | Licencia | Parámetros |",
            "|---|---|---|---|---|---|---|---|"]
    for g in PANEL:
        rows.append(f"| `{g.key}` | {g.repo} | {g.family} | {ROLE_TXT[g.role]} | {g.tier or 'MoE'} | "
                    f"{'**gated**' if g.gated else 'abierto'} | "
                    f"{g.license} | {g.params_b:g}B{' · ' + str(g.n_items) + ' ítems' if g.n_items else ''} |")
    return "\n".join(rows)


S6 = """
# 6 · Panel de generadores T-gen (ENMIENDA_03 §3, §10 y §11) — automático

**Por qué un panel.** El oro (High-Quality-DQA) lo escribió **Qwen3-30B-A3B** [VERIFIED, tarjeta de Nemotron-CC-v2.1] y los HN de T-gem, Gemini. T-gen mide si lo aprendido vale con respuestas incorrectas escritas por **otros** modelos: 23 generadores de 12 familias, 16 de 3–9B y 7 de 14–34B, cada uno con su propia auditoría.

**Qué pasó con la v1 (EXP-006a, 2026-10-02).** En los 16 generadores, un modelo que no ve la pregunta encontraba el oro el 77–91 % de las veces (azar 9 %), así que el macro quedó vacío. Causas encontradas en los textos:

- el oro suele terminar en punto y los negativos no;
- hay ítems de sí/no y de opción múltiple, que no admiten 10 respuestas incorrectas parecidas;
- el filtro de coseno > 0.95 sacaba justo las respuestas que imitaban el formato del oro;
- el oro es más largo y más específico que lo que un modelo inventa. Pasa incluso con Qwen3-30B-A3B, el mismo modelo que escribió el oro.

**Tres construcciones** (las mismas reglas para todos los generadores; ninguna mira un modelo entrenado):

| | Positivo | Negativos | Qué corrige |
|---|---|---|---|
| **S** | el oro original | las respuestas ya generadas, con el formato del oro | puntuación, tipos de ítem, falsos negativos ("6 pages" con oro "6") |
| **P** | el oro **reescrito por el mismo generador**, con un largo asignado al azar | las mismas de S, en formato canónico | además el estilo y la longitud (extensión pre-registrada en §3.6) |
| **E** | el oro original | **ediciones** del oro: cambian sólo el dato clave | además la longitud, el estilo y la especificidad |

**Piloto (6.0), regla fija:** las tres construcciones se arman con los mismos 500 ítems y los mismos 3 generadores, y se auditan con las sondas ciegas. Gana la de **menor mediana del atajo ciego**; un empate (< 1 pp) lo gana la más barata (S, luego P, luego E). Después, 6.1–6.3 arman el panel completo con la ganadora.

**Pasos:**

| Celda | Qué hace |
|---|---|
| **6.0** | Piloto y elección de la construcción (`data/eval/tgen/BUILD.json`) |
| **6.1** | Genera lo que pide la construcción en los 23 generadores; un subproceso por generador; borra los pesos al terminar |
| **6.2** | Construye y embebe los sets con Qwen3-8B, después de verificar G1 en esta sesión |
| **6.3** | Auditoría ciega y **PANEL.json** con el macro, por regla fija; aplica sola las exclusiones del QC manual si hay marcas |
| **6.4** | CSV para revisar falsos negativos a mano (opcional) |
| **6.5** | Agrega T-gen (y los controles de A1.4) a todo lo ya entrenado, sin re-entrenar |
| **6.6** | **EXP-006**: ¿la ganancia por HN se transfiere a otros generadores? |
| **6.7** | F1b otra vez, ya con T-gen: C1 en T-mine y en T-gen (con Holm) y resumen |

**Regla del macro (sin cambios):** entran los primarios con la generación terminada, cobertura ≥ 80 % de los ítems elegibles, sin atajo alto (< 30 %, bandas de G3) y sin exclusión por el QC manual; se usa la intersección de ítems. Si quedan menos de 5, se reportan además los respaldos de §3.6: todos los primarios ("con atajo") y los ítems donde el ciego propio falla.

**Gated (4):** Llama 3.1 8B, Aya Expanse 8B y 32B, y Gemma 2. Usan `HF_TOKEN` de los secretos de Colab; sin acceso se saltan solos.

**Tamaño:** "chico" = 2.6–9.4B densos; "grande" = 14–34B densos; los MoE (gpt-oss-20b, Qwen3-30B-A3B) no entran en el contraste por tamaño.

Licencias "other" / no comerciales: uso de investigación **[AUTHOR DECISION NEEDED antes de publicar los negativos]**.

{PANEL_TABLE}
"""

C60 = """
#@title 6.0 · Piloto del panel: ¿con qué construcción el set no delata al correcto? (automático, una vez)
#@markdown Arma S, P y E con los mismos 500 ítems y 3 generadores, las audita con las sondas ciegas y fija la de menor atajo (regla fija, ENMIENDA_03 §11). Lo que se genera aquí se reutiliza en 6.1. Si ya está decidido, sólo muestra el resultado.
rec = load_json(tgen.build_path(study))
liberar_gpu()                      # los generadores corren en otro proceso y necesitan la GPU libre
if get_bundle() is not None:
    if not (rec and rec.get("build")):
        blind_dir = study.paths.d("f0", "F0.6_shortcuts")
        blind = (f0.train_blind(bundle, blind_dir, log=lambda *a, **k: None)
                 if os.path.exists(os.path.join(blind_dir, "blind.pt")) else None)
        if PROFILE == "smoke":
            from clm_losses.smoke import synthetic_lengths
            synthetic_lengths(study, "test")
            rec = tgen.synthetic_pilot(study, bundle, blind=blind)
        elif entorno_panel_ok():
            try:
                rec = tgen.run_pilot(study, bundle, encoder_verificado, blind=blind)
            except Exception as e:           # no detiene "Ejecutar todo": lo demás sigue y el piloto se reanuda
                print(f"El piloto no terminó ({type(e).__name__}: {e}). Vuelve a Ejecutar todo: retoma donde quedó.")
            finally:
                liberar_gpu()
    if rec and rec.get("build"):
        print(f"Construcción del panel: {rec['build']} — {rec['build_txt']} (decidida el {rec['decided']})")
        print("Mediana del atajo ciego por construcción (menor es mejor; azar 0.091):",
              {b: round(v, 3) for b, v in (rec.get("medianas") or {}).items()})
        display(pd.DataFrame(rec["pilot"]["table"]).round(3))
"""

C61 = """
#@title 6.1 · Generar con los 23 generadores lo que pide la construcción (reanudable; uno a la vez)
#@markdown Primero los 16 originales y después los 7 grandes. Lo ya generado (incluido el piloto) se salta; los gated sin acceso se saltan solos; los pesos de cada modelo se borran al terminar.
liberar_gpu()
if get_bundle() is not None:
    if not os.path.exists(tgen.build_path(study)):
        print("Falta la construcción del panel (6.0). No se genera nada.")
    elif PROFILE == "smoke":
        print("Perfil smoke: panel SINTÉTICO (sin LLM, sólo prueba el código):", tgen.synthetic_panel(study, bundle.test))
    else:
        build = tgen.active_build(study)
        print(f"Construcción {build}: tareas {tgen.BUILDS[build]} en {len(tgen.PANEL)} generadores")
        estado = []
        for k in tgen.resolve_keys("todos"):
            try:
                st = tgen.run_tasks(study, bundle.test, k, tgen.BUILDS[build], n_items=tgen.PANEL_N)
            except Exception as e:                 # un generador que falla no detiene a los demás
                st = {"error": f"{type(e).__name__}: {e}"}
            estado.append({"generador": k, **st})
        display(pd.DataFrame(estado))
"""

C62 = """
#@title 6.2 · Construir y embeber los sets (Qwen3-8B, pipeline de G1 verificado en esta sesión)
if get_bundle() is not None and os.path.exists(tgen.build_path(study)) and PROFILE != "smoke":
    claves = [k for k in tgen.PANEL_BY_KEY if os.path.isdir(tgen.gen_dir(study, k))]
    pend = [k for k in claves if tgen.needs_embedding(study, k, n_items=tgen.PANEL_N)]
    sin_terminar = {k: tgen.build_gen_info(study, k)["tasks"] for k in claves
                    if tgen.build_gen_info(study, k)["status"] not in ("generado", "sin_acceso", "sin_soporte")}
    if sin_terminar:
        print("Generación sin terminar (se completa al volver a correr 6.1):", sin_terminar)
    if not pend:
        print("No hay sets nuevos para embeber.")
    elif entorno_panel_ok():
        try:
            enc = encoder_verificado()
            for k in pend:
                try:
                    tgen.embed_generator(study, bundle.test, k, enc, n_items=tgen.PANEL_N)
                except Exception as e:
                    print(f"{k}: no se pudo construir ({type(e).__name__}: {e})")
        except RuntimeError as e:
            print(e)
        finally:
            enc = None; liberar_gpu()
"""

C63 = """
#@title 6.3 · Auditar el panel (sondas ciegas) y fijar el macro de T-gen (regla fija)
blind_dir = study.paths.d("f0", "F0.6_shortcuts")
if get_bundle() is not None and os.path.isdir(bundle.tgen_dir):
    claves = sorted(k for k in os.listdir(bundle.tgen_dir) if tgen.load_gen(study, k) is not None)
    if not claves:
        print("Todavía no hay sets construidos (6.0 → 6.2). No se hace nada.")
    else:
        blind = (f0.train_blind(bundle, blind_dir, log=lambda *a, **k: None)
                 if os.path.exists(os.path.join(blind_dir, "blind.pt")) else None)
        for k in claves:
            tgen.audit_generator(study, bundle, k, blind=blind)
        mat = tgen.blind_transfer_matrix(study, bundle, claves, blind=blind)
        excl = tgen.qc_exclusions(study)
        if excl:
            print("Exclusiones por el QC manual (leídas de las marcas del CSV de la 6.4):", excl)
        panel = tgen.freeze_panel(study, exclude_qc=excl)
        display(tgen.panel_table(study))
        print(f"Construcción {panel['build']} · macro de T-gen (versión {panel['version']}): {panel['macro']} · "
              f"{panel['n_macro_items']} ítems en la intersección"
              + ("" if panel["robusto"] else f" · ATENCIÓN: menos de {panel['min_macro']} generadores (ENMIENDA_03 §3.6)"))
        print("Transferencia entre ciegos (fila = con qué set se entrenó; columna = dónde se evalúa; pliegues por ítem "
              "comunes: nunca se evalúa en un ítem cuyo oro se vio al entrenar; azar 0.091):")
        display(mat.round(3))
"""

C64 = """
#@title 6.4 · CSV para revisar falsos negativos a mano (opcional)
if get_bundle() is not None and os.path.isdir(bundle.tgen_dir):
    p = tgen.export_qc(study)
    print(p or "Todavía no hay sets construidos (6.2). No se hace nada.")
    if p:
        print("Revisión opcional. En la columna es_correcta(0/1): 1 si un 'negativo' en realidad responde bien la "
              "pregunta; en las filas 'positivo' (construcción P), 0 si la reescritura ya no es correcta. Con ≥ 50 filas "
              "marcadas de un generador, la 6.3 aplica sola el criterio de DISENO_01 (> 5 % sale del macro) la próxima "
              "vez que corra. Un CSV con marcas nunca se sobrescribe.")
"""

C65 = """
#@title 6.5 · Agregar T-gen y los controles nuevos (A1.4) a TODO lo ya entrenado (sin re-entrenar)
#@markdown F0.7, F1 piloto, F0.7b, F1b y F2b: carga el mejor checkpoint y evalúa sólo lo que falta (estados barajados, T-abst-mine, T-mine-50 y cada generador del panel). No toca lo ya evaluado.
if get_bundle() is not None:
    n = add_missing_evals(study, bundle, ["F0.7", "F1", "F0.7b", "F1b", "F2b_B1", "F2b_B2", "F2b_B3"])
    print("corridas actualizadas:", n)
"""

C66 = """
#@title 6.6 · EXP-006 · ¿Se transfiere la ganancia de los HN a otros generadores?
#@markdown Por condición: T-gem, T-mine, el macro de T-gen y cada generador, con estados reales y barajados. Contrastes pareados por documento. Lectura fijada de antemano (ENMIENDA_03 §6): **transfiere** si Δ(macro) > +1 pp y su IC95 excluye 0; **empeora** si Δ < −1 pp y su IC95 excluye 0; **no transfiere** si es equivalente por TOST ±1 pp; si no, no concluyente. Si el macro queda vacío, se leen los respaldos de §3.6: todos los primarios ("con atajo") y los ítems donde el ciego propio falla. Secundario (§10): el macro de grandes y el de chicos, y su interacción.
panel = tgen.load_panel(study)
if not panel:
    print("Todavía no hay panel (6.0 → 6.3). No se hace nada.")
elif get_bundle() is not None:
    if not panel.get("macro"):
        print("El macro de T-gen quedó vacío: ningún primario pasó la regla. Se reportan los respaldos de §3.6.")
    tr = transfer_report(study, bundle)
    display(tr["tabla"])
    c = tr["contrastes"]
    if len(c):
        principal = ["T-gem", "T-gem barajado", "T-mine", "T-gen (macro)", "T-gen (macro) barajado"]
        respaldo = c["métrica"].str.contains("respaldo|todos los primarios|ciego falla", regex=True)
        tam = c["métrica"].str.contains("grandes|chicos", regex=True)
        display(c[c["métrica"].isin(principal)])
        if respaldo.any():
            print("Respaldos de §3.6:"); display(c[respaldo])
        if tam.any():
            print("Secundario, por tamaño del generador:"); display(c[tam])
        print("Por generador:"); display(c[~c["métrica"].isin(principal) & ~tam & ~respaldo])
    print("Guardado en", tr["dir"])
"""

C66B = """
#@title 6.6b · EXP-006c · ¿La brecha entre T-gem y T-gen es del generador o del tipo de negativo?
#@markdown EXP-006 compara negativos **libres de Gemini** (T-gem) con negativos **editados** de otros generadores (macro de T-gen), así que la "fracción transferida" mezcla dos cosas: el generador y la construcción. Los sets del piloto (6.0) tienen los mismos 3 generadores con las tres construcciones; con ellos se separan **sin generar ni entrenar** (sólo se evalúan corridas que ya existen). En los ítems comunes: **generador** = Δ(T-gem) − Δ(S); **construcción** = Δ(S) − Δ(E); **oro reescrito** = Δ(P) − Δ(S). Lectura fijada antes de mirar (ENMIENDA_03 §12.3): un componente **explica parte de la brecha** si es > +1 pp con IC95 > 0; **no la explica** si es equivalente por TOST ±1 pp; si no, no concluyente.
if get_bundle() is not None and os.path.exists(tgen.build_path(study)):
    cr = construction_report(study, bundle)
    if len(cr["tabla"]):
        print(f"Generadores del piloto: {cr['keys']} · {cr['n_items']} ítems comunes a los 9 sets (azar 0.091)")
        display(cr["tabla"])
        c = cr["contrastes"]
        comp = c["cantidad"].str.contains("generador|construcción|oro reescrito|brecha|parte", regex=True)
        print("Descomposición de la brecha (lo que decide la lectura):"); display(c[comp])
        print("Ganancia por HN en cada set:"); display(c[~comp])
    print("Guardado en", cr["dir"])
"""

C67 = """
#@title 6.7 · F1b otra vez, ahora con T-gen: todos contra todos, C1 con su co-primaria y resumen
#@markdown La 5.1 corrió antes de que existiera el panel. Aquí se repite con el macro de T-gen ya agregado a cada corrida (6.5): C1 en T-mine y en T-gen, con Holm entre las dos.
if tgen.load_panel(study) and len(brazos_terminados("F1b")) >= 2 and get_bundle() is not None:
    rep = run_all_vs_all(study, bundle, "F1b")
    print("Panel T-gen:", rep.get("panel_version"), "· construcción", (tgen.load_panel(study) or {}).get("build"))
    if "C1" in rep:
        c = rep["C1"]
        for met, cc in (c.get("por_metrica") or {c.get("metrica"): c}).items():
            ph = f" p_Holm(C1)={cc['p_holm_c1']:.3g}" if "p_holm_c1" in cc else ""
            print(f"C1 en {met}: {c['A']} − infonce = {cc['diff']:+.4f} IC95 [{cc['ci95_lo']:+.4f}, {cc['ci95_hi']:+.4f}] "
                  f"p={cc['p']:.3g}{ph} -> {cc['verdict']}")
    out = study.paths.d("results", "F1b", "all_vs_all")
    for name in ("tmine_top1", "tgenmacro_top1"):
        f = os.path.join(out, f"summary_{name}.csv")
        if os.path.exists(f):
            print(name); display(pd.read_csv(f))
    print(export_summary_md(study, "F1b"))
    display(state_use_table(study, "F1b"))
else:
    print("Hacen falta el panel (6.3) y al menos 2 brazos de F1b. No se hace nada.")
"""

S7 = """
# 7 · F2b — experimentos focalizados (sobre F1b)

Usan los hiperparámetros elegidos en **F1b** para cada brazo, sin re-tuning. Heredan también la enmienda A1: receta de F0.7b, selección por val-mine y "HN + in-batch". Se evalúan en el test exploratorio, con T-mine y el macro de T-gen primero.

**Experimentos:**

- **B3** (C4/H4) está definido de antemano: InfoNCE, InfoNCE+SIGReg e InfoNCE+LSE con HN desde el inicio, todos con "HN + in-batch" (A1.7).
- **B1, régimen R2** usa los 2 brazos **sin HN** con mayor val-mine en F1b, excluidos los colapsados (≤ 2 pp sobre el azar). Esta regla quedó fijada el 2026-10-02, antes de correr F1b; la elección se guarda en `results/F2b/B1/TOP_ARMS.json`.
- **B1, "HN solos"** (7.7) es la condición explícita de mecanismo de A1.7.

Cada variante guarda sus semillas en `runs/F2b_*/<variante>/seed_N` y se reanuda sola. **Requieren autorización (4.0) y F1b del brazo base.**
"""

B3_CELLS = [
    ("7.1 · B3 — InfoNCE + HN desde el inicio (HN + in-batch)", "infonce_hn",
     'r = run_variant(study, bundle, "infonce_hn", "F2b_B3", "infonce_hn_start", hp_from="F1b", hn_from_start=True)'),
    ("7.2 · B3 — InfoNCE + SIGReg + HN desde el inicio", "infonce_sigreg",
     'r = run_variant(study, bundle, "infonce_sigreg", "F2b_B3", "infonce_sigreg_hn_start", hp_from="F1b", hn_from_start=True, hn_mode="native")'),
    ("7.3 · B3 — InfoNCE + logsumexp + HN desde el inicio (control simple)", "infonce_lse",
     'r = run_variant(study, bundle, "infonce_lse", "F2b_B3", "infonce_lse_hn_start", hp_from="F1b", hn_from_start=True, hn_mode="native")'),
]

C74 = """
#@title 7.4 · B3 — análisis (C4: ¿SIGReg evita la caída con HN desde el inicio?)
B3 = {"InfoNCE+HN (pre→mid, F1b)": ("F1b", "infonce_hn"),
      "InfoNCE+HN desde el inicio": ("F2b_B3", "infonce_hn_start"),
      "InfoNCE+SIGReg+HN desde el inicio": ("F2b_B3", "infonce_sigreg_hn_start"),
      "InfoNCE+LSE+HN desde el inicio": ("F2b_B3", "infonce_lse_hn_start")}
if get_bundle() is not None:
    r = compare_variants(study, bundle, B3, "B3", results_phase="F2b")
    if "metrics" not in r:
        print(r.get("nota", r))
    else:
        display(Image(plot_val_curves(study, B3, study.paths.d("results", "F2b", "B3", "curvas_val.png"), metric="valmine_top1")))
        for name in ("tmine_top1", "tgenmacro_top1", "tgem_top1", "tgem_shuf_top1"):
            f = study.paths.d("results", "F2b", "B3", f"pairwise_{name}.csv")
            if os.path.exists(f):
                print(name); display(pd.read_csv(f))
"""

C75 = """
#@title 7.5 · B2 — InfoNCE + HN con candidato "ninguna" aprendido (mecanismo justo de abstención)
if get_bundle() is not None and f1b_listo() and brazo_listo("infonce_hn", "F1b") and entrenamiento_autorizado():
    r = run_variant(study, bundle, "infonce_hn", "F2b_B2", "infonce_hn_null", hp_from="F1b",
                    loss_params={"null_candidate": True, "p_null": 0.2})
    display(pd.DataFrame([{"semilla": x["seed"], "abst_auroc_tmine": (x["test"].get("tabst_mine") or {}).get("abst_auroc_maxcos"),
                           "abst_auroc_null_tmine": (x["test"].get("tabst_mine") or {}).get("abst_auroc_null"),
                           "abst_auroc_tgem": x["test"]["tgem"].get("abst_auroc_maxcos"), "top1_tmine": x["test"]["tmine"]["top1"]}
                          for x in r if x.get("status") == "ok"]))
"""

C76 = """
#@title 7.6 · B1 — régimen R2: término auxiliar de HN sobre los 2 mejores brazos SIN HN (regla fija)
if get_bundle() is not None and f1b_listo() and entrenamiento_autorizado():
    ta = top_arms_rule(study)
    TOP_ARMS = ta["arms"]
    if not TOP_ARMS:
        print("Todavía no se pueden elegir los brazos:", ta.get("nota"), "· No se hace nada.")
    else:
        print("TOP_ARMS:", TOP_ARMS, "·", ta["rule"])
        for arm in TOP_ARMS:
            if brazo_listo(arm, "F1b"):
                print(arm); run_variant(study, bundle, arm, "F2b_B1", f"{arm}_R2aux", hp_from="F1b", hn_mode="aux")
"""

C77 = """
#@title 7.7 · B1 — InfoNCE + HN con "HN solos" (condición explícita de mecanismo, A1.7)
if get_bundle() is not None and f1b_listo() and brazo_listo("infonce_hn", "F1b") and entrenamiento_autorizado():
    r = run_variant(study, bundle, "infonce_hn", "F2b_B1", "infonce_hn_hnonly", hp_from="F1b",
                    loss_params={"hn_denominator": "hn_only"})
    display(pd.DataFrame([{"semilla": x["seed"], "estado": x["status"],
                           "top1_tmine": (x.get("test", {}).get("tmine") or {}).get("top1"),
                           "top1_tgem": (x.get("test", {}).get("tgem") or {}).get("top1"),
                           "top1_tgem_barajado": (x.get("test", {}).get("tgem_shuf") or {}).get("top1")} for x in r]))
"""

C78 = """
#@title 7.8 · B1 — análisis: ¿la ganancia por HN se sostiene en T-mine y con otros generadores?
top = (load_json(study.paths.d("results", "F2b", "B1", "TOP_ARMS.json")) or {}).get("arms") or []
V = {}
for arm in top:
    V[f"{arm} R0 (sin HN, F1b)"] = ("F1b", arm)
    V[f"{arm} R2 (HN auxiliar)"] = ("F2b_B1", f"{arm}_R2aux")
for arm in ("infonce", "infonce_hn", "siglip", "siglip_hn", "triplet_hn"):
    V[f"{arm} (F1b)"] = ("F1b", arm)
V["infonce_hn con HN solos"] = ("F2b_B1", "infonce_hn_hnonly")
if get_bundle() is not None:
    r = compare_variants(study, bundle, V, "B1", results_phase="F2b")
    if "metrics" not in r:
        print(r.get("nota", r))
    for name in ("tmine_top1", "tgenmacro_top1", "trand_top1", "tgem_top1", "tgem_shuf_top1"):
        f = study.paths.d("results", "F2b", "B1", f"summary_{name}.csv")
        if os.path.exists(f):
            print(name); display(pd.read_csv(f))
"""

S8 = """
# 8 · PROTOCOL FREEZE y F3 confirmatorio · **la única decisión del notebook**

**Reglas (§7.8–7.9).** El freeze congela, para cada brazo, los hiperparámetros elegidos en validación (val-mine) en **F1b**. Congela también:

- la receta y los ajustes de la fase;
- el brazo no contrastivo de C1;
- la métrica primaria (T-mine, con el macro de T-gen co-primario);
- el panel T-gen (construcción, versión y generadores del macro);
- las semillas (5), la estadística y el criterio de éxito;
- los tres análisis secundarios pre-especificados:
  - **C2:** ¿cae la ganancia por HN con otros generadores? (ENMIENDA_03 §12.4)
  - **C2b:** ¿esa caída es del generador o del tipo de negativo? (§13)
  - **C2c:** ¿comparar el oro sólo contra sus HN aprende el atajo sin estado y cuesta discriminación real? (§14) Para esto se entrena la variante "HN solos" de infonce_hn, con sus mismos HP congelados (8.17b).

**Después del freeze:**

* no se cambia ningún hiperparámetro ni el código de entrenamiento o evaluación (si cambia, la celda lo advierte y F3 deja de ser confirmatorio);
* el test **confirmatorio** (documentos que nunca se usaron en F0–F2) recién se abre aquí;
* su panel T-gen se genera **después** del freeze, con los mismos generadores del macro y la misma construcción (8.2b);
* también después del freeze se generan los sets del piloto (S, P y E de los mismos 3 generadores) para el confirmatorio, que usa C2b (8.2c);
* si algo se modifica después de mirar F3, esa parte vuelve a ser exploratoria.

**Tu decisión: CONFIRMADA (2026-10-05).** Confirmaste el freeze en la conversación del proyecto. El freeze quedó hecho el 2026-10-06 a las 02:33, desde la rev. 4.4 (código v0.3.4). Si lo vuelves a correr, no se repite ni se modifica.

**Adenda (v0.3.6).** La rev. 4.4 todavía no registraba C2c en el freeze. La 8.1 lo registra en `config/PROTOCOL_FREEZE_ADDENDUM.json`:

- con la especificación exacta de ENMIENDA_03 §14, fijada antes del freeze y antes de abrir el test confirmatorio;
- sin modificar el freeze;
- antes de entrenar la variante.

Al reportar C2c hay que declarar esta desviación: se registró fuera del freeze y se corrió después de ver los demás resultados de F3.
"""

C81 = """
#@title 8.1 · PROTOCOL FREEZE (desde F1b) · decisión del investigador: CONFIRMADA el 2026-10-05
CONFIRMO_FREEZE = True  #@param {type:"boolean"}
#@markdown **Confirmado por el investigador en la conversación del proyecto (2026-10-05).** Con Ejecutar todo, esta celda congela el protocolo y la Sección 8 corre F3. Si alguna compuerta de F0 no pasó, la nota del freeze la explica sola (enmienda A1).
DECISION_FREEZE = ("PROTOCOL FREEZE confirmado por el investigador el 2026-10-05 en la conversación del proyecto "
                   "(notebook rev. 4.5, casilla CONFIRMO_FREEZE marcada a pedido suyo).")
if os.path.exists(freeze_path(study)):
    display(Markdown(open(study.paths.d("config", "PROTOCOL_FREEZE.md")).read()))
    if freeze_addendum(study) is not None:     # el freeze se hizo con un código anterior a v0.3.5: C2c va en la adenda
        display(Markdown(open(study.paths.d("config", "PROTOCOL_FREEZE_ADDENDUM.md")).read()))
elif not CONFIRMO_FREEZE:
    print("Freeze no confirmado: no se congela nada y el test confirmatorio sigue sellado. Las celdas de F3 no hacen nada.")
elif len(brazos_terminados("F1b")) < len(ARMS):
    print(f"Faltan brazos de F1b ({len(brazos_terminados('F1b'))}/{len(ARMS)} terminados): no se puede congelar todavía.")
else:
    malas = [k for k, v in compuertas().items() if k.split()[0] in ("G2", "G3", "G4") and v is not True]
    nota = " ".join([DECISION_FREEZE] + ([f"Compuertas que no pasaron en F0: {malas}. {DECISION_A1}"] if malas else []))
    if not tgen.load_panel(study):
        print("Atención: no hay panel T-gen; el freeze quedará sólo con T-mine como primaria.")
    frz = protocol_freeze(study, phase="F1b", note=nota)
    display(Markdown(open(study.paths.d("config", "PROTOCOL_FREEZE.md")).read()))
"""

C82 = """
#@title 8.2 · Abrir el test confirmatorio (sólo tras el freeze)
if freeze_hecho():
    r_dl = f0.f0_download(study)
    bundle_c = load_bundle(study, test_split="confirm")
    print("test confirmatorio:", bundle_c.test.n, "ítems")
"""

C82B = """
#@title 8.2b · Panel T-gen del test confirmatorio (los generadores del macro congelado; sólo tras el freeze)
#@markdown Misma construcción, prompts, muestreo, reglas y auditoría que en el exploratorio. El macro del confirmatorio son exactamente los generadores congelados en el freeze (no se vuelve a elegir).
N_ITEMS_CONFIRM = 2000
if freeze_hecho():
    frz = load_json(freeze_path(study))
    macro = (frz.get("tgen_panel") or {}).get("macro") or []
    print(f"Disco local libre: {tgen.free_gb(CACHE_ROOT):.1f} GB")
    if not macro:
        print("El freeze no tiene panel T-gen (macro vacío): F3 usará sólo T-mine. No se hace nada.")
    else:
        bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")
        build = tgen.active_build(study)
        if PROFILE == "smoke":
            from clm_losses.smoke import synthetic_lengths
            synthetic_lengths(study, "confirm")
            tgen.synthetic_panel(study, bundle_c.test, split="confirm")
        else:
            for k in macro:
                try:
                    tgen.run_tasks(study, bundle_c.test, k, tgen.BUILDS[build], split="confirm", n_items=N_ITEMS_CONFIRM)
                except Exception as e:
                    print(f"{k}: la generación no terminó ({type(e).__name__}: {e}); vuelve a Ejecutar todo")
            pend = [k for k in macro if tgen.needs_embedding(study, k, "confirm", n_items=N_ITEMS_CONFIRM)]
            if pend and entorno_panel_ok():
                try:
                    enc = encoder_verificado()
                    for k in pend:
                        tgen.embed_generator(study, bundle_c.test, k, enc, split="confirm", n_items=N_ITEMS_CONFIRM)
                except RuntimeError as e:
                    print(e)
                finally:
                    enc = None; liberar_gpu()
        blind_dir = study.paths.d("f0", "F0.6_shortcuts")
        blind = f0.train_blind(get_bundle(), blind_dir, log=lambda *a, **k: None) if os.path.exists(os.path.join(blind_dir, "blind.pt")) else None
        for k in macro:
            tgen.audit_generator(study, bundle_c, k, split="confirm", blind=blind)
        pc = tgen.freeze_panel(study, split="confirm", force_macro=macro)
        display(tgen.panel_table(study, "confirm"))
        print("Macro del confirmatorio:", pc["macro"], "·", pc["n_macro_items"], "ítems")
        if brazos_terminados("F3"):
            print("corridas de F3 actualizadas con T-gen:", add_missing_evals(study, bundle_c, ["F3"]))
"""

C82C = """
#@title 8.2c · Sets del piloto en el test confirmatorio (S, P y E, para C2b; sólo tras el freeze)
#@markdown Los mismos 3 generadores del piloto, con las tres construcciones, en los primeros 500 ítems del panel del confirmatorio: mismos prompts, muestreo y reglas que en el exploratorio. Los usa C2b (8.18c), registrado en el freeze (ENMIENDA_03 §13). ~30–60 min; se retoma si se corta.
if freeze_hecho():
    bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")
    print(f"Disco local libre: {tgen.free_gb(CACHE_ROOT):.1f} GB")
    if PROFILE == "smoke":
        print("Perfil smoke: sets sintéticos del piloto en confirm:", tgen.synthetic_pilot_sets(study, bundle_c.test, split="confirm"))
    elif entorno_panel_ok():
        try:
            r = tgen.build_pilot_sets(study, bundle_c.test, encoder_verificado, split="confirm")
            print("Generadores del piloto:", r["keys"], "· con S, P y E completos:", r["completos"])
            if len(r["completos"]) < len(r["keys"]):
                print("Faltan sets: vuelve a Ejecutar todo (se retoma donde quedó).")
        except RuntimeError as e:
            print(e)
        finally:
            liberar_gpu()
"""

C8T = """
#@title 8.18 · Todos contra todos en F3 + C1 confirmatorio
if freeze_hecho():
    listos3 = brazos_terminados("F3")
    if len(listos3) < 2:
        print(f"Hacen falta al menos 2 brazos terminados en F3 (hay {len(listos3)}). No se hace nada.")
    else:
        bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")
        add_missing_evals(study, bundle_c, ["F3"])          # T-gen del confirmatorio si se generó después de F3
        rep3 = run_all_vs_all(study, bundle_c, "F3")
        c = rep3.get("C1")
        if c:
            for met, cc in (c.get("por_metrica") or {c.get("metrica"): c}).items():
                print(f"C1 (confirmatorio) en {met}: {c['A']} − infonce = {cc['diff']:+.4f} IC95 [{cc['ci95_lo']:+.4f}, "
                      f"{cc['ci95_hi']:+.4f}] p={cc['p']:.3g} p_Holm(C1)={cc.get('p_holm_c1', float('nan')):.3g} -> {cc['verdict']}")
        out3 = study.paths.d("results", "F3", "all_vs_all")
        for name in ("tmine_top1", "tgenmacro_top1", "tgem_top1", "tgem_shuf_top1"):
            f = os.path.join(out3, f"summary_{name}.csv")
            if os.path.exists(f):
                print(name); display(pd.read_csv(f))
        for f in ("heatmap_tmine_top1.png", "heatmap_tgenmacro_top1.png", "heatmap_tgem_top1.png", "cd_diagram.png"):
            if os.path.exists(os.path.join(out3, f)):
                display(Image(os.path.join(out3, f)))
        p = export_summary_md(study, "F3"); print(p)
"""

C8C2 = """
#@title 8.18b · C2 confirmatorio (secundario pre-especificado): ¿la ganancia por HN cae con otros generadores?
#@markdown Por par (con HN − sin HN): Δ en T-gem, Δ en el macro de T-gen del confirmatorio y **C2 = Δ(T-gem) − Δ(macro)**, pareado por documento, con Holm entre los dos pares. Regla registrada en el freeze (ENMIENDA_03 §12.4): H1b se sostiene en un par si los HN suben T-gem (Δ > +1 pp, IC95 > 0) y C2 > +1 pp con p_Holm < 0.05.
if freeze_hecho():
    if not all(a in brazos_terminados("F3") for a in ("infonce", "infonce_hn", "siglip", "siglip_hn")):
        print("Faltan brazos de F3 (infonce, infonce_hn, siglip y siglip_hn). No se hace nada.")
    else:
        bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")
        c2 = c2_report(study, bundle_c, "F3")
        for r in c2["pares"]:
            d1, d2, c = r["Δ T-gem"], r["Δ macro T-gen"], r["C2"]
            print(f"{r['par']}: Δ T-gem {100 * d1['diff']:+.1f} pp · Δ macro {100 * d2['diff']:+.1f} pp · "
                  f"C2 {100 * c['diff']:+.1f} pp IC95 [{100 * c['ci95_lo']:+.1f}, {100 * c['ci95_hi']:+.1f}] "
                  f"p_Holm={c['p_holm']:.3g} -> {r['lectura']}")
"""

C8C3 = """
#@title 8.18c · C2b confirmatorio (secundario pre-especificado): ¿la caída es del generador o del tipo de negativo?
#@markdown La descomposición de EXP-006c (6.6b), en el test confirmatorio, con los brazos de F3. Regla registrada en el freeze (ENMIENDA_03 §13): un componente **se confirma** si es > +1 pp con p_Holm < 0.05 (Holm entre los 4: generador y construcción, en los 2 pares), sólo si la brecha total es > +1 pp con IC95 > 0.
if freeze_hecho():
    if not all(a in brazos_terminados("F3") for a in ("infonce", "infonce_hn", "siglip", "siglip_hn")):
        print("Faltan brazos de F3 (infonce, infonce_hn, siglip y siglip_hn). No se hace nada.")
    else:
        bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")
        c2b = c2b_report(study, bundle_c, "F3")
        for t in c2b["pruebas"]:
            c = t["c"]
            print(f"{t['par']} · {t['componente']}: {100 * c['diff']:+.1f} pp IC95 [{100 * c['ci95_lo']:+.1f}, "
                  f"{100 * c['ci95_hi']:+.1f}] p_Holm={t['p_holm']:.3g} -> {t['lectura']}")
        cr = c2b.get("construccion") or {}
        if cr.get("contrastes") is not None and len(cr["contrastes"]):
            display(cr["contrastes"])
"""

C8V = """
#@title 8.17b · F3 · variante "HN solos" de infonce_hn (para C2c; registrada en el freeze)
#@markdown Mismos HP congelados y semillas que infonce_hn; sólo cambia el denominador: el oro compite sólo contra sus HN (ENMIENDA_03 §14). No entra al todos contra todos ni a C1; la usa C2c (8.18d). ~10 min.
if freeze_hecho():
    var, donde = frozen_variants(study)
    if not var:
        print("Ni el freeze ni su adenda registran variantes de F3: C2c no aplica. No se hace nada.")
    else:
        if donde == "adenda":
            print("Variante registrada en la adenda del freeze (ENMIENDA_03 §14; ver la 8.1).")
        bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")
        for name in var:
            r = run_confirm_variant(study, bundle_c, name)
            display(pd.DataFrame([{"variante": name, "semilla": x.get("seed"), "estado": x.get("status"),
                                   "top1_tmine": (x.get("test", {}).get("tmine") or {}).get("top1"),
                                   "top1_tgem": (x.get("test", {}).get("tgem") or {}).get("top1"),
                                   "top1_tgem_barajado": (x.get("test", {}).get("tgem_shuf") or {}).get("top1")}
                                  for x in r["seeds"]]))
"""

C8C4 = """
#@title 8.18d · C2c confirmatorio (secundario pre-especificado): ¿"HN solos" aprende el atajo sin estado?
#@markdown "HN solos" − infonce_hn ("HN + in-batch"), pareado por documento. Regla registrada en el freeze (ENMIENDA_03 §14): prueba 1, T-gem con estados barajados, se confirma si Δ > +1 pp; prueba 2, T-mine, se confirma si Δ < −1 pp; Holm entre las 2. Control: infonce_hn − infonce en T-gem barajado, equivalente por TOST ±1 pp.
if freeze_hecho():
    if not all(a in brazos_terminados("F3") for a in ("infonce", "infonce_hn")):
        print("Faltan brazos de F3 (infonce e infonce_hn). No se hace nada.")
    else:
        bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")
        c2c = c2c_report(study, bundle_c, "F3")
        for t in c2c["pruebas"]:
            c = t["c"]
            print(f"{t['prueba']}: {100 * c['diff']:+.1f} pp IC95 [{100 * c['ci95_lo']:+.1f}, {100 * c['ci95_hi']:+.1f}] "
                  f"p_Holm={t['p_holm']:.3g} -> {t['lectura']}")
        if c2c.get("control"):
            c = c2c["control"]["c"]
            print(f"Control ({c2c['control']['contraste']}, T-gem barajado): {100 * c['diff']:+.1f} pp IC95 "
                  f"[{100 * c['ci95_lo']:+.1f}, {100 * c['ci95_hi']:+.1f}] -> {c2c['control']['lectura']}")
        for k, c in (c2c.get("descriptivos") or {}).items():
            print(f"  (descriptivo) {k}: {100 * c['diff']:+.1f} pp IC95 [{100 * c['ci95_lo']:+.1f}, {100 * c['ci95_hi']:+.1f}]")
        print("Lectura:", c2c.get("lectura_global"))
"""

C8R = """
#@title 8.19 · T-Rex con y sin shield para cada brazo (D2) · requiere L4 o A100
if freeze_hecho():
    if not brazos_terminados("F3"):
        print("Todavía no hay brazos terminados en F3. No se hace nada.")
    else:
        pip("transformers>=4.51", "accelerate", "fastapi", "uvicorn", "httpx", "requests")
        tr = trex_for_arms(study, phase="F3", seed=0)
        display(pd.DataFrame({a: {f"{m}_{k}": v[m].get(k) for m in v
                                  for k in ("mean_agreement_with_planner", "deaths", "shield_interventions")}
                              for a, v in tr.items()}).T)
"""

S9 = """
# 9 · Qué hacer con los resultados

Cuando termine cada paso, avisa en la conversación del proyecto. Lo útil para revisar está en Drive:

| Paso | Archivos |
|---|---|
| Panel | `data/eval/tgen/test/PANEL.json` y `blind_transfer.csv` (6.3) |
| EXP-006 y EXP-006c | `results/transfer/EXP006/` (6.6) y `results/transfer/EXP006c/` (6.6b) |
| **F3 (confirmatorio)** | `config/PROTOCOL_FREEZE.md`, `results/F3/RESUMEN_F3.md`, `results/F3/all_vs_all/` (C1), `results/F3/C2/`, `results/F3/C2b/`, `results/F3/C2c/` y `results/F3/trex/` |
| F0.7b | `f0/F0.7b_recipes/result.json` y `config/RECETA_F1b.json` |
| F1b | `results/F1b/RESUMEN_F1b.md` y los CSV de `results/F1b/all_vs_all/` |
| F0 | `f0/F0_REPORT.md` |

Con eso se actualizan `PROJECT_STATE.md`, `EXPERIMENT_LOG.md` y la matriz claim–evidencia. Los resultados de F1/F1b son **exploratorios**; sólo F3 sostiene conclusiones confirmatorias.
"""


# --------------------------------------------------------------------------------------
def build(out_path: str) -> dict:
    data = package_zip()
    b64 = base64.b64encode(data).decode()
    sha = hashlib.sha256(data).hexdigest()
    h = code_hash()
    c12 = C12.replace("{TF_MIN}", PANEL_MIN_TRANSFORMERS)
    c14 = C14_CORE.replace("{NB_REV}", NB_REV)
    boot = "\n".join(x.strip("\n") for x in (C11, c12, C13_BOOT, c14)) + "\n"
    c14 = c14.rstrip("\n") + "\n" + C14_WRITER.strip("\n").replace(
        "{BOOT_B64}", base64.b64encode(boot.encode()).decode()) + "\n"
    cells = [md(INTRO.replace("{HASH}", h)), md(RESUME), md(RESOURCES), md(S1),
             code(C11, form=True, guard=False), code(c12, form=True, guard=False),
             code(C13.replace("{B64}", b64).replace("{SHA}", sha), form=True, guard=False),
             code(c14, form=True, guard=False),
             md(S2), code(C21, form=True), md(S3)]
    for title, desc, src in F0_STEPS:
        cells += [md(f"## {title}\n\n{desc}"), code(src, form=True)]
    cells += [md(S4), code(C40, form=True), code(C4PILOT, form=True)]
    for i, (arm, spec) in enumerate(sorted(ARMS.items(), key=lambda kv: kv[1]["num"]), start=1):
        name, desc = ARM_DOCS[arm]
        hn = "sí" if spec["hn"] else "no"
        pool = " · **pool de C1**" if arm in NONCONTRASTIVE_POOL else ""
        txt = (f"## 4.{i} · F1b · Brazo {spec['num']} — {name}\n\n"
               f"Familia: **{spec['family']}** · usa HN: **{hn}**{pool} · id: `{arm}`\n\n{desc}\n\n"
               f"Valores por defecto: `{spec['defaults'] or '{}'}` · Espacio de búsqueda (F1b): {hpo_space_text(arm, 'F1b')}")
        src = (f'#@title 4.{i} · F1b · Brazo {spec["num"]} · {arm}\n'
               f'resumen = load_json(study.paths.d("runs", "F1b", "{arm}", "ARM_SUMMARY.json"))\n'
               f'if brazo_completo("{arm}", "F1b") and resumen:          # ya terminado: sólo muestra el resumen\n'
               f'    display(summarize_arm(resumen))\n'
               f'elif f1b_listo() and get_bundle() is not None and (brazo_completo("{arm}", "F1b") or entrenamiento_autorizado()):\n'
               f'    r = run_arm(study, bundle, "{arm}", phase="F1b")\n'
               f'    display(summarize_arm(r))')
        cells += [md(txt), code(src, form=True)]
    cells += [code(C4P, form=True), code(C4B, form=True), md(S5), code(C51, form=True), code(C52, form=True),
              code(C53, form=True), md("## 5.4 · Atajos en el F1 piloto: ¿cuánto depende cada brazo del estado?"),
              code(C54, form=True),
              md("## 5.5 · Atajos según cómo entran los HN (F0.7): ¿los crea el denominador?"), code(C55, form=True),
              md("## 5.6 · Uso del estado por brazo (cualquier fase)"), code(C56, form=True),
              md(S6.replace("{PANEL_TABLE}", panel_md())), code(C60, form=True), code(C61, form=True), code(C62, form=True),
              code(C63, form=True), code(C64, form=True), code(C65, form=True), code(C66, form=True),
              code(C66B, form=True), code(C67, form=True),
              md(S7)]
    for title, needs, line in B3_CELLS:
        cells += [md(f"## {title}"), code(f'#@title {title}\n'
                                           f'if get_bundle() is not None and f1b_listo() and brazo_listo("{needs}", "F1b") '
                                           f'and entrenamiento_autorizado():\n'
                                           f'    {line}\n'
                                           f'    display(pd.DataFrame([{{"semilla": x["seed"], "estado": x["status"], '
                                           f'"top1_tmine": x.get("test", {{}}).get("tmine", {{}}).get("top1"), '
                                           f'"top1_tgem": x.get("test", {{}}).get("tgem", {{}}).get("top1"), '
                                           f'"top1_tgem_barajado": x.get("test", {{}}).get("tgem_shuf", {{}}).get("top1")}} for x in r]))',
                                           form=True)]
    cells += [md("## 7.4 · B3 — análisis"), code(C74, form=True),
              md("## 7.5 · B2 — candidato \"ninguna\""), code(C75, form=True),
              md("## 7.6–7.8 · B1 — regímenes de HN y \"HN solos\""), code(C76, form=True), code(C77, form=True),
              code(C78, form=True),
              md(S8), code(C81, form=True), code(C82, form=True), code(C82B, form=True), code(C82C, form=True)]
    for i, (arm, spec) in enumerate(sorted(ARMS.items(), key=lambda kv: kv[1]["num"]), start=3):
        name, _ = ARM_DOCS[arm]
        src = (f'#@title 8.{i} · F3 · brazo {spec["num"]} · {arm}\n'
               f'if freeze_hecho():\n'
               f'    bundle_c = globals().get("bundle_c") or load_bundle(study, test_split="confirm")\n'
               f'    r = run_confirm_arm(study, bundle_c, "{arm}")\n'
               f'    display(summarize_arm(r))')
        cells += [md(f"## 8.{i} · F3 — Brazo {spec['num']}: {name}"), code(src, form=True)]
    cells += [md("## 8.17b · F3 — variante \"HN solos\" de infonce_hn (para C2c)"), code(C8V, form=True),
              code(C8T, form=True), code(C8C2, form=True), code(C8C3, form=True), code(C8C4, form=True),
              code(C8R, form=True), md(S9)]
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": [], "toc_visible": True, "gpuType": "T4"},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}, "accelerator": "GPU"},
          "cells": cells}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return {"cells": len(cells), "zip_bytes": len(data), "zip_sha256": sha, "code_hash": h,
            "notebook_bytes": os.path.getsize(out_path)}


# --------------------------------------------------------------------------------------
# Notebook de análisis post hoc de F3 (no entrena, no evalúa, no necesita GPU): sensibilidad del panel T-gen.
# Usa el MISMO paquete (mismo zip y hash que el notebook principal) más un script de analisis/.
NB_A_REV = "rev. A1 · 2026-10-07"
SENS_SCRIPT = os.path.join(ROOT, "analisis", "F3_sensibilidad", "sensibilidad_panel.py")

INTRO_A = """
# CLM losses — sensibilidad del panel T-gen en F3 (post hoc)

**{NB_A_REV}** · código clm_losses {VERSION} (hash `{HASH}`, el mismo del notebook principal rev. 4.7) + script `analisis/F3_sensibilidad/sensibilidad_panel.py` (sha256 `{SSHA}`).

**Qué hace.** Lee las corridas de F3 que ya están en tu Drive y repite la parte T-gen de C1 y C2 cambiando qué generadores entran al macro:

- los 5 congelados: es el control, tiene que dar lo mismo que el confirmatorio;
- sin cada uno de ellos;
- cada uno solo.

**Qué no hace.** No entrena, no evalúa, no descarga datos, no necesita GPU y no modifica ninguna corrida. Sólo escribe `results/F3/sensibilidad_panel/`.

**Por qué.** En el test confirmatorio, la auditoría de Phi-4 dio 0.312 (umbral 0.30). Además, la ventaja de InfoNCE sobre VICReg es muy distinta según el generador (EXP-011). El macro congelado se mantiene; este análisis es descriptivo y post hoc, y así se reporta.

**Cómo.** Entorno de ejecución → **Ejecutar todo** (basta con CPU). Tarda ~5 min. Al terminar, sube el notebook ejecutado.
"""

CA1 = """
#@title A.1 · Parámetros y Drive (fijos: no hay nada que cambiar)
import os, sys
PROFILE = "colab"
DRIVE_BASE = "/content/drive/MyDrive/CLM_losses"
CACHE_ROOT = "/content/cache"
USE_DRIVE_CODE = False
from google.colab import drive
drive.mount("/content/drive")
if not os.path.isdir(os.path.join(DRIVE_BASE, PROFILE)):
    raise SystemExit(f"No encuentro {DRIVE_BASE}/{PROFILE} en tu Drive: abre este notebook con la cuenta del estudio.")
print("Carpeta del estudio:", os.path.join(DRIVE_BASE, PROFILE))
"""

CA3 = """
#@title A.3 · Sensibilidad del panel T-gen en F3 (post hoc, descriptivo; sólo lee las corridas)
import base64, hashlib, importlib.util
import pandas as pd
from IPython.display import Markdown, display
from clm_losses.pipeline import freeze_path, setup
SCRIPT_B64 = "{SCRIPT_B64}"
SCRIPT_SHA256 = "{SCRIPT_SHA}"
src = base64.b64decode(SCRIPT_B64)
assert hashlib.sha256(src).hexdigest() == SCRIPT_SHA256, "el script embebido está corrupto"
os.makedirs("/content/analisis", exist_ok=True)
open("/content/analisis/sensibilidad_panel.py", "wb").write(src)
spec = importlib.util.spec_from_file_location("sensibilidad_panel", "/content/analisis/sensibilidad_panel.py")
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)
study = setup(PROFILE, drive_base=DRIVE_BASE, cache_root=CACHE_ROOT)
if not os.path.exists(freeze_path(study)):
    raise SystemExit("En esta carpeta no hay PROTOCOL FREEZE: este análisis es sólo para F3. No se hizo nada.")
res = sp.panel_sensitivity(study, meta={"code_hash": CODE_HASH, "script_sha256": SCRIPT_SHA256, "notebook": "{NB_A_REV}"})
if res:
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
    display(sp.tabla(res))
    display(Markdown(open(os.path.join(study.paths.d("results", "F3", "sensibilidad_panel"), "SENSIBILIDAD.md"),
                          encoding="utf-8").read()))
"""


def build_analysis(out_path: str) -> dict:
    """CLM_losses_Analisis_F3.ipynb: sensibilidad del panel en F3, sin GPU y sin tocar las corridas."""
    data = package_zip()
    b64, sha, h = base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest(), code_hash()
    script = open(SENS_SCRIPT, "rb").read()
    ssha = hashlib.sha256(script).hexdigest()

    def cell(src, hidden):
        src = src.strip("\n")
        return {"cell_type": "code", "execution_count": None, "metadata": {"cellView": "form"} if hidden else {},
                "outputs": [], "source": src.splitlines(keepends=True)}
    ca2 = C13.replace("#@title 1.3 · Instalar el código del estudio (embebido en el notebook)",
                      "#@title A.2 · Instalar el código del estudio (el mismo del notebook principal)")
    assert ca2 != C13
    ca2 = ca2.replace("{B64}", b64).replace("{SHA}", sha)
    ca3 = CA3.replace("{SCRIPT_B64}", base64.b64encode(script).decode()).replace("{SCRIPT_SHA}", ssha) \
        .replace("{NB_A_REV}", NB_A_REV)
    intro = INTRO_A.replace("{NB_A_REV}", NB_A_REV).replace("{VERSION}", __version__).replace("{HASH}", h) \
        .replace("{SSHA}", ssha[:12])
    cells = [md(intro), cell(CA1, False), cell(ca2, True), cell(ca3, True)]
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": []}, "kernelspec": {"name": "python3", "display_name": "Python 3"},
                       "language_info": {"name": "python"}},
          "cells": cells}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return {"cells": len(cells), "zip_sha256": sha, "code_hash": h, "script_sha256": ssha,
            "notebook_bytes": os.path.getsize(out_path)}


# --------------------------------------------------------------------------------------
# Notebook pre-envío (v0.3.8; fuera del protocolo): extracción de Drive, calidad de los sets, condición de borde de B5
# y referencias sin entrenar. Mismo paquete embebido (v0.3.8) que se genera aquí.
NB_P_REV = "rev. P2 · 2026-11-16"

INTRO_P = """
# CLM losses — notebook pre-envío

**{NB_P_REV}** · código clm_losses {VERSION} (hash `{HASH}`)

**Qué es.** Lo que falta correr antes de enviar el paper a JAIR. Está **fuera del protocolo**: no toca el freeze, no re-entrena nada de F3 y no cambia ningún veredicto confirmatorio. Cada bloque guarda en Drive su regla de lectura, con fecha, **antes** de correr, y no la reescribe.

**Cómo.** Con la GPU de siempre (RTX PRO 6000): Entorno de ejecución → **Ejecutar todo**. No hay nada que elegir. **Si la GPU tiene menos de 70 GB, la primera celda se detiene:** el 2026-10-12 tocó una T4 (15 GB), el juez no cabía y en un día juzgó 17 de 24,732 ítems.

- Tarda unas **3–5 horas**. Lo más lento es el juez (~1–1.5 h para ~25,000 ítems) y las descargas: unos 150 GB entre el juez, los embeddings de entrenamiento y Qwen3-Embedding-8B. Los pesos del juez y de Qwen3-Embedding se borran del disco local al terminar.
- Si la sesión se corta, vuelve a darle **Ejecutar todo**: cada paso retoma donde quedó y no repite lo terminado.
- Si un paso falla, queda anotado y el notebook sigue con el siguiente. El resumen final dice qué pasó en cada uno.

| Bloque | Qué hace | ¿Entrena? | Descarga | Tiempo aprox. |
|---|---|---|---|---|
| 0 | Extrae de Drive lo que ya existe: entorno por fase, F3 frente a F1b, Friedman sin T-gem barajado, puntuaciones secundarias y B4 | no | nada | minutos |
| 1 | Calidad de los sets del sellado: falsos negativos (juez OLMo 3.1 32B sobre todos los ítems de T-gem, T-mine y los 5 generadores del macro) y fuga entre splits; con eso, sensibilidad de C1, C2 y C2c | no | tokens 10.5 GB y juez 64 GB | ~2 h |
| 2 | Condición de borde de B5: 6 denominadores de InfoNCE+HN × 3 semillas, en el split de test (exploratorio) | sí, 18 corridas de ~1 min | embeddings 66 GB | ~1 h |
| 3 | Referencias sin entrenar: Qwen3-Embedding-8B, BM25 y el coseno crudo de Qwen3-8B, con estados reales y barajados | no | 16 GB | ~45 min |

**Etiquetas a mano: ya están en Drive.** `CLM_losses/colab/results/PRE/b1_calidad/etiquetas/ETIQUETADAS.xlsx`: 6 alumnos, cada pregunta marcada por dos por separado; es la referencia estricta (cuenta como correcto sólo lo que marcaron los dos; DP-10 y DP-11). La amplia se calcula aparte. No hay nada que hacer a mano.

**Dónde queda todo.** En `CLM_losses/colab/results/PRE/`. `RESUMEN_PREENVIO.md` junta los cuatro informes; las corridas del bloque 2 quedan en `runs/PRE_borde/`. Al terminar, avisa en la conversación: los resultados se leen de Drive.

**Juez de falsos negativos: `allenai/Olmo-3.1-32B-Instruct`** (bf16, Apache-2.0). El diseño proponía gpt-oss-120b, pero no corre en esta GPU: sus kernels MXFP4 no cubren sm_120 y des-cuantizado no cabe. OLMo 3.1 32B ya corrió aquí en el panel T-gen. No es de la familia del oro (Qwen3), ni de la de los HN de entrenamiento (Gemini/Gemma), ni uno de los 5 generadores del macro.
"""

CP1 = """
#@title P.1 · Parámetros, Drive y GPU (fijos: no hay nada que cambiar)
import os, shutil, subprocess, sys
PROFILE = "colab"
DRIVE_BASE = "/content/drive/MyDrive/CLM_losses"
CACHE_ROOT = "/content/cache"
USE_DRIVE_CODE = False
from google.colab import drive
drive.mount("/content/drive")
if not os.path.isdir(os.path.join(DRIVE_BASE, PROFILE)):
    raise SystemExit(f"No encuentro {DRIVE_BASE}/{PROFILE} en tu Drive: abre este notebook con la cuenta del estudio.")
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else ""
print("Carpeta del estudio:", os.path.join(DRIVE_BASE, PROFILE))
print("GPU (nombre, driver, memoria):", gpu or "ninguna")
try:
    GPU_MIB = int(gpu.splitlines()[0].split(",")[-1].strip().split()[0]) if gpu else 0
except (ValueError, IndexError):
    GPU_MIB = 0
if GPU_MIB < 70000:
    raise SystemExit("ALTO: este notebook necesita la GPU grande (RTX PRO 6000, 96 GB). Con esta GPU "
                     f"({gpu or 'ninguna'}) el juez (64 GB) no cabe y corre en CPU: tarda días y no termina. "
                     "Entorno de ejecución → Cambiar tipo de entorno de ejecución → RTX PRO 6000, y vuelve a darle "
                     "Ejecutar todo. Lo ya hecho no se repite.")
if "RTX PRO 6000" not in gpu:
    print("ATENCIÓN: no es la RTX PRO 6000 de F1b y F3. El bloque 2 compara con esas corridas: con otra GPU el chequeo "
          "de reproducción puede dar diferencias numéricas.")
"""

CP2 = """
#@title P.2 · Dependencias
def pip(*pkgs):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--root-user-action=ignore", *pkgs], check=True)
pip("huggingface_hub>=0.24", "pyarrow>=14", "scipy", "pandas", "matplotlib", "openpyxl", "transformers>={TF_MIN}",
    "accelerate>=0.33", "sentence-transformers>=5.0")
# transformers se instala aquí, antes de importarlo: actualizarlo a mitad de sesión mezcla versiones en memoria
try:
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    print("HF_TOKEN cargado desde los secretos de Colab")
except Exception:
    print("Sin HF_TOKEN (opcional): los modelos de este notebook no tienen acceso restringido")
"""

CP4 = """
#@title P.4 · Configuración del estudio
import warnings
import torch
from IPython.display import Markdown, display
from clm_losses import preenvio as pre
from clm_losses.pipeline import setup
warnings.filterwarnings("ignore", message="To exit")
study = setup(PROFILE, drive_base=DRIVE_BASE, cache_root=CACHE_ROOT)
GPU = torch.cuda.is_available()

def paso(nombre, fn):
    # corre un paso; si falla, lo anota en results/PRE/ESTADO.json y el notebook sigue
    return pre.correr(study, nombre, fn)

def mostrar(*rel):
    p = pre.pdir(study, *rel)
    if os.path.exists(p):
        display(Markdown(open(p, encoding="utf-8").read()))
    else:
        print("Todavía no hay", p)

print(f"Resultados en {pre.pdir(study)} · GPU: {'sí' if GPU else 'no'}")
"""

CB0 = """
#@title 0 · Bloque 0: extracción de lo que ya está en Drive (sin entrenar ni descargar)
paso("bloque 0: extracción", lambda: pre.bloque0(study))
mostrar("b0_extraccion", "EXTRACCION.md")
"""

CB11 = """
#@title 1.1 · Textos de test y sellado, y corpus de BM25 (tokens de Nemotron-DQA, 10.5 GB)
paso("bloque 1: textos y corpus de BM25", lambda: pre.paso_textos(study))
"""

CB12 = """
#@title 1.2 · CSV para etiquetar a mano (sale ANTES del juez; se exporta una sola vez)
r = paso("bloque 1: CSV para etiquetar a mano", lambda: pre.paso_etiquetas(study))
if r:
    print("Para etiquetar:", r)
    print("Cuando lo tengas, súbelo a esa carpeta como ETIQUETADAS.csv (o .xlsx) y vuelve a Ejecutar todo.")
"""

CB13 = """
#@title 1.3 · Juez de falsos negativos (OLMo 3.1 32B, ~64 GB; reanudable)
if GPU:
    paso("bloque 1: juez de falsos negativos", lambda: pre.paso_juez(study))
else:
    pre.mark(study, "bloque 1: juez de falsos negativos", "omitido", motivo="sin GPU")
    print("Sin GPU: el juez se salta.")
"""

CB14 = """
#@title 1.4 · Fuga entre splits (pares pregunta–oro repetidos o casi repetidos en entrenamiento)
paso("bloque 1: fuga entre splits", lambda: pre.paso_fuga(study))
"""

CB15 = """
#@title 1.5 · Informe de calidad: tasas, acuerdo con las etiquetas a mano y sensibilidad de C1, C2 y C2c
paso("bloque 1: informe de calidad", lambda: pre.paso_calidad(study))
mostrar("b1_calidad", "CALIDAD.md")
"""

CB2 = """
#@title 2 · Bloque 2: condición de borde de B5 (18 corridas en el split de test; reanudable)
if GPU:
    paso("bloque 2: condición de borde", lambda: pre.bloque2(study))
else:
    pre.mark(study, "bloque 2: condición de borde", "omitido", motivo="sin GPU")
    print("Sin GPU: el bloque 2 se salta.")
pre._free_gpu()
mostrar("b2_borde", "BORDE.md")
"""

CB3 = """
#@title 3 · Bloque 3: referencias sin entrenar (Qwen3-Embedding-8B, BM25, coseno crudo)
paso("bloque 3: referencias sin entrenar", lambda: pre.bloque3(study, qwen=GPU))
mostrar("b3_referencias", "REFERENCIAS.md")
"""

CR = """
#@title R · Resumen (súbelo a la conversación junto con este notebook ejecutado)
import pandas as pd
md = pre.resumen(study)
est = pre.state(study)
display(pd.DataFrame([{"paso": k, "estado": v.get("estado"), "hora": v.get("hora"),
                       "detalle": v.get("motivo") or v.get("error") or (f"{v.get('minutos')} min" if v.get("minutos") is not None else "")}
                      for k, v in est.items()]))
print("Resumen completo:", pre.pdir(study, "RESUMEN_PREENVIO.md"))
if not pre.read_hand_labels(study):
    print("Falta subir ETIQUETADAS.csv (bloque 1): sin eso, el acuerdo del juez con las etiquetas a mano queda pendiente.")
"""


def build_preenvio(out_path: str) -> dict:
    """CLM_losses_PreEnvio.ipynb: el notebook pre-envío (v0.3.8), con el paquete embebido."""
    data = package_zip()
    b64, sha, h = base64.b64encode(data).decode(), hashlib.sha256(data).hexdigest(), code_hash()

    def cell(src):
        src = src.strip("\n")
        return {"cell_type": "code", "execution_count": None, "metadata": {"cellView": "form"}, "outputs": [],
                "source": src.splitlines(keepends=True)}
    cp3 = C13.replace("#@title 1.3 · Instalar el código del estudio (embebido en el notebook)",
                      "#@title P.3 · Instalar el código (v0.3.8, embebido en el notebook)")
    assert cp3 != C13
    cp3 = cp3.replace("{B64}", b64).replace("{SHA}", sha)
    intro = INTRO_P.replace("{NB_P_REV}", NB_P_REV).replace("{VERSION}", __version__).replace("{HASH}", h)
    cells = [md(intro), md("## Preparación"), cell(CP1), cell(CP2.replace("{TF_MIN}", PANEL_MIN_TRANSFORMERS)),
             cell(cp3), cell(CP4),
             md("## Bloque 0 · Extracción de lo que ya está en Drive\n\nNo entrena ni descarga nada. Cierra los [VERIFY] "
                "del Diseño del estudio: entorno por fase, si F3 repitió los modelos de F1b, Friedman con 7 bloques, "
                "puntuaciones secundarias y B4."),
             cell(CB0),
             md("## Bloque 1 · Calidad de los sets del sellado\n\nEl diseño lo pedía en F0.4 y F0.5 y no se hizo. "
                "Primero salen los textos y el CSV ciego para etiquetar; después corre el juez sobre todos los ítems; "
                "luego, la fuga entre splits. El informe aplica la regla escrita antes de correr y recalcula C1, C2 y C2c "
                "sin los ítems marcados. Sin exclusiones, ese cálculo tiene que reproducir lo confirmatorio: el informe "
                "lo comprueba."),
             cell(CB11), cell(CB12), cell(CB13), cell(CB14), cell(CB15),
             md("## Bloque 2 · Condición de borde de B5 (exploratoria, split de test)\n\n¿Por qué los oros de otros "
                "estados quitan la preferencia sin estado que aprende \"sólo HN\"? Seis denominadores, con los "
                "hiperparámetros de F1b y semillas 0–2. Antes de entrenar las condiciones nuevas se comprueba que "
                "\"sólo HN\" y \"HN + in-batch\" reproducen sus corridas de referencia; si no, se detiene. "
                "Regla de lectura: INNOVATION_MAP §2.8."),
             cell(CB2),
             md("## Bloque 3 · Referencias sin entrenar\n\nDescriptivas, en test y sellado, con estados reales y "
                "barajados. El estado es la pregunta sola para Qwen3-Embedding-8B y BM25."),
             cell(CB3), md("## Resumen"), cell(CR)]
    nb = {"nbformat": 4, "nbformat_minor": 0,
          "metadata": {"colab": {"provenance": [], "toc_visible": True}, "accelerator": "GPU",
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
          "cells": cells}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    return {"cells": len(cells), "zip_sha256": sha, "code_hash": h, "notebook_bytes": os.path.getsize(out_path)}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "CLM_losses_Colab.ipynb"))
    ap.add_argument("--zip-out", default=os.path.join(ROOT, "clm_losses_pkg.zip"))
    ap.add_argument("--analisis", default=None, help="además, escribe el notebook de sensibilidad de F3 en esta ruta")
    ap.add_argument("--preenvio", default=None, help="además, escribe el notebook pre-envío (v0.3.8) en esta ruta")
    a = ap.parse_args()
    info = build(a.out)
    with open(a.zip_out, "wb") as f:
        f.write(package_zip())
    if a.analisis:
        info["analisis"] = build_analysis(a.analisis)
    if a.preenvio:
        info["preenvio"] = build_preenvio(a.preenvio)
    print(json.dumps(info, indent=1))
