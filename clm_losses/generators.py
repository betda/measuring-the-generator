"""Panel de generadores para T-gen (ENMIENDA_03; reemplaza a P3).

Para cada generador del panel: los mismos ítems del test (≤ 1 por documento, orden fijo), los mismos prompts
y el mismo muestreo. Pasos (cada uno reanudable y guardado en Drive, en data/eval/tgen/<split>/<clave>/):

    1. textos      pregunta y oro de los ítems (tokens públicos de Nemotron-DQA, decodificados con Qwen3)
    2. generar     un subproceso por generador (clm_losses.genworker): libera la GPU al terminar. Tareas:
                   "neg" (respuestas incorrectas libres), "edit" (el oro con su dato clave cambiado) y
                   "para" (el oro reescrito por el mismo generador, con un largo asignado)
    3. construir   el set de cada ítem según la construcción elegida por el piloto (S, P o E; ENMIENDA_03 §11)
                   y embeberlo con Qwen3-8B (pipeline validado por G1)
    4. auditar     sondas ciegas: longitud, ciego de Gemini transferido, ciego propio (validación cruzada),
                   coseno crudo; matriz de transferencia entre ciegos
    5. panel       PANEL.json: qué generadores entran al macro de T-gen (regla fija, §3.6 de la enmienda)

Construcciones (§11; el piloto elige una con regla fija, sin mirar ningún modelo entrenado):
    S  negativos libres + oro original, con reglas de texto (formato igualado al oro, sin ítems de sí/no u
       opción múltiple, sin respuestas que contienen al oro)
    P  negativos libres + oro REESCRITO por el mismo generador (extensión pre-registrada en §3.6), con un largo
       asignado al azar para que la longitud no delate cuál es el correcto; formato canónico para los 11
    E  negativos EDITADOS (el oro con su dato clave cambiado: mismo largo, estilo y formato) + oro original

Hecho que motiva los controles [VERIFIED, tarjeta de nvidia/Nemotron-CC-v2.1]: el oro (High-Quality-DQA)
lo escribió Qwen3-30B-A3B. La v1 del panel (2026-10-02) mostró que, aun con ese mismo modelo como
generador, el oro se reconoce sin ver la pregunta (sobre todo por su longitud): por eso las construcciones.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field

import numpy as np

from .utils import (append_jsonl, atomic_np_save, load_json, makedirs, now_iso, read_jsonl, save_json)

PROMPT_VERSION = "tgen-v1"       # respuestas incorrectas libres (tarea "neg"; construcciones S y P)
PROMPT = (
    "You are helping build a benchmark of hard incorrect answers for question answering.\n\n"
    "Question: {question}\n"
    "Correct answer: {answer}\n\n"
    "Write {k} different answers to the question that are plausible and on-topic but factually INCORRECT.\n"
    "Rules:\n"
    "- Each answer must have about the same length as the correct answer (around {n_words} words) and the "
    "same style and format.\n"
    "- Do not copy, paraphrase, or simply negate the correct answer; each one must state something wrong.\n"
    "- Do not say that the answers are incorrect and do not add explanations.\n"
    "Return exactly {k} lines, one answer per line, with no numbering, bullets, or extra text."
)
EDIT_PROMPT_VERSION = "tgen-edit-v1"   # el oro con su dato clave cambiado (tarea "edit"; construcción E)
EDIT_PROMPT = (
    "You are helping build a benchmark of hard incorrect answers for question answering.\n\n"
    "Question: {question}\n"
    "Correct answer: {answer}\n\n"
    "Write {k} different INCORRECT answers by editing the correct answer: replace its key fact or facts (a name, "
    "number, term, place, date, method, cause, or relation) with different, plausible, but wrong ones, and keep the "
    "rest of the wording the same.\n"
    "Rules:\n"
    "- Each answer must be clearly wrong for this question, not a synonym or a rewording of the correct answer.\n"
    "- Keep about the same number of words as the correct answer ({n_words}) and the same format, punctuation, and "
    "capitalization.\n"
    "- Make each replacement as specific as the original (for example, the same number of digits and the same "
    "units).\n"
    "- Do not make an answer wrong just by adding a negation such as \"not\" or \"no\".\n"
    "- Do not say that the answers are incorrect and do not add explanations.\n"
    "Return exactly {k} lines, one answer per line, with no numbering, bullets, or extra text."
)
PARA_PROMPT_VERSION = "tgen-para-v1"   # el oro reescrito por el generador (tarea "para"; construcción P)
PARA_PROMPT = (
    "Rewrite the correct answer to this question in your own words.\n\n"
    "Question: {question}\n"
    "Correct answer: {answer}\n\n"
    "Write {k} different rewrites. Each rewrite must:\n"
    "- keep the same meaning and stay correct, with the same key facts and no new facts;\n"
    "- be about {n_words} words long (shorten or expand the wording as needed);\n"
    "- not copy the correct answer word for word.\n"
    "Do not add explanations. Return exactly {k} lines, one rewrite per line, with no numbering, bullets, or "
    "extra text."
)

K_KEEP = 10          # negativos por ítem en el set final (como T-gem)
K_REQUEST = 14       # se piden 4 de reserva (parseo y filtros)
K_PARA = 3           # reescrituras del oro pedidas por ítem (se usa una)
SHORT_GOLD_WORDS = 3 # en P, un oro de hasta 3 palabras sin reescritura válida usa su propio texto como positivo
FN_THRESHOLD = 0.95  # umbral de T-mine; en el panel v2 sólo se reporta (ENMIENDA_03 §11)
BANDS = (0.15, 0.30)  # bandas de G3
PANEL_N = 2000       # ítems del panel por generador (los controles pesados usan menos)
PILOT_N = 500        # ítems del piloto de construcción (los primeros del orden del panel)
PILOT_CANDIDATES = ("mistral7b", "granite33_8b", "llama31_8b", "olmo2_7b", "falcon3_7b", "yi15_9b")
PILOT_K = 3          # generadores del piloto: los primeros 3 de PILOT_CANDIDATES con acceso

# tarea -> archivos y parámetros de generación
TASKS = {
    "neg": {"raw": "raw.jsonl", "info": "gen_info.json", "version": PROMPT_VERSION, "k": K_REQUEST, "k_keep": K_KEEP},
    "edit": {"raw": "raw_edit.jsonl", "info": "gen_info_edit.json", "version": EDIT_PROMPT_VERSION, "k": K_REQUEST,
             "k_keep": K_KEEP},
    "para": {"raw": "para.jsonl", "info": "gen_info_para.json", "version": PARA_PROMPT_VERSION, "k": K_PARA,
             "k_keep": 1},
}
TASK_PROMPT = {"neg": PROMPT, "edit": EDIT_PROMPT, "para": PARA_PROMPT}
# construcción -> tareas que necesita (en orden: "para" usa los negativos de "neg")
BUILDS = {"S": ("neg",), "P": ("neg", "para"), "E": ("edit",)}
BUILD_TXT = {"S": "negativos libres + oro original (reglas de texto)",
             "P": "negativos libres + oro reescrito por el mismo generador (largo asignado al azar)",
             "E": "negativos editados (cambian sólo el dato clave) + oro original"}
BUILD_PREF = ("S", "P", "E")   # desempate del piloto: la más barata (S no genera nada; E regenera todo)


@dataclass(frozen=True)
class Generator:
    key: str
    repo: str
    family: str
    role: str                 # primario | control_gemini | control_qwen | control_oro
    license: str
    gated: bool
    params_b: float
    chat_kwargs: dict = field(default_factory=dict)
    harmony: bool = False     # formato "harmony" (gpt-oss): se guarda sólo el canal final
    min_transformers: str = "4.53"
    n_items: int | None = None  # None = los N del panel; los controles pesados usan menos
    dtype: str = "bfloat16"
    note: str = ""
    model_kwargs: dict = field(default_factory=dict)
    # salida con bloque de razonamiento (p. ej. Gemma 4): se guarda sólo lo que sigue a final_marker; si aparece
    # reasoning_start sin final_marker (razonamiento sin cerrar), la salida se descarta
    final_marker: str | None = None
    reasoning_start: str | None = None
    tier: str | None = None   # "chico" (2.6–9.4B densos) | "grande" (14–34B densos) | None (MoE): análisis por tamaño


# Verificado en la API de Hugging Face el 2026-10-01 (repo, gated, licencia, arquitectura, parámetros).
_C, _G = "chico", "grande"
PANEL: list[Generator] = [
    Generator("mistral7b", "mistralai/Mistral-7B-Instruct-v0.3", "Mistral AI", "primario", "apache-2.0", False, 7.25,
              tier=_C),
    Generator("phi4mini", "microsoft/Phi-4-mini-instruct", "Microsoft (Phi)", "primario", "mit", False, 3.84, tier=_C),
    Generator("olmo2_7b", "allenai/OLMo-2-1124-7B-Instruct", "AI2 (OLMo)", "primario", "apache-2.0", False, 7.30,
              tier=_C),
    Generator("granite33_8b", "ibm-granite/granite-3.3-8b-instruct", "IBM (Granite)", "primario", "apache-2.0",
              False, 8.17, tier=_C),
    Generator("falcon3_7b", "tiiuae/Falcon3-7B-Instruct", "TII (Falcon)", "primario", "other (Falcon)", False, 7.46,
              tier=_C),
    Generator("yi15_9b", "01-ai/Yi-1.5-9B-Chat", "01.AI (Yi)", "primario", "apache-2.0", False, 8.83, tier=_C),
    Generator("glm4_9b", "zai-org/glm-4-9b-chat-hf", "Zhipu / Z.ai (GLM)", "primario", "other (glm-4)", False, 9.40,
              tier=_C),
    Generator("smollm3_3b", "HuggingFaceTB/SmolLM3-3B", "Hugging Face (SmolLM)", "primario", "apache-2.0", False,
              3.08, chat_kwargs={"enable_thinking": False}, tier=_C),
    Generator("apertus8b", "swiss-ai/Apertus-8B-Instruct-2509", "Swiss AI (Apertus)", "primario", "apache-2.0",
              False, 8.05, min_transformers="4.56", tier=_C),
    Generator("lfm2_2b6", "LiquidAI/LFM2-2.6B", "Liquid AI (LFM2)", "primario", "other (lfm1.0)", False, 2.57,
              min_transformers="4.55", tier=_C),
    Generator("gptoss20b", "openai/gpt-oss-20b", "OpenAI (gpt-oss)", "primario", "apache-2.0", False, 20.9,
              chat_kwargs={"reasoning_effort": "low"}, harmony=True, min_transformers="4.55", dtype="bfloat16",
              note="MoE con pesos MXFP4 (se des-cuantizan a bf16, ~42 GB); razonamiento bajo; sólo el canal final",
              model_kwargs={"dequantize_mxfp4": True}),
    Generator("llama31_8b", "meta-llama/Llama-3.1-8B-Instruct", "Meta (Llama)", "primario", "llama3.1", True, 8.03,
              tier=_C),
    Generator("aya8b", "CohereLabs/aya-expanse-8b", "Cohere (Aya)", "primario", "cc-by-nc-4.0", True, 8.03, tier=_C),
    Generator("gemma2_9b", "google/gemma-2-9b-it", "Google (Gemma ≈ Gemini)", "control_gemini", "gemma", True, 9.24,
              note="misma casa que Gemini: ¿el atajo de T-gem es estilo Gemini?",
              model_kwargs={"attn_implementation": "eager"}, tier=_C),   # Gemma 2: soft-capping (recomendado: eager)
    Generator("qwen25_7b", "Qwen/Qwen2.5-7B-Instruct", "Alibaba (Qwen)", "control_qwen", "apache-2.0", False, 7.62,
              note="familia del oro y del encoder", tier=_C),
    Generator("qwen3_30b", "Qwen/Qwen3-30B-A3B", "Alibaba (Qwen3)", "control_oro", "apache-2.0", False, 30.5,
              chat_kwargs={"enable_thinking": False}, n_items=1000,
              note="el mismo modelo que escribió el oro (MoE: lento en transformers)"),
    # --- Ampliación del 2026-10-01 (ENMIENDA_03 §10; decidida antes de generar nada): los modelos más fuertes de
    # cada familia que caben en la GPU (96 GB) en bf16 y cargan con transformers. Verificados en la API de HF.
    # La celda 6.1 instala transformers >= PANEL_MIN_TRANSFORMERS (Gemma 4 lo necesita; los demás, menos).
    Generator("phi4_14b", "microsoft/phi-4", "Microsoft (Phi)", "primario", "mit", False, 14.66, tier=_G),
    Generator("mistral_small24b", "mistralai/Mistral-Small-24B-Instruct-2501", "Mistral AI", "primario", "apache-2.0",
              False, 23.57, tier=_G,
              note="Mistral Small 3.2 (2506) no trae tokenizer de transformers; 2501 es sólo texto y sí lo trae"),
    Generator("olmo31_32b", "allenai/Olmo-3.1-32B-Instruct", "AI2 (OLMo)", "primario", "apache-2.0", False, 32.23,
              tier=_G),
    Generator("glm4_32b", "zai-org/GLM-4-32B-0414", "Zhipu / Z.ai (GLM)", "primario", "mit", False, 32.57, tier=_G),
    Generator("yi15_34b", "01-ai/Yi-1.5-34B-Chat", "01.AI (Yi)", "primario", "apache-2.0", False, 34.39, tier=_G),
    Generator("aya32b", "CohereLabs/aya-expanse-32b", "Cohere (Aya)", "primario", "cc-by-nc-4.0", True, 32.30,
              tier=_G),
    Generator("gemma4_31b", "google/gemma-4-31B-it", "Google (Gemma 4 ≈ Gemini)", "control_gemini", "apache-2.0",
              False, 31.27, chat_kwargs={"enable_thinking": False}, min_transformers="5.5", tier=_G,
              final_marker="<channel|>", reasoning_start="<|channel>",
              note="control de Gemini actual; sin modo de pensamiento (la plantilla deja el bloque de pensamiento vacío)"),
]
PANEL_BY_KEY = {g.key: g for g in PANEL}
PANEL_MIN_TRANSFORMERS = "5.5"
MACRO_ROLES = ("primario",)
ROLE_TXT = {"primario": "primario", "control_gemini": "control (≈ Gemini)", "control_qwen": "control (familia Qwen)",
            "control_oro": "control (modelo del oro)", "sintetico": "sintético (smoke)"}


N_ORIGINAL = 16      # los 16 de la primera versión del panel; los siguientes son la ampliación (§10)


def resolve_keys(spec: str | list | None) -> list[str]:
    """'todos' | 'originales' | 'grandes' | 'sin_gated' | 'primarios' | 'controles' | claves separadas por coma."""
    if spec is None or spec == "" or spec == "ninguno":
        return []
    if isinstance(spec, (list, tuple)):
        keys = list(spec)
    elif spec == "todos":
        keys = [g.key for g in PANEL]
    elif spec == "sin_gated":
        keys = [g.key for g in PANEL if not g.gated]
    elif spec == "primarios":
        keys = [g.key for g in PANEL if g.role == "primario"]
    elif spec == "controles":
        keys = [g.key for g in PANEL if g.role != "primario"]
    elif spec == "originales":
        keys = [g.key for g in PANEL[:N_ORIGINAL]]
    elif spec == "grandes":
        keys = [g.key for g in PANEL[N_ORIGINAL:]]
    else:
        keys = [k.strip() for k in str(spec).split(",") if k.strip()]
    bad = [k for k in keys if k not in PANEL_BY_KEY and not k.startswith("syn_")]
    if bad:
        raise KeyError(f"generadores desconocidos {bad}; opciones: {sorted(PANEL_BY_KEY)}")
    return keys


# --------------------------------------------------------------------------------------
# Rutas
def tgen_base(study) -> str:
    return study.paths.d("data", "eval", "tgen")


def tgen_root(study, split: str = "test") -> str:
    return study.paths.d("data", "eval", "tgen", split)


def gen_dir(study, key: str, split: str = "test") -> str:
    """Generaciones de un generador (raw*.jsonl, gen_info*.json) y su set construido para el panel."""
    return os.path.join(tgen_root(study, split), key)


def set_dir(study, key: str, split: str = "test", sub: str | None = None) -> str:
    """Set construido y embebido: el del panel (sub=None, en gen_dir) o uno del piloto (sub="pilot_<B>")."""
    return gen_dir(study, key, split) if sub is None else os.path.join(gen_dir(study, key, split), sub)


def panel_path(study, split: str = "test") -> str:
    return os.path.join(tgen_root(study, split), "PANEL.json")


def build_path(study) -> str:
    """Construcción elegida por el piloto (vale para test y confirm)."""
    return os.path.join(tgen_base(study), "BUILD.json")


def active_build(study) -> str:
    """Construcción del panel: la que fijó el piloto; "S" si todavía no hay piloto (perfil smoke y pruebas)."""
    b = (load_json(build_path(study)) or {}).get("build")
    return b if b in BUILDS else "S"


def task_info(study, key: str, split: str = "test", task: str = "neg") -> dict:
    return load_json(os.path.join(gen_dir(study, key, split), TASKS[task]["info"])) or {}


def build_gen_info(study, key: str, split: str = "test", build: str | None = None) -> dict:
    """Estado de generación que necesita una construcción: "generado" sólo si todas sus tareas terminaron.
    Conserva los metadatos de la primera tarea (repo, familia, rol, tamaño, revisión, tiempos)."""
    build = build or active_build(study)
    infos = [task_info(study, key, split, t) for t in BUILDS[build]]
    out = dict(infos[0])
    st = [i.get("status") for i in infos]
    if all(x == "generado" for x in st):
        out["status"] = "generado"
    else:
        out["status"] = next((x for x in st if x != "generado"), None) or "no_generado"
    out["n_target"] = min(int(i.get("n_target") or i.get("items_requested") or 0) for i in infos)
    out["seconds_total"] = float(sum(float(i.get("seconds_total", 0.0)) for i in infos))
    out["tasks"] = {t: i.get("status") for t, i in zip(BUILDS[build], infos)}
    return out


# --------------------------------------------------------------------------------------
# Prompt y parseo
def n_words(text: str) -> int:
    return max(1, len(str(text).split()))


def build_prompt(question: str, answer: str, k: int = K_REQUEST, task: str = "neg", words: int | None = None) -> str:
    """Prompt de una tarea. `words` = largo pedido (por defecto, el del oro; en "para", el largo asignado)."""
    return TASK_PROMPT[task].format(question=question.strip(), answer=answer.strip(), k=k,
                                    n_words=int(words) if words else n_words(answer))


def max_new_tokens_for(gold_tokens: int, k: int = K_REQUEST, harmony: bool = False) -> int:
    """Presupuesto de generación: k respuestas de ~1.6× el largo del oro + holgura (cada tokenizer cuenta
    distinto). Con razonamiento (gpt-oss) se suma espacio para el canal de análisis."""
    per = int(1.6 * max(8, int(gold_tokens))) + 8
    m = int(min(3072, max(256, 64 + k * per)))
    return m + (1024 if harmony else 0)


# Numeración y viñetas sólo si van seguidas de espacio: así no se tocan respuestas como "2.5 billion",
# "12.5%", "-40 degrees" o "3:00 pm" (la revisión independiente encontró ese error en la primera versión).
_PREFIX = re.compile(r"^\s*(?:[-*•·]+\s+|\(?\d{1,2}[.)\]]\s+|(?:incorrect\s+)?answer\s*\d*\s*[:.)\-]\s*)",
                     re.IGNORECASE)
# si el oro mismo empieza con "Answer:", no se le quita ese prefijo a los negativos (sería una diferencia de estilo)
_PREFIX_NO_ANSWER = re.compile(r"^\s*(?:[-*•·]+\s+|\(?\d{1,2}[.)\]]\s+|incorrect\s+answer\s*\d*\s*[:.)\-]\s*|"
                               r"answer\s*\d+\s*[:.)\-]\s*)", re.IGNORECASE)
_PREAMBLE = re.compile(r"\b(here (are|is)|below (are|is)|the following)\b.*\b(answers?|options|responses|lines|"
                       r"rewrites?|rephrasings?|paraphrases?|versions?|edits?)\b", re.IGNORECASE)


def _is_preamble(s: str) -> bool:
    """Encabezados como "Here are 14 incorrect answers:" (no respuestas que empiecen con "Sure" u "OK")."""
    w = len(s.split())
    if s.endswith(":") and w <= 15:
        return True
    return w <= 15 and bool(_PREAMBLE.search(s))


def harmony_final(text: str) -> str:
    """Canal final de una salida en formato harmony (gpt-oss). Vacío si la generación no llegó a él."""
    tag = "<|channel|>final<|message|>"
    j = text.rfind(tag)
    if j < 0:
        return ""
    out = text[j + len(tag):]
    for stop in ("<|return|>", "<|end|>", "<|endoftext|>", "<|start|>"):
        k = out.find(stop)
        if k >= 0:
            out = out[:k]
    return out


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .;:")


def parse_negatives(text: str, gold: str, k: int = K_REQUEST, harmony: bool = False) -> list[str]:
    """Una respuesta por línea: quita numeración, viñetas, comillas y preámbulos; descarta vacías,
    copias del oro y duplicados. Devuelve como máximo k, en el orden generado."""
    if harmony:
        text = harmony_final(text)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    g = _norm(gold)
    rx = _PREFIX_NO_ANSWER if re.match(r"^\s*answer\s*[:.)\-]", str(gold), re.IGNORECASE) else _PREFIX
    seen, out = set(), []
    for line in text.splitlines():
        s = line.strip()
        for _ in range(3):                     # "• Answer 4: ..." -> dos prefijos seguidos
            t = rx.sub("", s, count=1).strip()
            if t == s:
                break
            s = t
        s = re.sub(r"[\x00-\x08\x0b-\x1f\x7f�]", "", s)       # caracteres de control / bytes inválidos
        s = s.strip().strip('"“”\'`').strip()
        if not s or len(s) < 2 or len(s) > 4000:
            continue
        if _is_preamble(s):
            continue
        low = _norm(s)
        if not low or low == g or low in seen:
            continue
        seen.add(low)
        out.append(s)
        if len(out) >= k:
            break
    return out


# --------------------------------------------------------------------------------------
# 1. Ítems y textos
def panel_order(study, test, split: str = "test", seed: int = 0) -> np.ndarray:
    """Orden fijo de ítems elegibles: ≤ 1 por documento, al azar (semilla 0). Cada generador usa un prefijo."""
    p = os.path.join(tgen_root(study, split), "panel_order.npy")
    if os.path.exists(p):
        return np.load(p)
    order = np.random.default_rng(seed).permutation(test.n)
    seen, pick = set(), []
    for i in order:
        d = test.doc_ids[i]
        if d in seen:
            continue
        seen.add(d)
        pick.append(int(i))
    arr = np.asarray(pick, np.int64)
    atomic_np_save(arr, p)
    return arr


def hf_decoder(model: str = "Qwen/Qwen3-8B"):
    """Decodifica token ids de Qwen3 (los tokens públicos de Nemotron-DQA) igual que texts.decode_items."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    return lambda ids: tok.decode(ids)


def prepare_texts(study, test, n_items: int, split: str = "test", decoder=None, lookup_fn=None, log=print) -> list:
    """Pregunta y oro (texto) de los primeros n_items del orden del panel. Se guarda en items_text.jsonl
    (reanudable: sólo agrega los que faltan)."""
    root = makedirs(tgen_root(study, split))
    order = panel_order(study, test, split)
    want = [int(i) for i in order[:n_items]]
    path = os.path.join(root, "items_text.jsonl")
    have = {r["i"]: r for r in read_jsonl(path)}
    todo = [i for i in want if i not in have]
    if todo:
        if lookup_fn is None:
            from .texts import lookup as lookup_fn
        decoder = decoder or hf_decoder()
        docs = {test.doc_ids[i] for i in todo}
        log(f"[tgen] textos de {len(todo)} ítems ({len(docs)} documentos)")
        qa = {(r["doc_id"], int(r["pair_idx"])): r for r in
              lookup_fn(study.paths, docs, "qa", ["doc_id", "pair_idx", "question_ids", "answer_ids"])}
        for i in todo:
            r = qa.get((test.doc_ids[i], int(test.pair_idx[i])))
            if r is None:
                rec = {"i": i, "missing": True}
            else:
                rec = {"i": i, "doc_id": test.doc_ids[i], "pair_idx": int(test.pair_idx[i]),
                       "question": decoder(r["question_ids"]), "answer": decoder(r["answer_ids"]),
                       "gold_tokens": int(len(r["answer_ids"]))}
            append_jsonl(path, rec)
            have[i] = rec
    return [have[i] for i in want if not have[i].get("missing")]


# --------------------------------------------------------------------------------------
# 2. Generación (subproceso por generador)
def check_access(repo: str, token: str | None = None) -> tuple[bool, str]:
    """¿Se puede descargar el modelo? (gated sin licencia aceptada o sin token -> False)."""
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token or os.environ.get("HF_TOKEN") or None)
        info = api.model_info(repo)
        if getattr(info, "gated", False):
            # gated: listar archivos puede funcionar sin haber aceptado la licencia; auth_check no
            try:
                if hasattr(api, "auth_check"):
                    api.auth_check(repo)
                else:
                    from huggingface_hub import hf_hub_download
                    hf_hub_download(repo, "config.json", token=api.token, local_dir=os.path.join("/tmp", "tgen_auth", repo.replace("/", "__")))
            except Exception as e:  # noqa: BLE001
                return False, f"gated sin acceso ({type(e).__name__}); acepta la licencia en HF y usa HF_TOKEN"
        return True, f"ok (sha {str(getattr(info, 'sha', ''))[:10]})"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:200]}"


def model_revision(repo: str, token: str | None = None) -> str | None:
    try:
        from huggingface_hub import HfApi
        return HfApi(token=token or os.environ.get("HF_TOKEN") or None).model_info(repo).sha
    except Exception:  # noqa: BLE001
        return None


def write_prompts(path: str, items: list, gen: Generator, k: int = K_REQUEST, task: str = "neg",
                  targets: dict | None = None) -> int:
    """Un prompt por ítem. En "para" sólo los ítems con largo asignado (`targets`, ver para_targets)."""
    makedirs(os.path.dirname(path))
    tmp = path + ".tmp"
    n = 0
    with open(tmp, "w") as f:
        for it in items:
            words, gold_tok = None, int(it.get("gold_tokens", 32))
            if task == "para":
                tg = (targets or {}).get(it["i"])
                if tg is None:
                    continue
                words = int(tg["target_words"])
                gold_tok = max(gold_tok, int(1.6 * words) + 4)      # reescrituras más largas que el oro
            f.write(json.dumps({"i": it["i"], "prompt": build_prompt(it["question"], it["answer"], k, task, words),
                                "gold": it["answer"],
                                "max_new_tokens": max_new_tokens_for(gold_tok, k, gen.harmony)}) + "\n")
            n += 1
    os.replace(tmp, path)
    return n


def raw_status(raw_path: str, k_keep: int = K_KEEP) -> dict:
    """Por ítem: intentos hechos y la mayor cantidad de respuestas parseadas."""
    att, best = {}, {}
    for r in read_jsonl(raw_path):
        i = r["i"]
        att[i] = att.get(i, 0) + 1
        best[i] = max(best.get(i, 0), len(r.get("negatives", [])))
    return {"attempts": att, "best": best, "complete": sum(1 for v in best.values() if v >= k_keep)}


def _dir_bytes(d: str) -> int:
    """Bytes reales de una carpeta: los enlaces simbólicos (snapshots/ -> blobs/ en la caché de HF) no se cuentan
    dos veces (hasta v0.3.5 el mensaje "borrados X GB" daba el doble)."""
    tot = 0
    for r, _, fs in os.walk(d):
        for f in fs:
            try:
                st = os.lstat(os.path.join(r, f))
            except OSError:
                continue
            if not os.path.islink(os.path.join(r, f)):
                tot += st.st_size
    return tot


def free_gb(path: str) -> float:
    """GB libres en el disco de `path` (o de su carpeta existente más cercana)."""
    p = os.path.abspath(path)
    while not os.path.exists(p) and os.path.dirname(p) != p:
        p = os.path.dirname(p)
    return shutil.disk_usage(p).free / 1e9


SHARED_BLOBS = os.path.join("hub", "blobs")     # caché compartida de huggingface_hub (archivos reales)


def _link_targets(d: str) -> set:
    """Destinos reales, fuera de `d`, de los enlaces simbólicos que hay dentro de `d`."""
    base = os.path.realpath(d) + os.sep
    out = set()
    for r, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(r, f)
            if os.path.islink(p):
                t = os.path.realpath(p)
                if not t.startswith(base):
                    out.add(t)
    return out


def _remove_shared(paths) -> int:
    """Borra archivos de la caché compartida (con sus .refs y .lock). Devuelve los bytes borrados."""
    tot = 0
    for t in paths:
        for f in (t, t + ".refs", t + ".lock"):
            try:
                if os.path.isfile(f) and not os.path.islink(f):
                    tot += os.path.getsize(f)
                    os.remove(f)
            except OSError:
                pass
    return tot


def purge_model_cache(repo: str, hf_home: str) -> float:
    """Borra los pesos descargados de un generador (cada uno ocupa 5–65 GB del disco local) y devuelve los GB reales
    borrados. Con la caché compartida de huggingface_hub («shared blobs», 2026) los archivos reales están en
    hub/blobs/<xx>/<hash> y los de models--X/blobs son enlaces simbólicos: por eso hasta v0.3.6 se borraban sólo
    los enlaces y el disco no se liberaba (rev. 4.6: "borrados 0.0 GB"). Ahora se borran también sus destinos en
    la caché compartida, salvo los que use otro repo."""
    hub = os.path.join(hf_home, "hub")
    d = os.path.join(hub, "models--" + repo.replace("/", "--"))
    if not os.path.isdir(d):
        return 0.0
    store = os.path.realpath(os.path.join(hf_home, SHARED_BLOBS)) + os.sep
    mine = {t for t in _link_targets(d) if t.startswith(store)}
    others = set()
    for name in os.listdir(hub):
        q = os.path.join(hub, name)
        if q != d and "--" in name and os.path.isdir(q):
            others |= _link_targets(q)
    size = _dir_bytes(d)
    shutil.rmtree(d, ignore_errors=True)
    size += _remove_shared(mine - others)
    return size / 1e9


def purge_hf_caches(hf_home: str, keep_repos=()) -> dict:
    """Libera el disco antes de bajar un generador: borra los pesos de todos los generadores que queden en
    `hf_home/hub` salvo `keep_repos`, la caché compartida de huggingface_hub (`hub/blobs`, salvo lo que usen los
    repos que se conservan) y la caché de chunks de Xet (`hf_home/xet`). Devuelve lo borrado y el espacio libre
    antes y después, medido en el disco (no por tamaño de archivos)."""
    keep = {"models--" + r.replace("/", "--") for r in keep_repos}
    before = free_gb(hf_home)
    removed = []
    hub = os.path.join(hf_home, "hub")
    if os.path.isdir(hub):
        for name in sorted(os.listdir(hub)):
            if name.startswith("models--") and name not in keep:
                shutil.rmtree(os.path.join(hub, name), ignore_errors=True)
                removed.append(name[len("models--"):].replace("--", "/"))
        store = os.path.join(hf_home, SHARED_BLOBS)
        if os.path.isdir(store):
            used = set()
            for k in keep:
                if os.path.isdir(os.path.join(hub, k)):
                    used |= _link_targets(os.path.join(hub, k))
            files = set()
            for r, _, fs in os.walk(store):
                for f in fs:
                    p = os.path.realpath(os.path.join(r, f))
                    if not p.endswith((".refs", ".lock")) and p not in used:
                        files.add(p)
            gb = _remove_shared(files) / 1e9
            if files:
                removed.append(f"hub/blobs (caché compartida, {gb:.1f} GB)")
    xet = os.path.join(hf_home, "xet")
    if os.path.isdir(xet):
        shutil.rmtree(xet, ignore_errors=True)
        removed.append("xet (caché de chunks)")
    return {"removed": removed, "free_before_gb": round(before, 1), "free_after_gb": round(free_gb(hf_home), 1)}


def worker_env(hf_home: str, pkg_root: str, base: dict | None = None) -> dict:
    """Entorno del subproceso que genera: su propia caché de HF, sin caché de chunks de Xet y sin la caché
    compartida de huggingface_hub (así los pesos de cada generador quedan dentro de models--X y se borran enteros)."""
    env = dict(os.environ if base is None else base)
    env["PYTHONPATH"] = pkg_root + os.pathsep + env.get("PYTHONPATH", "")
    env["HF_HOME"] = hf_home
    env.pop("HF_HUB_CACHE", None)                   # la caché tiene que quedar dentro de hf_home
    env.pop("HUGGINGFACE_HUB_CACHE", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("HF_XET_CHUNK_CACHE_SIZE_BYTES", "0")          # sin caché de chunks: cada modelo se baja una vez
    env["HF_HUB_DISABLE_SHARED_BLOBS"] = "1"                       # v0.3.7: pesos dentro de models--X
    return env


def deleted_open_files(min_mb: float = 100.0) -> list:
    """Archivos borrados que algún proceso todavía tiene abiertos (ocupan disco hasta que el proceso los cierra).
    Sólo diagnóstico: lee /proc; no mata nada. [(pid, comando, GB)] ordenado de mayor a menor."""
    out = {}
    for pid in (x for x in os.listdir("/proc") if x.isdigit()) if os.path.isdir("/proc") else []:
        fd_dir = f"/proc/{pid}/fd"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            continue
        seen = set()
        for fd in fds:
            try:
                target = os.readlink(os.path.join(fd_dir, fd))
                if not target.endswith(" (deleted)") or target in seen:
                    continue
                seen.add(target)
                size = os.stat(os.path.join(fd_dir, fd)).st_size
            except OSError:
                continue
            out[pid] = out.get(pid, 0) + size
    rows = []
    for pid, b in out.items():
        if b / 1e6 < min_mb:
            continue
        try:
            cmd = open(f"/proc/{pid}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace")[:120]
        except OSError:
            cmd = "?"
        rows.append((int(pid), cmd.strip(), round(b / 1e9, 3)))
    return sorted(rows, key=lambda r: -r[2])


def disk_diagnostics(roots=("/content", "/root/.cache", "/tmp"), top: int = 12) -> list:
    """Qué ocupa el disco local: las carpetas más grandes (dos niveles bajo cada raíz) y los archivos borrados que
    siguen abiertos. Para entender un "No space left on device" la próxima vez, sin adivinar."""
    sizes = []
    for root in roots:
        if not os.path.isdir(root):
            continue
        for a in sorted(os.listdir(root)):
            pa = os.path.join(root, a)
            if os.path.islink(pa) or not os.path.isdir(pa) or pa == "/content/drive":
                continue
            subs = [os.path.join(pa, b) for b in os.listdir(pa)] if os.access(pa, os.R_OK) else []
            subs = [x for x in subs if os.path.isdir(x) and not os.path.islink(x)]
            for x in (subs or [pa]):
                try:
                    sizes.append((_dir_bytes(x) / 1e9, x))
                except OSError:
                    pass
    lines = [f"{gb:7.1f} GB  {path}" for gb, path in sorted(sizes, reverse=True)[:top] if gb >= 0.5]
    for root in roots:                              # cachés compartidas de huggingface_hub (archivos reales)
        for r, ds, _ in os.walk(root) if os.path.isdir(root) else []:
            if r.count(os.sep) - root.count(os.sep) >= 4 or r.startswith("/content/drive"):
                ds[:] = []
                continue
            if os.path.basename(r) == "hub" and "blobs" in ds:
                gb = _dir_bytes(os.path.join(r, "blobs")) / 1e9
                if gb >= 0.5:
                    lines.append(f"{gb:7.1f} GB  {os.path.join(r, 'blobs')} (caché compartida de huggingface_hub)")
    for pid, cmd, gb in deleted_open_files():
        lines.append(f"{gb:7.1f} GB  borrados pero abiertos por el proceso {pid}: {cmd}")
    return lines


LOW_DISK_GB = 40.0


def _tf_version() -> str | None:
    try:
        from importlib.metadata import version as _v
        return _v("transformers")
    except Exception:  # noqa: BLE001
        return None


def installed_transformers_ok(minimum: str | None) -> str | None:
    """None si la versión instalada de transformers alcanza `minimum`; si no, el motivo (para gen_info)."""
    if not minimum:
        return None
    try:
        from importlib.metadata import version as _v
        from packaging.version import Version
        have = _v("transformers")
        if Version(have) < Version(minimum):
            return (f"transformers {have} < {minimum}, que este modelo necesita: corre la celda 6.1 completa "
                    f"(instala transformers>={PANEL_MIN_TRANSFORMERS}) o reinicia el entorno")
    except Exception:  # noqa: BLE001
        return None
    return None


def gen_n(key: str, n_items: int) -> int:
    """Ítems pedidos a un generador: los n_items del panel, o menos para los controles pesados."""
    g = PANEL_BY_KEY.get(key)
    return int(min(g.n_items, n_items)) if g is not None and g.n_items else int(n_items)


def run_generator(study, test, key: str, split: str = "test", n_items: int = PANEL_N, k: int | None = None,
                  task: str = "neg", batch: int = 64, backend: str = "hf", temperature: float = 0.8,
                  top_p: float = 0.95, max_attempts: int = 2, seed: int = 0, python: str | None = None,
                  hf_home: str | None = None, purge: bool = True, check: bool = True, decoder=None, lookup_fn=None,
                  log=print) -> dict:
    """Genera (o reanuda) una tarea de un generador ("neg", "edit" o "para") sobre los primeros n_items del
    orden del panel, en un subproceso. Devuelve su gen_info. "para" necesita "neg" terminada en esos ítems."""
    gen = PANEL_BY_KEY[key]
    T = TASKS[task]
    k = int(k or T["k"])
    gd = makedirs(gen_dir(study, key, split))
    n = gen_n(key, n_items)
    raw = os.path.join(gd, T["raw"])
    info_p = os.path.join(gd, T["info"])
    info = load_json(info_p, {}) or {}
    st = raw_status(raw, T["k_keep"])
    items = prepare_texts(study, test, n, split, decoder=decoder, lookup_fn=lookup_fn, log=log)
    targets = None
    if task == "para":
        ni = task_info(study, key, split, "neg")
        if ni.get("status") != "generado" or int(ni.get("n_target") or ni.get("items_requested") or 0) < n:
            log(f"[tgen:{key}] reescrituras: faltan los negativos de esos {n} ítems (tarea neg)")
            info.update(status="sin_negativos", task=task, checked=now_iso())
            save_json(info, info_p)
            return info
        targets = para_targets(study, key, split, items)
        save_json({str(i): v for i, v in targets.items()}, os.path.join(gd, "para_targets.json"))
        items = [it for it in items if it["i"] in targets]
    # n_target = el prefijo más largo del orden del panel ya generado completo (un piloto con menos ítems no
    # lo achica: los ítems ya generados no se vuelven a pedir)
    prev_n = int(info.get("n_target") or (info.get("items_requested") if info.get("status") == "generado" else 0) or 0)
    base = dict(task=task, prompt_version=T["version"], n_target=max(prev_n, n))
    pending = [it for it in items if st["best"].get(it["i"], 0) < T["k_keep"]
               and st["attempts"].get(it["i"], 0) < max_attempts]
    if not pending:
        log(f"[tgen:{key}] {task}: ya generado ({st['complete']}/{len(items)} ítems con ≥ {T['k_keep']} respuestas)")
        info.update(status="generado", items_complete_prefix=st["complete"], **base)
        info.setdefault("items_requested", len(items))
        save_json(info, info_p)
        return info
    if backend == "hf":
        need = installed_transformers_ok(gen.min_transformers)
        if need:
            log(f"[tgen:{key}] se omite: {need}")
            info.update(status="sin_soporte", motivo=need, checked=now_iso(), task=task)
            save_json(info, info_p)
            return info
    if check and backend != "fake":
        ok, why = check_access(gen.repo)
        if not ok:
            log(f"[tgen:{key}] se omite: {why}")
            info.update(status="sin_acceso", motivo=why, checked=now_iso(), task=task)
            save_json(info, info_p)
            return info
    hf_home = hf_home or os.environ.get("HF_HOME") or os.path.join(study.paths.cache_root, "hf_home")
    if backend != "fake":
        fr = purge_hf_caches(hf_home, keep_repos=(gen.repo,))       # pesos de otros generadores y caché de Xet
        if fr["removed"]:
            log(f"[tgen:{key}] disco: se borraron {', '.join(fr['removed'])} · libres {fr['free_before_gb']} -> "
                f"{fr['free_after_gb']} GB")
        else:
            log(f"[tgen:{key}] disco: {fr['free_after_gb']} GB libres")
        if fr["free_after_gb"] < LOW_DISK_GB:
            log(f"[tgen:{key}] ATENCIÓN: poco disco libre ({fr['free_after_gb']} GB). Lo que más ocupa:\n  "
                + "\n  ".join(disk_diagnostics()))
    prompts_p = os.path.join(study.paths.cache_root, "tgen_prompts", split, f"{key}_{task}.jsonl")
    write_prompts(prompts_p, items, gen, k, task, targets)
    cmd = [python or sys.executable, "-m", "clm_losses.genworker", "generate", "--model", gen.repo,
           "--prompts", prompts_p, "--out", raw, "--backend", backend, "--batch", str(batch),
           "--temperature", str(temperature), "--top-p", str(top_p), "--k", str(k), "--k-keep", str(T["k_keep"]),
           "--max-attempts", str(max_attempts), "--seed", str(seed), "--dtype", gen.dtype,
           "--chat-kwargs", json.dumps(gen.chat_kwargs), "--model-kwargs", json.dumps(gen.model_kwargs)] \
        + (["--harmony"] if gen.harmony else []) \
        + (["--final-marker", gen.final_marker] if gen.final_marker else []) \
        + (["--reasoning-start", gen.reasoning_start] if gen.reasoning_start else [])
    env = worker_env(hf_home, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    log(f"[tgen:{key}] {gen.repo} · {task}: {len(pending)} ítems pendientes de {len(items)} (backend {backend}, "
        f"batch {batch})")
    t0 = time.time()
    tail = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env, bufsize=1,
                            start_new_session=True)
    for line in proc.stdout:
        line = line.rstrip()
        tail = (tail + [line])[-40:]
        if line.startswith("[genworker]"):
            log(line.replace("[genworker]", f"[tgen:{key}:{task}]"))
    rc = proc.wait()
    try:                                            # nada del subproceso queda vivo con los pesos abiertos
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    secs = time.time() - t0
    st = raw_status(raw, T["k_keep"])
    info.update(key=key, repo=gen.repo, revision=model_revision(gen.repo) if backend != "fake" else None,
                family=gen.family, role=gen.role, license=gen.license, prompt=TASK_PROMPT[task],
                k_request=k, k_keep=T["k_keep"], temperature=temperature, top_p=top_p,
                max_attempts=max_attempts, seed=seed, backend=backend, dtype=gen.dtype, chat_kwargs=gen.chat_kwargs,
                final_marker=gen.final_marker, tier=gen.tier, transformers=_tf_version(),
                items_requested=len(items), items_complete=st["complete"], updated=now_iso(),
                seconds_last_call=round(secs, 1), seconds_total=round(float(info.get("seconds_total", 0.0)) + secs, 1),
                **base)
    if rc != 0:
        info.update(status="fallido", error_tail=tail[-15:], n_target=prev_n)
        log(f"[tgen:{key}] {task} FALLÓ (código {rc}). Últimas líneas:\n  " + "\n  ".join(tail[-15:]))
    else:
        info.update(status="generado")
        log(f"[tgen:{key}] {task} listo en {secs / 60:.1f} min: {st['complete']}/{len(items)} ítems con "
            f"≥ {T['k_keep']} respuestas")
    save_json(info, info_p)
    if purge and backend != "fake":
        gb = purge_model_cache(gen.repo, hf_home)
        if gb:
            log(f"[tgen:{key}] borrados {gb:.1f} GB de pesos del disco local · libres {free_gb(hf_home):.1f} GB")
    return info


def run_tasks(study, test, key: str, tasks, split: str = "test", n_items: int = PANEL_N, purge: bool = True,
              stop_on_fail: bool = False, log=print, **kw) -> dict:
    """Corre varias tareas de un generador seguidas; los pesos se borran una sola vez, al final. "para" sola
    se niega si "neg" no terminó. Devuelve {tarea: estado}."""
    out = {}
    for t in tasks:
        info = run_generator(study, test, key, split, n_items=n_items, task=t, purge=False, log=log, **kw)
        out[t] = info.get("status")
        if out[t] != "generado" and stop_on_fail:
            break
    if purge and kw.get("backend", "hf") != "fake":
        hf_home = kw.get("hf_home") or os.environ.get("HF_HOME") or os.path.join(study.paths.cache_root, "hf_home")
        gb = purge_model_cache(PANEL_BY_KEY[key].repo, hf_home)
        if gb:
            log(f"[tgen:{key}] borrados {gb:.1f} GB de pesos del disco local · libres {free_gb(hf_home):.1f} GB")
    return out


# --------------------------------------------------------------------------------------
# 3. Construcción de los sets (ENMIENDA_03 §11) y embebido
#
# La v1 (2026-10-01) dejó atajos en los 16 generadores originales (un modelo que no ve el estado encontraba el
# oro el 77–91 % de las veces): el oro suele terminar en punto y los negativos no; hay ítems de sí/no y de opción
# múltiple, que no admiten 10 respuestas incorrectas; el filtro de coseno > 0.95 descartaba sobre todo las
# respuestas que imitaban el formato del oro, sin atrapar falsos negativos reales ("6 pages" con oro "6"); y el
# oro es más largo y más específico que lo que un modelo inventa, aun cuando el generador es el mismo modelo que
# escribió el oro (Qwen3-30B-A3B). Las reglas de abajo se aplican igual a todos los generadores y no miran ningún
# resultado de modelos entrenados; el piloto (§5b) elige la construcción con una regla fija.
PANEL_RULES = "v3"
ITEM_KINDS = ("ok", "si_no", "opcion_multiple", "binaria")
_AUX = r"(?:is|are|was|were|does|do|did|can|could|will|would|should|has|have|had|may|might|must|shall)"
_MC_PAREN = re.compile(r"(?:^|\s)\(?([A-E])\)\s")
_MC_DOT = re.compile(r"(?:^|\s)([A-E])\.\s")
_GOLD_OPT = re.compile(r"^\(?[A-E][).]\s")
_TRAIL_PAREN = re.compile(r"\s+\([^()]*\)\s*$")       # explicación final entre paréntesis (con espacio antes)
_STOP = frozenset("a an the of in on at to for and or with by from as is are was were be been being it its this that "
                  "these those their there which who whom whose".split())


def item_kind(question: str, gold: str) -> str:
    """Tipo de ítem: "si_no" (el oro empieza con yes/no), "opcion_multiple" (opciones con letra en la pregunta,
    o el oro empieza con una letra de opción), "binaria" (pregunta cerrada con "or": sólo hay una alternativa
    incorrecta) u "ok". Sólo los "ok" admiten 10 respuestas incorrectas del mismo estilo que el oro."""
    q, g = str(question or "").strip(), str(gold or "").strip()
    words = g.lower().split()
    first = re.sub(r"[^a-z]", "", words[0]) if words else ""
    if first in ("yes", "no"):
        return "si_no"
    if (len(set(_MC_PAREN.findall(" " + q))) >= 2 or len(set(_MC_DOT.findall(" " + q))) >= 3
            or _GOLD_OPT.match(g)):
        return "opcion_multiple"
    if re.match(rf"^{_AUX}\b", q.lower()) and re.search(r"\bor\b", q.lower()):
        return "binaria"
    return "ok"


def match_format(neg: str, gold: str) -> str | None:
    """Iguala el formato superficial del negativo al del oro (puntuación final y mayúscula inicial) y le quita
    una explicación final entre paréntesis si el oro no usa paréntesis. None si no es una respuesta utilizable
    (vacía, o una pregunta cuando el oro no lo es). Para S y E (el oro original queda tal cual)."""
    s, g = str(neg).strip(), str(gold).strip()
    if not s or not g:
        return None
    if s.endswith("?") and not g.endswith("?"):
        return None
    if "(" not in g:
        s = _TRAIL_PAREN.sub("", s.rstrip(" .;:!,")).strip()
    s = s.rstrip(" .;:!,")
    if not s:
        return None
    if g[-1] in ".!":
        s += g[-1]
    if g[0].isalpha() and s[0].isalpha():
        s = (s[0].upper() if g[0].isupper() else s[0].lower()) + s[1:]
    return s


def canonical(text: str) -> str | None:
    """Formato canónico, el mismo para los 11 candidatos de P (ninguno es el oro original): sin comillas, sin
    explicación final entre paréntesis, sin puntuación final y con mayúscula inicial. None si es una pregunta."""
    s = str(text).strip().strip('"“”\'`').strip()
    if not s or s.endswith("?"):
        return None
    s = s.rstrip(" .;:!,")
    s = (_TRAIL_PAREN.sub("", s).strip() or s).rstrip(" .;:!,")
    if not s:
        return None
    return s[0].upper() + s[1:] if s[0].isalpha() else s


def contains_gold(neg: str, gold: str, max_words: int = 4) -> bool:
    """Falso negativo probable: un oro corto aparece entero dentro del negativo ("6 pages" con oro "6")."""
    g, n = _norm(gold), _norm(neg)
    if not g or len(g.split()) > max_words:
        return False
    return re.search(rf"(?<!\w){re.escape(g)}(?!\w)", n) is not None


def _content(text: str) -> set:
    return {t for t in re.findall(r"[a-z0-9]+", str(text).lower()) if t not in _STOP}


def is_edit_of(neg: str, gold: str) -> bool:
    """Edición válida (construcción E): saca al menos una palabra de contenido del oro y pone al menos una nueva
    (no es el mismo texto con otras palabras vacías ni un fragmento del oro)."""
    a, b = _content(neg), _content(gold)
    return bool(a - b) and bool(b - a)


def pick_by_length(cands: list, gold: str, k: int = K_KEEP) -> list:
    """Los k candidatos de largo (en caracteres) más parecido al del oro; empates por orden de generación."""
    gl = len(str(gold).strip())
    order = sorted(range(len(cands)), key=lambda j: (abs(len(cands[j]) - gl), j))
    return [cands[j] for j in sorted(order[:k])]


def build_candidates(gold: str, recs: list, k: int = K_KEEP, require_edit: bool = False) -> tuple:
    """Negativos de un ítem (S y E) a partir de sus registros de raw*.jsonl: unión de intentos en orden, formato
    igualado al oro, sin preguntas, sin respuestas que contienen al oro, sin duplicados y, en E, sólo ediciones
    válidas; si sobran, los k de largo más parecido al oro. Devuelve (negativos, conteo de descartes)."""
    stats = {"no_respuesta": 0, "contiene_oro": 0, "duplicado": 0, "no_es_edicion": 0}
    seen, out = {_norm(gold)}, []
    for r in sorted(recs, key=lambda r: r.get("attempt", 0)):
        for s in r.get("negatives", []):
            t = match_format(s, gold)
            if t is None:
                stats["no_respuesta"] += 1
                continue
            low = _norm(t)
            if not low or low in seen:                          # copia del oro o repetida
                stats["duplicado"] += 1
                continue
            if contains_gold(t, gold):
                stats["contiene_oro"] += 1
                continue
            if require_edit and not is_edit_of(t, gold):
                stats["no_es_edicion"] += 1
                continue
            seen.add(low)
            out.append(t)
    return (pick_by_length(out, gold, k) if len(out) > k else out), stats


def p_negatives(gold: str, recs: list, k: int = K_KEEP) -> list | None:
    """Los k negativos de la construcción P: los de S, en formato canónico. None si no alcanzan."""
    negs, _ = build_candidates(gold, recs, k)
    out, seen = [], set()
    for x in negs:
        c = canonical(x)
        if c is not None and _norm(c) not in seen:
            seen.add(_norm(c))
            out.append(c)
    return out[:k] if len(out) >= k else None


def _key_seed(key: str) -> int:
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16)


def para_target(key: str, i: int, negs: list, k: int = K_KEEP) -> tuple:
    """Largo asignado al oro reescrito (P): t ~ Uniforme{0..k} negativos quedarán más largos que él (semilla fija
    por generador e ítem), así la longitud no dice cuál es el correcto. Devuelve (t, caracteres, palabras)."""
    rs = np.random.default_rng([_key_seed(key), int(i)])
    t = int(rs.integers(0, k + 1))
    L = sorted(len(x) for x in negs[:k])
    lo = L[k - t - 1] if t < k else 0.6 * L[0]
    hi = L[k - t] if t > 0 else 1.4 * L[-1]
    tc = 0.5 * (lo + hi)
    cpw = sum(len(x) for x in negs[:k]) / max(1, sum(n_words(x) for x in negs[:k]))
    return t, float(tc), max(1, int(round(tc / max(1.0, cpw))))


def para_targets(study, key: str, split: str, items: list, k: int = K_KEEP) -> dict:
    """{ítem: {t, target_chars, target_words}} para los ítems "ok" con k negativos de P."""
    by = _by_item(read_jsonl(os.path.join(gen_dir(study, key, split), TASKS["neg"]["raw"])))
    out = {}
    for it in items:
        i = it["i"]
        if i not in by or item_kind(it.get("question", ""), it["answer"]) != "ok":
            continue
        negs = p_negatives(it["answer"], by[i], k)
        if negs is None:
            continue
        t, tc, tw = para_target(key, i, negs, k)
        out[i] = {"t": t, "target_chars": round(tc, 1), "target_words": tw}
    return out


def choose_positive(paras: list, gold: str, negs: list, t: int, target_chars: float) -> str | None:
    """Entre las reescrituras del oro (formato canónico), la que deja exactamente t negativos más largos o la más
    cercana a eso; empates por cercanía al largo asignado. Nunca una copia del oro ni de un negativo."""
    L = [len(x) for x in negs]
    bad = {_norm(gold)} | {_norm(x) for x in negs}
    best, best_key = None, None
    for s in paras:
        c = canonical(s)
        if c is None or _norm(c) in bad:
            continue
        r = sum(1 for x in L if x > len(c))
        kk = (abs(r - t), abs(len(c) - target_chars))
        if best_key is None or kk < best_key:
            best, best_key = c, kk
    return best


def _by_item(recs: list) -> dict:
    out: dict = {}
    for r in recs:
        out.setdefault(r["i"], []).append(r)
    return out


def construct_sets(study, key: str, build: str, split: str = "test", n_limit: int | None = None,
                   k: int = K_KEEP) -> tuple:
    """Sets de texto de un generador según la construcción: [(ítem, positivo o None, [k negativos])] y
    estadísticas. positivo None = el oro original (su embedding es el del dataset)."""
    from collections import Counter
    gd = gen_dir(study, key, split)
    texts = {r["i"]: r for r in read_jsonl(os.path.join(tgen_root(study, split), "items_text.jsonl"))
             if not r.get("missing")}
    allowed = None
    if n_limit is not None:
        allowed = set(int(x) for x in panel_order_cached(study, split)[:n_limit])
    raws = {t: _by_item(read_jsonl(os.path.join(gd, TASKS[t]["raw"]))) for t in BUILDS[build]}
    base = raws[BUILDS[build][0]]
    items_all = sorted(i for i in base if i in texts and (allowed is None or i in allowed))
    kinds = {i: item_kind(texts[i].get("question", ""), texts[i]["answer"]) for i in items_all}
    drops, rows = Counter(), []
    targets = {}
    if build == "P":
        tp = load_json(os.path.join(gd, "para_targets.json")) or {}
        targets = {int(i): v for i, v in tp.items()}
    n_short = n_no_pos = n_copy = 0
    for i in items_all:
        if kinds[i] != "ok":
            continue
        gold = texts[i]["answer"]
        if build == "S":
            negs, st = build_candidates(gold, base[i], k)
            drops.update(st)
            pos = None
        elif build == "E":
            negs, st = build_candidates(gold, base[i], k, require_edit=True)
            drops.update(st)
            pos = None
        else:                                                   # P
            negs = p_negatives(gold, base[i], k)
            if negs is None:
                n_short += 1
                continue
            tg = targets.get(i)
            if tg is None:
                t, tc, _ = para_target(key, i, negs, k)
            else:
                t, tc = int(tg["t"]), float(tg["target_chars"])
            paras = [x for r in sorted(raws["para"].get(i, []), key=lambda r: r.get("attempt", 0))
                     for x in r.get("negatives", [])]
            pos = choose_positive(paras, gold, negs, t, tc)
            if pos is None and n_words(gold) <= SHORT_GOLD_WORDS:
                # oro de 1–3 palabras ("Nature.", "2813") que el generador sólo copió: el positivo es el texto
                # del oro en formato canónico, embebido igual que los negativos (sin largo asignado)
                c = canonical(gold)
                if c is not None and _norm(c) not in {_norm(x) for x in negs}:
                    pos = c
                    n_copy += 1
            if pos is None:
                n_no_pos += 1
                continue
        if len(negs) < k:
            n_short += 1
            continue
        rows.append((i, pos, list(negs[:k])))
    stats = {"items_generated": len(items_all), "items_by_kind": dict(Counter(kinds.values())),
             "items_eligible": sum(1 for i in items_all if kinds[i] == "ok"), "items_short": n_short,
             "items_without_positive": n_no_pos, "positive_is_gold_text": n_copy, "negatives_dropped": dict(drops)}
    return rows, stats


_ORDER_CACHE: dict = {}


def panel_order_cached(study, split: str = "test") -> np.ndarray:
    p = os.path.join(tgen_root(study, split), "panel_order.npy")
    key = (p, os.path.getmtime(p) if os.path.exists(p) else None)
    if key not in _ORDER_CACHE:
        _ORDER_CACHE[key] = np.load(p) if os.path.exists(p) else np.zeros(0, np.int64)
    return _ORDER_CACHE[key]


_SHA_CACHE: dict = {}


def _file_sha(p: str, chunk: int = 1 << 20) -> str:
    """sha1 del contenido (en caché por ruta, tamaño y mtime dentro de la sesión)."""
    st = os.stat(p)
    key = (p, int(st.st_size), float(st.st_mtime))
    if key not in _SHA_CACHE:
        h = hashlib.sha1()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        _SHA_CACHE[key] = h.hexdigest()[:16]
    return _SHA_CACHE[key]


def _raw_sigs(study, key: str, split: str, build: str) -> dict:
    """Firma de las generaciones crudas que usa la construcción: [tamaño, sha1 del contenido]. Hasta v0.3.3 era
    [tamaño, mtime]; en Drive el mtime cambia entre sesiones, y eso hacía re-embeber sets idénticos."""
    out = {}
    for t in BUILDS[build]:
        p = os.path.join(gen_dir(study, key, split), TASKS[t]["raw"])
        out[t] = [int(os.path.getsize(p)), _file_sha(p)] if os.path.exists(p) else None
    return out


def raw_sigs_match(old, new) -> bool:
    """¿Las generaciones son las mismas que cuando se embebió el set? Con una firma vieja ([tamaño, mtime]) sólo
    compara el tamaño: las generaciones sólo se agregan al final del archivo, así que cualquier cambio lo cambia."""
    if not isinstance(old, dict) or not isinstance(new, dict) or set(old) != set(new):
        return False
    for t, n in new.items():
        o = old.get(t)
        if o is None or n is None:
            if o != n:
                return False
            continue
        if int(o[0]) != int(n[0]):
            return False
        if isinstance(o[1], str) and o[1] != n[1]:
            return False
    return True


def needs_embedding(study, key: str, split: str = "test", build: str | None = None, sub: str | None = None,
                    n_items: int | None = None) -> bool:
    """True si la generación que pide la construcción terminó (en los primeros n_items, si se indica) y el set
    no está embebido, es de otra construcción o reglas, o sus generaciones cambiaron después."""
    build = build or active_build(study)
    gi = build_gen_info(study, key, split, build)
    if gi["status"] != "generado":
        return False
    if n_items is not None and gi["n_target"] < gen_n(key, n_items):
        return False
    sigs = _raw_sigs(study, key, split, build)
    if any(v is None for v in sigs.values()):
        return False
    d = set_dir(study, key, split, sub)
    emb = load_json(os.path.join(d, "embed_info.json"))
    if emb is None or not os.path.exists(os.path.join(d, "negs.npy")):
        return True
    if emb.get("build") != build or emb.get("rules") != PANEL_RULES:
        return True
    return not raw_sigs_match(emb.get("raw_sigs"), sigs)


def _embed_texts(encoder, texts: list) -> tuple:
    ids = encoder.tokenize(texts, keep="head")
    E = np.asarray(encoder.embed_ids(ids), np.float32) if texts else np.zeros((0, 1), np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
    return E, np.asarray([len(x) for x in ids], np.int32)


def embed_generator(study, test, key: str, encoder, split: str = "test", build: str | None = None,
                    sub: str | None = None, n_items: int | None = None, k: int = K_KEEP, force: bool = False,
                    log=print) -> dict:
    """Construye el set del generador (construct_sets), lo embebe con el pipeline de G1 y guarda items.npy,
    negs.npy, neg_tokens.npy, neg_cos.npy, kept.jsonl y, en P, pos.npy y pos_tokens.npy (el positivo es el oro
    reescrito). n_items limita a los primeros ítems del orden del panel (piloto). Si había un set de otra
    construcción, su resumen queda en history/."""
    build = build or active_build(study)
    d = makedirs(set_dir(study, key, split, sub))
    out_info = os.path.join(d, "embed_info.json")
    gi = build_gen_info(study, key, split, build)
    if gi["status"] != "generado":
        raise RuntimeError(f"{key}: la generación que necesita la construcción {build} no terminó ({gi['tasks']}); "
                           "vuelve a correr la generación")
    if not force and not needs_embedding(study, key, split, build, sub, n_items):
        return load_json(out_info)
    old = load_json(out_info)
    if old and (old.get("build"), old.get("rules")) != (build, PANEL_RULES):
        hist = makedirs(os.path.join(d, "history"))
        tag = f"{old.get('build') or 'v1'}_{old.get('rules') or old.get('build') or 'v1'}"
        for fn in ("embed_info.json", "audit.json"):
            if os.path.exists(os.path.join(d, fn)):
                shutil.copyfile(os.path.join(d, fn), os.path.join(hist, f"{tag}_{fn}"))
    for fn in ("embed_info.json", "audit.json", "pos.npy", "pos_tokens.npy"):   # nada viejo a medio camino
        if os.path.exists(os.path.join(d, fn)):
            os.remove(os.path.join(d, fn))
    rows, stats = construct_sets(study, key, build, split, n_limit=n_items, k=k)
    if not rows:
        raise RuntimeError(f"{key}: ningún ítem quedó con {k} negativos válidos (construcción {build})")
    log(f"[tgen:{key}] construcción {build}: {len(rows)}/{stats['items_eligible']} ítems elegibles con set completo "
        f"({stats['items_by_kind']}); embebiendo {len(rows) * k} negativos"
        + (f" y {len(rows)} positivos reescritos" if build == "P" else ""))
    E, L = _embed_texts(encoder, [x for _, _, ng in rows for x in ng])
    NEG = E.reshape(len(rows), k, -1)
    NT = L.reshape(len(rows), k)
    items = np.asarray([i for i, _, _ in rows], np.int64)
    POS, PT = None, None
    if build == "P":
        POS, PT = _embed_texts(encoder, [p for _, p, _ in rows])
        ref = POS
    else:
        A = np.asarray(test.a_all(), np.float32)[items]
        ref = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    COS = np.einsum("nkh,nh->nk", NEG, ref).astype(np.float32)
    atomic_np_save(items, os.path.join(d, "items.npy"))
    atomic_np_save(NEG.astype(np.float16), os.path.join(d, "negs.npy"))
    atomic_np_save(NT, os.path.join(d, "neg_tokens.npy"))
    atomic_np_save(COS, os.path.join(d, "neg_cos.npy"))
    if POS is not None:
        atomic_np_save(POS.astype(np.float16), os.path.join(d, "pos.npy"))
        atomic_np_save(PT, os.path.join(d, "pos_tokens.npy"))
    tmp = os.path.join(d, "kept.jsonl.tmp")
    with open(tmp, "w") as f:
        for j, (i, pos, ng) in enumerate(rows):
            f.write(json.dumps({"i": int(i), "positive": pos, "negatives": ng,
                                "cos_to_positive": [round(float(x), 4) for x in COS[j]]}, ensure_ascii=False) + "\n")
    os.replace(tmp, os.path.join(d, "kept.jsonl"))
    n_hi = int((COS > FN_THRESHOLD).sum())
    info = {"key": key, "build": build, "build_txt": BUILD_TXT[build], "rules": PANEL_RULES,
            "prompt_versions": {t: TASKS[t]["version"] for t in BUILDS[build]}, **stats,
            "items_kept": int(len(rows)), "items_requested": int(stats["items_eligible"]),   # cobertura: sobre elegibles
            "n_limit": n_items, "k": k, "cos_gt_095": n_hi,
            "negs_sha": content_sha(NEG.astype(np.float16), items,
                                    None if POS is None else POS.astype(np.float16)),
            "raw_sigs": _raw_sigs(study, key, split, build), "created": now_iso()}
    save_json(info, out_info)                              # al final: sin él, el set no cuenta como hecho
    log(f"[tgen:{key}] set {build} listo: {len(rows)} ítems; descartes {stats['negatives_dropped']}; "
        f"{n_hi} negativos con coseno > {FN_THRESHOLD} al positivo (sólo diagnóstico)")
    return info


def content_sha(negs, items, pos=None) -> str:
    """Huella del contenido de un set (ítems + negativos [+ positivos propios]): invalida auditorías y
    evaluaciones viejas. Sin positivos propios da lo mismo que en v0.3.1."""
    h = hashlib.sha1(np.asarray(items, np.int64).tobytes())
    a = np.asarray(negs)
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a).tobytes())
    if pos is not None:
        b = np.asarray(pos)
        h.update(b"pos" + str(b.shape).encode())
        h.update(np.ascontiguousarray(b).tobytes())
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------------------
# 4. Auditoría (no usa ningún modelo entrenado)
class GenSet(tuple):
    """(items, negs, pos): pos es None cuando el positivo es el oro original (su embedding es el del dataset)."""
    __slots__ = ()

    def __new__(cls, items, negs, pos=None):
        return super().__new__(cls, (items, negs, pos))

    items = property(lambda self: self[0])
    negs = property(lambda self: self[1])
    pos = property(lambda self: self[2])


def load_gen(study, key: str, split: str = "test", sub: str | None = None, build: str | None = None):
    """GenSet de un generador con set embebido completo y consistente; None si no lo hay. Para el panel
    (sub=None) sólo cuenta un set de la construcción activa."""
    d = set_dir(study, key, split, sub)
    p_i, p_n = os.path.join(d, "items.npy"), os.path.join(d, "negs.npy")
    emb = load_json(os.path.join(d, "embed_info.json"))
    if emb is None or not (os.path.exists(p_i) and os.path.exists(p_n)):
        return None
    if emb.get("rules") != PANEL_RULES:
        return None                                        # set de reglas viejas (p. ej. v1): hay que rehacerlo
    if sub is None and emb.get("build", "S") != (build or active_build(study)):
        return None                                        # set de otra construcción: hay que rehacerlo
    items, negs = np.load(p_i), np.load(p_n, mmap_mode="r")
    if len(items) != negs.shape[0]:
        return None                                        # escritura interrumpida: se trata como no embebido
    p_p = os.path.join(d, "pos.npy")
    pos = None
    if emb.get("build") == "P" or os.path.exists(p_p):
        if not os.path.exists(p_p):
            return None
        pos = np.load(p_p)
        if len(pos) != len(items):
            return None
    return GenSet(items, negs, pos)


def gen_sig(study, key: str, split: str = "test", sub: str | None = None) -> str | None:
    return (load_json(os.path.join(set_dir(study, key, split, sub), "embed_info.json")) or {}).get("negs_sha")


def _cands(A, items, negs, pos=None) -> np.ndarray:
    first = np.asarray(A[items] if pos is None else pos, np.float32)
    return np.concatenate([first[:, None, :], np.asarray(negs, np.float32)], 1)


def train_blind_on(C: np.ndarray, hidden: int, steps: int = 300, batch: int = 128, lr: float = 1e-3,
                   seed: int = 0, device=None):
    """Modelo ciego (sólo ve los candidatos; el oro va en la columna 0) entrenado con C (n x (1+K) x H)."""
    import torch
    from .losses import BlindScorer
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    model = BlindScorer(hidden).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    rs = np.random.default_rng(seed)
    X = torch.from_numpy(np.asarray(C, np.float32))
    tgt = torch.zeros(min(batch, len(C)), dtype=torch.long, device=dev)
    for _ in range(steps):
        b = rs.choice(len(C), size=min(batch, len(C)), replace=False)
        s = model(X[b].to(dev))
        loss = torch.nn.functional.cross_entropy(s, tgt[:len(b)])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model.eval()


def blind_correct(model, C: np.ndarray, bs: int = 1024) -> np.ndarray:
    import torch
    from .evaluate import rank_metrics
    dev = next(model.parameters()).device
    with torch.no_grad():
        S = np.concatenate([model(torch.from_numpy(np.asarray(C[s:s + bs], np.float32)).to(dev)).cpu().numpy()
                            for s in range(0, len(C), bs)])
    return rank_metrics(S)[1]


def band_of(x: float | None) -> str | None:
    if x is None or not np.isfinite(x):
        return None
    lo, hi = BANDS
    return "despreciable" if x < lo else ("moderado" if x < hi else "alto")


def audit_generator(study, bundle, key: str, split: str = "test", blind=None, folds: int = 5, steps: int = 300,
                    seed: int = 0, force: bool = False, sub: str | None = None, log=print) -> dict | None:
    """Sondas ciegas sobre el set de un generador (sólo ven los 11 candidatos, nunca el estado). "Atajo ciego" =
    el máximo de las sondas. sub = un set del piloto."""
    from .evaluate import rank_metrics, raw_cosine_scores
    from .texts import length_heuristic
    d = set_dir(study, key, split, sub)
    p = os.path.join(d, "audit.json")
    g = load_gen(study, key, split, sub=sub)
    if g is None:
        return None
    items, negs, pos = g
    sha = gen_sig(study, key, split, sub) or hashlib.sha1(np.asarray(items, np.int64).tobytes()).hexdigest()[:12]
    old = load_json(p)
    if old is not None and not force and old.get("items_sha") == sha and ("blind_gemini" in old or blind is None):
        return old
    test = bundle.test
    A = test.a_all()
    C = _cands(A, items, negs, pos)
    chance = 1.0 / C.shape[1]
    res = {"key": key, "n_items": int(len(items)), "items_sha": sha, "chance": chance,
           "build": (load_json(os.path.join(d, "embed_info.json")) or {}).get("build")}
    lp = os.path.join(study.paths.d("data", "eval", split), "lengths.npz")
    pt = os.path.join(d, "neg_tokens.npy")
    gold = None
    if pos is not None and os.path.exists(os.path.join(d, "pos_tokens.npy")):
        gold = np.load(os.path.join(d, "pos_tokens.npy"))           # el positivo es el oro reescrito
    elif pos is None and os.path.exists(lp):
        gold = np.load(lp)["gold"][items]
    if gold is not None and os.path.exists(pt):
        nt = np.load(pt)
        res["len_gold_mean"] = float(np.mean(gold[gold >= 0])) if (gold >= 0).any() else None
        res["len_neg_mean"] = float(nt.mean())
        res["longest"] = float(np.nanmean(length_heuristic(gold, nt, "longest")))
        res["shortest"] = float(np.nanmean(length_heuristic(gold, nt, "shortest")))
    if blind is not None:
        res["blind_gemini"] = float(blind_correct(blind, C).mean())
    # ciego propio con validación cruzada (pliegues por ítem; ≤ 1 ítem por documento)
    order = np.random.default_rng(seed).permutation(len(items))
    correct = np.zeros(len(items), np.float32)
    for f in range(folds):
        te = order[f::folds]
        tr = np.setdiff1d(order, te)
        m = train_blind_on(C[tr], C.shape[2], steps=steps, seed=seed + f)
        correct[te] = blind_correct(m, C[te])
    res["blind_own_cv"] = float(correct.mean())
    atomic_np_save(correct, os.path.join(d, "blind_own_cv_correct.npy"))
    res["raw_cosine"] = float(rank_metrics(raw_cosine_scores(test, "external", items=items, negs=np.asarray(negs),
                                                             pos=pos))[0]["top1"])
    probes = [res.get(x) for x in ("longest", "shortest", "blind_gemini", "blind_own_cv")]
    probes = [x for x in probes if x is not None and np.isfinite(x)]
    res["atajo_ciego"] = float(max(probes)) if probes else None
    res["banda"] = band_of(res["atajo_ciego"])
    res["created"] = now_iso()
    save_json(res, p)
    log(f"[tgen:{key}{'' if sub is None else ':' + sub}] auditoría: ciego propio {res['blind_own_cv']:.3f}"
        + (f", ciego de Gemini {res['blind_gemini']:.3f}" if "blind_gemini" in res else "")
        + (f", más larga {res['longest']:.3f}, más corta {res['shortest']:.3f}" if "longest" in res else "")
        + f", coseno crudo {res['raw_cosine']:.3f} -> atajo {res['atajo_ciego']:.3f} ({res['banda']})")
    return res


def common_item_folds(item_lists, folds: int = 5, seed: int = 0):
    """Pliegue de cada ítem, el MISMO en todos los sets: (ítems únicos ordenados, pliegue de cada uno)."""
    u = np.unique(np.concatenate([np.asarray(x, np.int64) for x in item_lists])) if item_lists else np.zeros(0, np.int64)
    fold_u = np.empty(len(u), np.int64)
    fold_u[np.random.default_rng(seed).permutation(len(u))] = np.arange(len(u)) % folds
    return u, fold_u


def fold_of(items, u: np.ndarray, fold_u: np.ndarray) -> np.ndarray:
    return fold_u[np.searchsorted(u, np.asarray(items, np.int64))]


BLIND_TRANSFER_METHOD = "pliegues por ítem comunes a todos los generadores (v0.3.3)"


def transfer_rows(data: dict, its: dict, folds: int = 5, steps: int = 300, seed: int = 0) -> dict:
    """Núcleo de blind_transfer_matrix: data[k] = candidatos (n_k × 11 × H, oro en la columna 0), its[k] = ítems.
    Devuelve {fila: {columna: top-1}} con pliegues por ítem comunes (ningún ítem evaluado se vio al entrenar)."""
    rows = {}
    if not data:
        return rows
    u, fold_u = common_item_folds(list(its.values()), folds, seed)
    fold = {k: fold_of(it, u, fold_u) for k, it in its.items()}
    for ka, Ca in data.items():
        acc = {kb: np.full(len(its[kb]), np.nan, np.float32) for kb in data}
        for f in range(folds):
            tr = fold[ka] != f
            if tr.sum() < 2:
                continue
            m = train_blind_on(Ca[tr], Ca.shape[2], steps=steps, seed=seed + f)
            for kb, Cb in data.items():
                te = fold[kb] == f
                if te.any():
                    acc[kb][te] = blind_correct(m, Cb[te])
        rows[ka] = {kb: (float(np.nanmean(v)) if np.isfinite(v).any() else None) for kb, v in acc.items()}
    return rows


def blind_transfer_matrix(study, bundle, keys: list, split: str = "test", blind=None, steps: int = 300,
                          seed: int = 0, folds: int = 5, log=print):
    """Fila = con qué set se entrena el ciego; columna = en qué set se evalúa (top-1; azar 1/11).

    Pliegues por ítem COMUNES a todos los generadores: el ciego de la fila se entrena con sus ítems de los otros
    pliegues y se evalúa, en cada columna, sólo en los ítems del pliegue que quedó afuera. Así nunca se lo evalúa
    en un ítem cuyo oro vio al entrenar (en S y E el oro es el mismo texto en todos los generadores). La diagonal
    es el ciego propio con estos mismos pliegues (puede diferir unas centésimas del de audit.json, que usa los
    pliegues de cada generador). Fila "gemini (F0.6)": el ciego de F0.6, entrenado en el split de entrenamiento.

    Hasta v0.3.2 la fila se entrenaba con TODOS los ítems de su set y se evaluaba en todos los de la columna:
    con el mismo oro en todos los sets, eso mezclaba memorización del oro con atajo compartido (fuga por ítem).
    Esa versión, si existe en Drive, se mueve a history/ y no se usa. Sólo se muestra: no entra en ninguna regla."""
    import pandas as pd
    A = bundle.test.a_all()
    data, its = {}, {}
    for k in keys:
        g = load_gen(study, k, split)
        if g is not None:
            data[k] = _cands(A, g.items, g.negs, g.pos)
            its[k] = np.asarray(g.items, np.int64)
    rows = {}
    if blind is not None:
        rows["gemini (F0.6)"] = {k: float(blind_correct(blind, C).mean()) for k, C in data.items()}
    rows.update(transfer_rows(data, its, folds=folds, steps=steps, seed=seed))
    df = pd.DataFrame(rows).T[list(data)] if data else pd.DataFrame()
    if len(df):
        root = tgen_root(study, split)
        path, info_p = os.path.join(root, "blind_transfer.csv"), os.path.join(root, "blind_transfer_info.json")
        if os.path.exists(path) and (load_json(info_p) or {}).get("method") != BLIND_TRANSFER_METHOD:
            makedirs(os.path.join(root, "history"))
            shutil.move(path, os.path.join(root, "history", "blind_transfer_v032_fuga_por_item.csv"))
        df.to_csv(path)
        save_json({"method": BLIND_TRANSFER_METHOD, "folds": folds, "steps": steps, "seed": seed,
                   "keys": list(data), "created": now_iso()}, info_p)
    return df


# --------------------------------------------------------------------------------------
# 5. Panel y macro (regla fija)
def gen_status(study, key: str, split: str = "test") -> str:
    """ok | sin_auditar | sin_embeber | <estado de la generación que pide la construcción activa>."""
    gd = gen_dir(study, key, split)
    if load_gen(study, key, split) is not None:
        aud = load_json(os.path.join(gd, "audit.json")) or {}
        return "ok" if aud.get("items_sha") == gen_sig(study, key, split) else "sin_auditar"
    st = build_gen_info(study, key, split)["status"]
    return "sin_embeber" if st == "generado" else (st or "no_generado")


MIN_COVERAGE = 0.8   # fracción mínima de los ítems pedidos con 10 negativos válidos para entrar al macro
TIERS = ("grande", "chico")
TIER_METRIC = {"grande": "tgenbig", "chico": "tgensmall"}   # macro por tamaño (secundario, ENMIENDA_03 §10)


def _intersect(study, keys, split):
    items = None
    for k in keys:
        it = load_gen(study, k, split)[0]
        items = it if items is None else np.intersect1d(items, it)
    return np.asarray(items if items is not None else [], np.int64)


def freeze_panel(study, split: str = "test", keys: list | None = None, min_macro: int = 5,
                 force_macro: list | None = None, exclude_qc: dict | None = None, log=print) -> dict:
    """PANEL.json: generadores, auditoría y macro de T-gen. Regla (ENMIENDA_03 §3.6): entran al macro los de
    rol primario cuya generación TERMINÓ, con embebido y auditoría, cobertura ≥ 80 % de los ítems pedidos,
    atajo ciego NO alto y sin exclusión por el QC manual (> 5 % de falsos negativos); ítems = intersección.
    No mira ningún resultado de modelos. "todos" = lo mismo sin el criterio de atajo (el reporte de respaldo
    si quedan menos de min_macro). Si el contenido cambia, la versión anterior queda en history/.
    force_macro (test confirmatorio): el macro son exactamente los generadores congelados en el freeze."""
    root = makedirs(tgen_root(study, split))
    keys = keys or sorted({d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != "history"
                           and any(os.path.exists(os.path.join(root, d, f))
                                   for f in [t["info"] for t in TASKS.values()] + ["embed_info.json"])})
    exclude_qc = dict(exclude_qc or {})
    gens, macro, todos, faltan = {}, [], [], []
    build = active_build(study)
    for k in keys:
        gen = PANEL_BY_KEY.get(k)
        aud = load_json(os.path.join(gen_dir(study, k, split), "audit.json")) or {}
        info = build_gen_info(study, k, split, build)
        emb = load_json(os.path.join(gen_dir(study, k, split), "embed_info.json")) or {}
        role = gen.role if gen else info.get("role", "sintetico")
        st = gen_status(study, k, split)
        req = emb.get("items_requested") or info.get("items_requested")
        cov = (float(emb.get("items_kept", 0)) / float(req)) if req else None
        done = info.get("status") == "generado" and st == "ok"
        base_ok = bool(role in MACRO_ROLES and done and (cov is None or cov >= MIN_COVERAGE) and k not in exclude_qc)
        if force_macro is not None:
            inm = bool(k in force_macro and done)
            if k in force_macro and not done:
                faltan.append(k)
        else:
            inm = bool(base_ok and aud.get("banda") in ("despreciable", "moderado"))
        gens[k] = {"repo": gen.repo if gen else info.get("repo"), "family": gen.family if gen else info.get("family"),
                   "role": role, "tier": gen.tier if gen else info.get("tier"),
                   "status": st, "generation": info.get("status"), "items": aud.get("n_items") or emb.get("items_kept"),
                   "coverage": None if cov is None else round(cov, 4), "atajo_ciego": aud.get("atajo_ciego"),
                   "banda": aud.get("banda"), "excluido_qc": exclude_qc.get(k), "negs_sha": emb.get("negs_sha"),
                   "in_macro": inm, "revision": info.get("revision")}
        if inm:
            macro.append(k)
        if base_ok:
            todos.append(k)
    if force_macro is not None:
        faltan += [k for k in force_macro if k not in keys]
        if faltan:
            log(f"[tgen] ATENCIÓN: generadores congelados sin negativos completos en {split}: {faltan}. El macro del "
                "confirmatorio NO es el congelado hasta completarlos (8.2b).")
    items = _intersect(study, macro, split)
    items_todos = _intersect(study, todos, split)
    # análisis secundario (ENMIENDA_03 §10): el macro separado por tamaño del generador, cada uno en la
    # intersección de sus propios ítems. No cambia el macro ni ninguna regla.
    por_tamano = {}
    for t in TIERS:
        ks = [k for k in macro if gens[k]["tier"] == t]
        if ks:
            por_tamano[t] = {"keys": ks, "items": _intersect(study, ks, split).tolist()}
    sig = hashlib.sha256(json.dumps({"macro": macro, "items": items.tolist(),
                                     "sha": [gens[k]["negs_sha"] for k in macro],
                                     "tiers": {k: gens[k]["tier"] for k in macro},
                                     "build": build}).encode()).hexdigest()[:12]
    rule = ("macro = generadores de rol primario con generación terminada, embebido y auditoría, cobertura ≥ 80 %, "
            "atajo ciego (máximo de longitud, ciego de Gemini y ciego propio) < 30 % y sin exclusión por QC; "
            "ítems = intersección. No depende de resultados de modelos.")
    if force_macro is not None:
        rule = f"macro fijado por el PROTOCOL FREEZE: {list(force_macro)}; ítems = intersección."
    panel = {"split": split, "version": sig, "created": now_iso(), "rule": rule,
             "build": build, "build_txt": BUILD_TXT[build], "rules": PANEL_RULES,
             "generators": gens, "macro": macro, "macro_items": items.tolist(), "n_macro_items": int(len(items)),
             "todos": todos, "todos_items": items_todos.tolist(), "macro_por_tamano": por_tamano,
             "robusto": len(macro) >= min_macro,
             "min_macro": min_macro, "faltan_del_freeze": faltan if force_macro is not None else None,
             "exclusiones_qc": exclude_qc}
    p = panel_path(study, split)
    old = load_json(p)
    if old is not None and old.get("version") != sig:
        save_json(old, os.path.join(root, "history", f"PANEL_{old.get('version')}.json"))
    if old is None or old.get("version") != sig or old.get("generators") != gens or old.get("todos") != todos:
        save_json(panel, p)
    else:
        panel = old
    if not panel["robusto"]:
        log(f"[tgen] ATENCIÓN: sólo {len(macro)} generadores en el macro (mínimo {min_macro}). Ver ENMIENDA_03 §3.6: "
            "se reporta además el macro de todos los primarios ('tgenall', con atajo) y los controles.")
    return panel


def load_panel(study, split: str = "test") -> dict | None:
    return load_json(panel_path(study, split))


def _local_copy(src: str, dst: str) -> str:
    """Copia a disco local (lecturas rápidas; Drive es lento para leer lo mismo en cada evaluación)."""
    try:
        if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src) \
                and os.path.getmtime(dst) >= os.path.getmtime(src):
            return dst
        makedirs(os.path.dirname(dst))
        tmp = dst + ".tmp"
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
        return dst
    except OSError:
        return src


def load_panel_sets(study, split: str = "test", localize: bool = True) -> dict:
    """{"tgen_<clave>": (items, negs, huella, positivos)} de todos los generadores con set completo de la
    construcción activa (macro o no). positivos = None si el positivo es el oro original. La huella (contenido
    del set) invalida evaluaciones viejas si el set cambia. Los negativos se leen de una copia en el disco local
    (se rehace sola si cambian en Drive)."""
    root = tgen_root(study, split)
    out = {}
    if not os.path.isdir(root):
        return out
    for k in sorted(os.listdir(root)):
        if k == "history" or not os.path.isdir(os.path.join(root, k)):
            continue
        g = load_gen(study, k, split)
        if g is None:
            continue
        p_n = os.path.join(root, k, "negs.npy")
        if localize and not study.data.synthetic:
            p_n = _local_copy(p_n, study.paths.c("tgen_local", split, k, "negs.npy"))
        out[f"tgen_{k}"] = (g.items, np.load(p_n, mmap_mode="r"), gen_sig(study, k, split), g.pos)
    return out


MACRO_FILES = ("correct_{n}.npy", "correct_{n}_shuf.npy", "items_{n}.npy", "{n}.json")


def _clear_macro(ev: str, name: str) -> None:
    for f in MACRO_FILES:
        p = os.path.join(ev, f.format(n=name))
        if os.path.exists(p):
            os.remove(p)


def _macro(ev: str, keys: list, items: np.ndarray, version: str, name: str, extra_sigs: dict) -> dict | None:
    """Promedio por ítem del acierto sobre `keys`, alineado en `items`. Si falta algo, borra el macro viejo."""
    sig_p = os.path.join(ev, f"{name}.json")
    old = load_json(sig_p)
    if old and old.get("version") == version and os.path.exists(os.path.join(ev, f"correct_{name}.npy")):
        return old
    cols, cols_s = [], []
    for k in keys:
        tn = f"tgen_{k}"
        pc, pi = os.path.join(ev, f"correct_{tn}.npy"), os.path.join(ev, f"items_{tn}.npy")
        if not (os.path.exists(pc) and os.path.exists(pi)) or (extra_sigs.get(k) and
                                                               (load_json(os.path.join(ev, "extra_sigs.json")) or {}).get(tn) != extra_sigs[k]):
            _clear_macro(ev, name)
            return None
        it = np.load(pi)
        if not np.isin(items, it).all():
            _clear_macro(ev, name)
            return None
        o = np.argsort(it)
        pos = o[np.searchsorted(it[o], items)]
        cols.append(np.load(pc)[pos])
        ps = os.path.join(ev, f"correct_{tn}_shuf.npy")
        cols_s.append(np.load(ps)[pos] if os.path.exists(ps) else None)
    M = np.mean(cols, 0).astype(np.float32)
    atomic_np_save(M, os.path.join(ev, f"correct_{name}.npy"))
    atomic_np_save(items, os.path.join(ev, f"items_{name}.npy"))
    out = {"version": version, "keys": list(keys), "n": int(len(items)), "top1": float(M.mean())}
    if all(c is not None for c in cols_s):
        Ms = np.mean(cols_s, 0).astype(np.float32)
        atomic_np_save(Ms, os.path.join(ev, f"correct_{name}_shuf.npy"))
        out["shuf_top1"] = float(Ms.mean())
    save_json(out, sig_p)
    return out


def macro_for_run(run_dir: str, panel: dict | None) -> dict | None:
    """Acierto por ítem del macro de T-gen para una corrida (promedio sobre los generadores del macro, en la
    intersección de ítems): eval/correct_tgenmacro[_shuf].npy. También "tgenall" (todos los primarios, sin el
    criterio de atajo; respaldo de §3.6), que se calcula AUNQUE el macro quede vacío. Si la corrida no tiene todo
    lo necesario, borra los archivos viejos del macro (nunca se usa un macro de otra versión del panel) y
    devuelve None."""
    ev = os.path.join(run_dir, "eval")
    if not panel:
        for nm in ("tgenmacro", "tgenall") + tuple(TIER_METRIC.values()):
            _clear_macro(ev, nm)
        return None
    sigs = {k: (panel["generators"].get(k) or {}).get("negs_sha") for k in panel.get("generators", {})}
    if panel.get("todos") and panel.get("todos_items"):
        _macro(ev, panel["todos"], np.asarray(panel["todos_items"], np.int64), panel["version"] + "-todos", "tgenall", sigs)
    else:
        _clear_macro(ev, "tgenall")
    if not panel.get("macro") or not panel.get("macro_items"):
        for nm in ("tgenmacro",) + tuple(TIER_METRIC.values()):
            _clear_macro(ev, nm)
        return None
    out = _macro(ev, panel["macro"], np.asarray(panel["macro_items"], np.int64), panel["version"], "tgenmacro", sigs)
    for t, nm in TIER_METRIC.items():
        sub = (panel.get("macro_por_tamano") or {}).get(t) or {}
        if sub.get("keys") and sub.get("items"):
            _macro(ev, sub["keys"], np.asarray(sub["items"], np.int64), f"{panel['version']}-{t}", nm, sigs)
        else:
            _clear_macro(ev, nm)
    return out


QC_COL = "es_correcta(0/1)"
QC_MIN_MARKED = 50       # filas marcadas por generador para aplicar la regla del QC
QC_MAX_ERR = 0.05        # criterio de DISENO_01: < 5 % de falsos negativos


def export_qc(study, split: str = "test", keys: list | None = None, n_items: int = 20, seed: int = 0) -> str | None:
    """CSV para la revisión manual (criterio de DISENO_01: < 5 %): por generador, n_items ítems con sus 10
    negativos y, en P, también el oro reescrito (fila "positivo": ¿sigue siendo correcto?). Si ya hay un CSV
    del panel con marcas, NO se sobreescribe: se guarda el nuevo como QC_falsos_negativos_panel_nuevo.csv."""
    import pandas as pd
    root = tgen_root(study, split)
    texts = {r["i"]: r for r in read_jsonl(os.path.join(root, "items_text.jsonl")) if not r.get("missing")}
    rows = []
    for k in keys or sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)) and d != "history"):
        if load_gen(study, k, split) is None:
            continue
        kept = read_jsonl(os.path.join(root, k, "kept.jsonl"))
        if not kept:
            continue
        pick = np.random.default_rng(seed).choice(len(kept), size=min(n_items, len(kept)), replace=False)
        sub = []
        for j in sorted(pick):
            r = kept[int(j)]
            t = texts.get(r["i"], {})
            base = {"generador": k, "item": r["i"], "pregunta": t.get("question"), "oro": t.get("answer")}
            if r.get("positive"):
                sub.append({**base, "tipo": "positivo", "candidato": r["positive"], "coseno_al_positivo": 1.0,
                            QC_COL: "", "notas": ""})
            for neg, c in zip(r["negatives"], r.get("cos_to_positive") or r.get("cos_to_gold") or []):
                sub.append({**base, "tipo": "negativo", "candidato": neg, "coseno_al_positivo": round(float(c), 3),
                            QC_COL: "", "notas": ""})
        pd.DataFrame(sub).to_csv(os.path.join(root, k, "QC_falsos_negativos.csv"), index=False)
        rows += sub
    if not rows:
        return None
    p = os.path.join(root, "QC_falsos_negativos_panel.csv")
    if qc_marked_rows(p):
        p = os.path.join(root, "QC_falsos_negativos_panel_nuevo.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def qc_marked_rows(path: str) -> int:
    try:
        import pandas as pd
        df = pd.read_csv(path)
    except Exception:  # noqa: BLE001
        return 0
    if QC_COL not in df.columns:
        return 0
    return int(pd.to_numeric(df[QC_COL], errors="coerce").notna().sum())


def qc_exclusions(study, split: str = "test", min_marked: int = QC_MIN_MARKED, max_err: float = QC_MAX_ERR) -> dict:
    """Exclusiones automáticas por el QC manual (ENMIENDA_03 §3.6): se leen las marcas del CSV del panel; un
    generador con ≥ min_marked filas marcadas sale del macro si > max_err de sus negativos marcados son en
    realidad correctos o (en P) > max_err de sus positivos marcados no lo son. Sin marcas, no excluye a nadie.
    Las filas de una construcción anterior (sin columna "tipo" o con otro set) se ignoran."""
    import pandas as pd
    root = tgen_root(study, split)
    frames = []
    for fn in ("QC_falsos_negativos_panel.csv", "QC_falsos_negativos_panel_nuevo.csv"):
        p = os.path.join(root, fn)
        if os.path.exists(p):
            try:
                d = pd.read_csv(p, dtype={"candidato": str, "generador": str})
            except Exception:  # noqa: BLE001
                continue
            if QC_COL in d.columns and "tipo" in d.columns and "candidato" in d.columns:
                frames.append(d)
    if not frames:
        return {}
    df = pd.concat(frames, ignore_index=True)
    df["marca"] = pd.to_numeric(df[QC_COL], errors="coerce")
    df = df[df["marca"].isin([0, 1])]
    out = {}
    for k, g in df.groupby("generador"):
        if len(g) < min_marked or load_gen(study, k, split) is None:
            continue
        kept = {(r["i"], c) for r in read_jsonl(os.path.join(root, k, "kept.jsonl"))
                for c in (r["negatives"] + ([r["positive"]] if r.get("positive") else []))}
        g = g[[(int(i), str(c)) in kept for i, c in zip(g["item"], g["candidato"])]]   # sólo el set actual
        g = g.drop_duplicates(subset=["item", "tipo", "candidato"], keep="last")
        if len(g) < min_marked:
            continue
        neg, pos = g[g["tipo"] == "negativo"], g[g["tipo"] == "positivo"]
        fn = float(neg["marca"].mean()) if len(neg) else 0.0
        bad_pos = float(1.0 - pos["marca"].mean()) if len(pos) else 0.0
        if fn > max_err or bad_pos > max_err:
            out[str(k)] = (f"QC manual: {fn:.1%} de negativos correctos"
                           + (f", {bad_pos:.1%} de positivos incorrectos" if len(pos) else "")
                           + f" ({len(g)} filas marcadas; criterio > {max_err:.0%})")
    return out


def panel_table(study, split: str = "test"):
    """Tabla del panel para el notebook."""
    import pandas as pd
    panel = load_panel(study, split) or {}
    rows = []
    for k, g in (panel.get("generators") or {}).items():
        aud = load_json(os.path.join(gen_dir(study, k, split), "audit.json")) or {}
        emb = load_json(os.path.join(gen_dir(study, k, split), "embed_info.json")) or {}
        info = build_gen_info(study, k, split)
        rows.append({"generador": k, "familia": g.get("family"), "rol": ROLE_TXT.get(g.get("role"), g.get("role")),
                     "tamaño": g.get("tier"), "estado": g.get("status"), "ítems": g.get("items"),
                     "cobertura": g.get("coverage"), "más_larga": aud.get("longest"), "más_corta": aud.get("shortest"),
                     "ciego_Gemini": aud.get("blind_gemini"), "ciego_propio": aud.get("blind_own_cv"),
                     "coseno_crudo": aud.get("raw_cosine"), "atajo_ciego": aud.get("atajo_ciego"),
                     "banda": aud.get("banda"), "excluido_qc": g.get("excluido_qc"), "en_macro": g.get("in_macro"),
                     "construcción": emb.get("build"),
                     "min_generación": round(float(info.get("seconds_total", 0)) / 60, 1) if info else None})
    return pd.DataFrame(rows)


def panel_summary(study) -> dict:
    """Descripción del panel (para la celda y los documentos)."""
    return {g.key: {k: v for k, v in asdict(g).items() if k != "key"} for g in PANEL}


# --------------------------------------------------------------------------------------
# Panel sintético (perfil smoke y pruebas): mismos archivos, sin LLM
def _synthetic_latent(study, test):
    from .data import _l2  # noqa: F401  (se usa en _synthetic_set)
    pre_root = study.paths.c("synthetic", "pre")
    shard = study.data.shard
    info = load_json(os.path.join(pre_root, shard, "SYNTHETIC.json"))
    Wa = np.load(os.path.join(pre_root, shard, "W_a.npy"))
    man = load_json(os.path.join(test.dir, "manifest.json"))
    lat = []
    for c in man["chunks"]:
        rows = load_json(os.path.join(test.dir, f"c{c:04d}_meta.json"))["row"]
        lat.append(np.load(os.path.join(pre_root, shard, f"chunk_{c:04d}_latent.npy"))[rows])
    lat = np.concatenate(lat)
    sub = os.path.join(test.dir, "subsample.npy")
    if os.path.exists(sub):
        lat = lat[np.load(sub)]
    return info, Wa, lat


def _synthetic_set(study, test, d: str, key: str, items: np.ndarray, art: float, len_bias: float, seed: int,
                   build: str, lat_info, split: str = "test") -> None:
    """Escribe un set sintético (mismos archivos que embed_generator) en d. En P, el positivo es el oro con ruido
    (como una reescritura) y la longitud se asigna como en para_target."""
    from .data import _l2
    info, Wa, lat = lat_info
    makedirs(d)
    rs = np.random.default_rng(seed)
    v = _l2(rs.standard_normal(Wa.shape[1]))
    delta = rs.standard_normal((len(items), K_KEEP, info["latent"])).astype(np.float32)
    n = _l2((lat[items][:, None, :] + info["hardness"] * delta) @ Wa
            + info["noise"] * rs.standard_normal((len(items), K_KEEP, Wa.shape[1])) + art * v)
    lp = os.path.join(study.paths.d("data", "eval", split), "lengths.npz")
    gold_len = np.load(lp)["gold"] if os.path.exists(lp) else None
    base = gold_len[items] if gold_len is not None else rs.integers(10, 60, len(items))
    nt = np.maximum(1, (base[:, None] * (1 - len_bias) + rs.integers(-6, 7, (len(items), K_KEEP)))).astype(np.int32)
    pos = None
    if build == "P":
        A = np.asarray(test.a_all(), np.float32)[items]
        pos = _l2(A + 0.05 * rs.standard_normal(A.shape).astype(np.float32) * np.abs(A).mean()).astype(np.float16)
        L = np.sort(nt, 1)
        t = rs.integers(0, K_KEEP + 1, len(items))
        pt = np.asarray([max(1, int(L[j, K_KEEP - t[j] - 1] + 1)) if t[j] < K_KEEP else max(1, int(L[j, 0]) - 1)
                         for j in range(len(items))], np.int32)
        atomic_np_save(pos, os.path.join(d, "pos.npy"))
        atomic_np_save(pt, os.path.join(d, "pos_tokens.npy"))
    atomic_np_save(items.astype(np.int64), os.path.join(d, "items.npy"))
    atomic_np_save(n.astype(np.float16), os.path.join(d, "negs.npy"))
    atomic_np_save(nt, os.path.join(d, "neg_tokens.npy"))
    with open(os.path.join(d, "kept.jsonl"), "w") as f:
        for i in items:
            f.write(json.dumps({"i": int(i), "positive": f"{key} oro reescrito {i}" if build == "P" else None,
                                "negatives": [f"{key} negativo {t}" for t in range(K_KEEP)],
                                "cos_to_positive": [0.5] * K_KEEP}) + "\n")
    save_json({"key": key, "build": build, "build_txt": BUILD_TXT[build], "rules": PANEL_RULES,
               "items_kept": int(len(items)), "items_requested": int(len(items)), "items_eligible": int(len(items)),
               "negs_sha": content_sha(n.astype(np.float16), items, pos), "synthetic": True, "created": now_iso()},
              os.path.join(d, "embed_info.json"))


SYN_SPECS = [("syn_limpio", "primario", 0.0, 0.0), ("syn_debil", "primario", 0.08, 0.0),
             ("syn_largo", "primario", 0.0, 0.6), ("syn_fuerte", "primario", 0.6, 0.0),
             ("syn_control", "control_qwen", 0.0, 0.0)]
SYN_TIER = {"syn_limpio": "grande", "syn_debil": "chico", "syn_largo": "grande", "syn_fuerte": "chico"}


def _synthetic_texts(study, split: str, items) -> None:
    tpath = os.path.join(tgen_root(study, split), "items_text.jsonl")
    if not os.path.exists(tpath):
        with open(tpath, "w") as f:
            for i in items:
                f.write(json.dumps({"i": int(i), "question": f"pregunta {i}", "answer": f"oro {i}",
                                    "gold_tokens": 20}) + "\n")


def synthetic_panel(study, test, split: str = "test", n_items: int = 300, specs=None, seed: int = 0,
                    build: str | None = None) -> list:
    """Generadores sintéticos con artefactos de distinta fuerza (en el espacio de embedding y en la longitud),
    para probar auditoría, macro, evaluación y estadística, con la construcción activa (en P, con positivos
    propios). No dice nada sobre generadores reales."""
    build = build or active_build(study)
    specs = specs or SYN_SPECS
    lat_info = _synthetic_latent(study, test)
    order = panel_order(study, test, split)
    items = np.sort(order[:n_items])
    made = []
    for j, (key, role, art, len_bias) in enumerate(specs):
        gd = makedirs(gen_dir(study, key, split))
        emb = load_json(os.path.join(gd, "embed_info.json")) or {}
        if os.path.exists(os.path.join(gd, "negs.npy")) and emb.get("build") == build \
                and emb.get("rules") == PANEL_RULES:
            made.append(key)
            continue
        for fn in ("audit.json", "pos.npy", "pos_tokens.npy"):
            if os.path.exists(os.path.join(gd, fn)):
                os.remove(os.path.join(gd, fn))
        for t in BUILDS[build]:
            save_json({"key": key, "role": role, "family": "sintético", "repo": None, "status": "generado",
                       "task": t, "artifact": art, "len_bias": len_bias, "items_requested": int(len(items)),
                       "n_target": int(len(items)), "tier": SYN_TIER.get(key)},
                      os.path.join(gd, TASKS[t]["info"]))
        _synthetic_set(study, test, gd, key, items, art, len_bias, seed + 101 * (j + 1), build, lat_info, split)
        made.append(key)
    _synthetic_texts(study, split, items)
    return made


# --------------------------------------------------------------------------------------
# 5b. Piloto de construcción (ENMIENDA_03 §11): ¿con cuál construcción el set NO delata el correcto?
PILOT_RULE = ("Se arman las construcciones S, P y E con los mismos ítems (los primeros {n} del orden del panel) y "
              "los mismos {k} generadores primarios; se audita cada set con las sondas ciegas de §3.5 (máximo de "
              "las sondas = atajo ciego). Gana la construcción con la MENOR mediana del atajo ciego entre los "
              "generadores; empates (< 1 pp) a favor de la más barata (S, luego P, luego E). Una construcción con "
              "menos de 2 generadores auditados no compite. No mira ningún modelo entrenado.")


def decide_build(results: dict, tie: float = 0.01) -> dict:
    """results = {construcción: {generador: atajo_ciego}} -> {"build", "medianas", "regla"}."""
    med = {}
    for b in BUILD_PREF:
        vals = [float(v) for v in (results.get(b) or {}).values() if v is not None and np.isfinite(v)]
        if len(vals) >= 2:
            med[b] = float(np.median(vals))
    if not med:
        return {"build": None, "medianas": {}, "motivo": "ninguna construcción con ≥ 2 generadores auditados"}
    best = min(med.values())
    chosen = next(b for b in BUILD_PREF if b in med and med[b] <= best + tie)
    return {"build": chosen, "medianas": med}


def pilot_keys(candidates=PILOT_CANDIDATES, k: int = PILOT_K, check: bool = True) -> list:
    """Los primeros k generadores primarios de la lista con acceso y soporte en este entorno."""
    out = []
    for key in candidates:
        g = PANEL_BY_KEY[key]
        if installed_transformers_ok(g.min_transformers):
            continue
        if check and not check_access(g.repo)[0]:
            continue
        out.append(key)
        if len(out) >= k:
            break
    return out


def save_build(study, decision: dict, pilot: dict) -> dict:
    rec = {"build": decision["build"], "build_txt": BUILD_TXT.get(decision["build"]), "decided": now_iso(),
           "rule": PILOT_RULE.format(n=pilot.get("n"), k=len(pilot.get("keys") or [])),
           "medianas": decision.get("medianas"), "pilot": pilot, "rules": PANEL_RULES}
    save_json(rec, build_path(study))
    return rec


def run_pilot(study, bundle, make_encoder_fn, blind=None, keys: list | None = None, n: int = PILOT_N,
              split: str = "test", gen_kwargs: dict | None = None, log=print) -> dict:
    """Piloto automático: genera lo que falta en los generadores del piloto (primeros n ítems), arma y embebe
    S, P y E, las audita y fija la construcción del panel en BUILD.json. Reanudable; si BUILD.json ya existe,
    lo devuelve sin hacer nada. make_encoder_fn() -> encoder (se libera al terminar de embeber)."""
    rec = load_json(build_path(study))
    if rec and rec.get("build") in BUILDS:
        return rec
    gk = dict(gen_kwargs or {})
    keys = list(keys) if keys else pilot_keys(check=gk.get("backend", "hf") != "fake")
    if len(keys) < 2:
        raise RuntimeError("piloto: hacen falta al menos 2 generadores con acceso (revisa HF_TOKEN y las licencias)")
    log(f"[piloto] generadores {keys} · {n} ítems · construcciones {list(BUILDS)}")
    status = {}
    for k in keys:                          # todas las tareas de un generador seguidas; pesos borrados al final
        tasks = list(dict.fromkeys(t for b in BUILD_PREF for t in BUILDS[b]))   # neg, para, edit
        status[k] = run_tasks(study, bundle.test, k, tasks, split, n_items=n, log=log, **gk)
    todo = [(k, b) for k in keys for b in BUILD_PREF if needs_embedding(study, k, split, b, f"pilot_{b}", n)]
    if todo:
        enc = make_encoder_fn()
        try:
            for k, b in todo:
                try:
                    embed_generator(study, bundle.test, k, enc, split, build=b, sub=f"pilot_{b}", n_items=n, log=log)
                except Exception as e:  # noqa: BLE001
                    log(f"[piloto] {k}/{b}: no se pudo construir ({type(e).__name__}: {e})")
        finally:
            del enc
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    results, table = {}, []
    for b in BUILD_PREF:
        results[b] = {}
        for k in keys:
            a = audit_generator(study, bundle, k, split, blind=blind, sub=f"pilot_{b}", log=log)
            if a is None:
                continue
            results[b][k] = a.get("atajo_ciego")
            table.append({"construcción": b, "generador": k, "ítems": a.get("n_items"),
                          "más_larga": a.get("longest"), "más_corta": a.get("shortest"),
                          "ciego_Gemini": a.get("blind_gemini"), "ciego_propio": a.get("blind_own_cv"),
                          "coseno_crudo": a.get("raw_cosine"), "atajo_ciego": a.get("atajo_ciego"),
                          "banda": a.get("banda")})
    dec = decide_build(results)
    pilot = {"n": n, "keys": keys, "generation": status, "results": results, "table": table}
    if dec["build"] is None:
        save_json({"build": None, "decided": None, "pilot": pilot, "motivo": dec.get("motivo")},
                  os.path.join(tgen_base(study), "BUILD_intento.json"))
        raise RuntimeError(f"piloto sin decisión: {dec.get('motivo')}")
    rec = save_build(study, dec, pilot)
    log(f"[piloto] construcción elegida: {rec['build']} ({rec['build_txt']}); medianas del atajo ciego "
        f"{ {b: round(v, 3) for b, v in rec['medianas'].items()} }")
    return rec


def build_pilot_sets(study, test, make_encoder_fn, split: str = "confirm", keys: list | None = None,
                     n: int = PILOT_N, gen_kwargs: dict | None = None, log=print) -> dict:
    """Los sets del piloto (S, P y E, mismos generadores y misma regla de ítems: los primeros n del orden del panel
    de ese split) en otro split, sin decidir nada. Los usa C2b en el test confirmatorio (ENMIENDA_03 §13).
    Reanudable: genera y embebe sólo lo que falta."""
    rec = load_json(build_path(study)) or {}
    keys = list(keys or (rec.get("pilot") or {}).get("keys") or [])
    if not keys:
        raise RuntimeError("BUILD.json no registra los generadores del piloto")
    gk = dict(gen_kwargs or {})
    tasks = list(dict.fromkeys(t for b in BUILD_PREF for t in BUILDS[b]))       # neg, para, edit
    status = {}
    for k in keys:
        try:
            status[k] = run_tasks(study, test, k, tasks, split, n_items=n, log=log, **gk)
        except Exception as e:  # noqa: BLE001
            status[k] = {"error": f"{type(e).__name__}: {e}"}
            log(f"[piloto:{split}] {k}: la generación no terminó ({type(e).__name__}: {e})")
    todo = [(k, b) for k in keys for b in BUILD_PREF if needs_embedding(study, k, split, b, f"pilot_{b}", n)]
    if todo:
        enc = make_encoder_fn()
        try:
            for k, b in todo:
                try:
                    embed_generator(study, test, k, enc, split, build=b, sub=f"pilot_{b}", n_items=n, log=log)
                except Exception as e:  # noqa: BLE001
                    log(f"[piloto:{split}] {k}/{b}: no se pudo construir ({type(e).__name__}: {e})")
        finally:
            del enc
            try:
                import gc
                import torch
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
    done = [k for k in keys if all(load_gen(study, k, split, sub=f"pilot_{b}") is not None for b in BUILD_PREF)]
    return {"keys": keys, "generation": status, "embedded": todo, "completos": done}


def synthetic_pilot_sets(study, test, split: str = "confirm", n: int = 300, seed: int = 0) -> list:
    """Versión sintética de build_pilot_sets (perfil smoke): S, P y E de los generadores del piloto en otro split,
    con los mismos artefactos que synthetic_pilot. No dice nada sobre generadores reales."""
    rec = load_json(build_path(study)) or {}
    keys = list((rec.get("pilot") or {}).get("keys") or ["syn_pa", "syn_pb", "syn_pc"])
    lat_info = _synthetic_latent(study, test)
    items = np.sort(panel_order(study, test, split)[:n])
    _synthetic_texts(study, split, items)
    arts = {"S": (0.0, 0.6), "P": (0.0, 0.0), "E": (0.6, 0.0)}
    for bj, b in enumerate(BUILD_PREF):
        for j, k in enumerate(keys):
            d = set_dir(study, k, split, f"pilot_{b}")
            if not os.path.exists(os.path.join(d, "embed_info.json")):
                _synthetic_set(study, test, d, k, items, arts[b][0], arts[b][1], seed + 7 + 1000 * bj + 17 * j, b,
                               lat_info, split)
    return keys


def synthetic_pilot(study, bundle, blind=None, n: int = 300, split: str = "test", seed: int = 0, log=print) -> dict:
    """Piloto del perfil smoke (sin LLM): sets sintéticos por construcción con artefactos distintos (S con atajo
    de longitud, E con uno fuerte en el embedding, P limpio), auditados con las mismas sondas; prueba la regla y
    los archivos. No dice nada sobre generadores reales."""
    rec = load_json(build_path(study))
    if rec and rec.get("build") in BUILDS:
        return rec
    test = bundle.test
    lat_info = _synthetic_latent(study, test)
    items = np.sort(panel_order(study, test, split)[:n])
    _synthetic_texts(study, split, items)
    arts = {"S": (0.0, 0.6), "P": (0.0, 0.0), "E": (0.6, 0.0)}
    keys = ["syn_pa", "syn_pb", "syn_pc"]
    results, table = {}, []
    for bj, b in enumerate(BUILD_PREF):
        results[b] = {}
        for j, k in enumerate(keys):
            d = set_dir(study, k, split, f"pilot_{b}")
            if not os.path.exists(os.path.join(d, "embed_info.json")):
                _synthetic_set(study, test, d, k, items, arts[b][0], arts[b][1], seed + 1000 * bj + 17 * j, b,
                               lat_info, split)
            a = audit_generator(study, bundle, k, split, blind=blind, sub=f"pilot_{b}", log=log)
            results[b][k] = a.get("atajo_ciego")
            table.append({"construcción": b, "generador": k, "ítems": a.get("n_items"), "más_larga": a.get("longest"),
                          "ciego_propio": a.get("blind_own_cv"), "atajo_ciego": a.get("atajo_ciego"),
                          "banda": a.get("banda")})
    dec = decide_build(results)
    return save_build(study, dec, {"n": n, "keys": keys, "results": results, "table": table, "synthetic": True})
