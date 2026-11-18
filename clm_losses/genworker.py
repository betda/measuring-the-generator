"""Subproceso de generación de negativos para el panel T-gen (uno por generador).

    python -m clm_losses.genworker generate --model REPO --prompts prompts.jsonl --out raw.jsonl [opciones]

Lee prompts {"i", "prompt", "gold", "max_new_tokens"} y agrega a `raw.jsonl` un registro por ítem e intento:
{"i", "attempt", "raw", "negatives"}. Reanudable: salta los ítems que ya tienen ≥ k_keep respuestas o que
agotaron sus intentos. Corre en un proceso aparte para que cada generador libere la GPU al terminar y para
usar la versión de transformers recién instalada aunque el notebook ya la hubiera importado.

Backends:
    hf     transformers (por defecto; el que se usa en el estudio)
    vllm   vLLM (opcional, NO probado en este entorno; su instalación puede cambiar torch)
    fake   salidas sintéticas deterministas (sólo para pruebas del código)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time


def _read_jsonl(path):
    out = []
    if not os.path.exists(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return out


def _append(path, recs):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pre = ""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb") as f:                 # un corte pudo dejar la última línea a medias
            f.seek(-1, os.SEEK_END)
            pre = "" if f.read(1) == b"\n" else "\n"
    with open(path, "a") as f:
        f.write(pre + "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs))
        f.flush()


# --------------------------------------------------------------------------------------
END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<|end|>", "<end_of_turn>", "<|endoftext|>", "</s>", "<|end_of_text|>",
              "<|user|>", "<|observation|>", "<|END_OF_TURN_TOKEN|>", "<|im_start|>", "<turn|>")


def final_text(text: str, marker: str, start: str | None = None, control=()) -> str:
    """Respuesta final de una salida con bloque de razonamiento (p. ej. Gemma 4: "<|channel>thought ...<channel|>"):
    lo que sigue al ÚLTIMO marcador de cierre. Si hay un bloque abierto sin cerrar, la salida se descarta (""):
    nunca se parsea razonamiento como si fueran respuestas. Sin bloque, el texto entero. Luego se quitan los
    tokens de control que hayan quedado (fin de turno, relleno)."""
    j = text.rfind(marker)
    if j >= 0:
        out = text[j + len(marker):]
    elif start and start in text:
        return ""
    else:
        out = text
    for c in control:
        out = out.replace(c, "")
    return out


def is_oom(e: BaseException) -> bool:
    return type(e).__name__ == "OutOfMemoryError" or "out of memory" in str(e).lower()


def _cfg_attr(cfg, name):
    """Atributo de una config sin romper con configs heterogéneas por capa: en transformers recientes, leer el
    valor global de un atributo que varía por capa (p. ej. num_key_value_heads en Gemma 4) lanza
    AmbiguousGlobalPerLayerAttributeError, que no es AttributeError. Devuelve None si no se puede leer."""
    if cfg is None:
        return None
    if isinstance(cfg, dict):
        return cfg.get(name)
    try:
        return getattr(cfg, name, None)
    except Exception:  # noqa: BLE001
        return None


def _layer_cfgs(cfg) -> list:
    """Configs por capa (configs heterogéneas); [] si la config es homogénea o no las trae."""
    for attr in ("per_layer_config", "per_layer_configs", "layer_configs"):
        v = _cfg_attr(cfg, attr)
        if v is None or isinstance(v, (str, bytes)):
            continue
        if isinstance(v, dict):
            v = [v[k] for k in sorted(v, key=lambda x: (0, int(x)) if str(x).isdigit() else (1, str(x)))]
        try:
            v = list(v)
        except TypeError:
            continue
        if v:
            return v
    return []


def kv_bytes_per_token(cfg, dtype_bytes: int = 2) -> int | None:
    """Bytes de caché K/V por token, sumando todas las capas (K y V, en bf16 por defecto). Con una config
    heterogénea suma capa por capa (lo que no está en la capa se toma del valor global, si se puede leer);
    con una homogénea, n_capas × cabezas K/V × head_dim. None si falta algo: quien llama usa entonces el batch
    pedido y deja que el manejo de memoria insuficiente lo baje a la mitad."""
    tc = _cfg_attr(cfg, "text_config") or cfg

    def per_layer(lc):
        heads = _cfg_attr(lc, "num_attention_heads") or _cfg_attr(tc, "num_attention_heads")
        n_kv = _cfg_attr(lc, "num_key_value_heads") or _cfg_attr(tc, "num_key_value_heads") or heads
        hd = _cfg_attr(lc, "head_dim") or _cfg_attr(tc, "head_dim")
        if not hd:
            hidden = _cfg_attr(lc, "hidden_size") or _cfg_attr(tc, "hidden_size")
            hd = (int(hidden) // int(heads)) if hidden and heads else None
        return int(n_kv) * int(hd) if n_kv and hd else None

    try:
        layers = _layer_cfgs(tc)
        if layers:
            vals = [per_layer(lc) for lc in layers]
            if any(v is None for v in vals):
                return None
            return int(2 * sum(vals) * dtype_bytes)
        n_l = _cfg_attr(tc, "num_hidden_layers")
        v = per_layer(None)
        return int(2 * int(n_l) * v * dtype_bytes) if n_l and v else None
    except Exception:  # noqa: BLE001
        return None


def load_model(repo: str, td, token, mk: dict):
    """AutoModelForCausalLM; si la arquitectura sólo está registrada como multimodal (p. ej. Mistral 3), se carga
    con AutoModelForImageTextToText y se usa sólo con texto."""
    from transformers import AutoModelForCausalLM
    try:
        return AutoModelForCausalLM.from_pretrained(repo, dtype=td, device_map="auto", token=token, **mk).eval()
    except ValueError as e:
        if "Unrecognized configuration class" not in str(e):
            raise
        from transformers import AutoModelForImageTextToText
        print("[genworker] arquitectura multimodal: se carga con AutoModelForImageTextToText (sólo texto)", flush=True)
        return AutoModelForImageTextToText.from_pretrained(repo, dtype=td, device_map="auto", token=token, **mk).eval()


class HFBackend:
    def __init__(self, repo: str, dtype: str, chat_kwargs: dict, harmony: bool, model_kwargs: dict | None = None,
                 final_marker: str | None = None, reasoning_start: str | None = None):
        import torch
        from transformers import AutoTokenizer
        token = os.environ.get("HF_TOKEN") or None
        self.torch = torch
        self.harmony = harmony
        self.final_marker, self.reasoning_start = final_marker, reasoning_start
        self.last_raw = None
        self.chat_kwargs = dict(chat_kwargs or {})
        self.tok = AutoTokenizer.from_pretrained(repo, token=token)
        if getattr(self.tok, "chat_template", None) is None:
            # algunos modelos multimodales guardan la plantilla sólo en el procesador
            try:
                from transformers import AutoProcessor
                self.tok.chat_template = getattr(AutoProcessor.from_pretrained(repo, token=token), "chat_template", None)
            except Exception:  # noqa: BLE001
                pass
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.control = sorted({t for t in set(self.tok.all_special_tokens) | set(self.tok.get_added_vocab())
                               if len(t) > 2 and t.startswith("<") and t.endswith(">")}, key=len, reverse=True)
        td = "auto" if dtype == "auto" else getattr(torch, dtype)
        mk = dict(model_kwargs or {})
        if mk.pop("dequantize_mxfp4", False):
            # gpt-oss trae pesos MXFP4; sus kernels no cubren todas las GPU (p. ej. Blackwell de estación de
            # trabajo, sm_120). Se des-cuantiza a bf16 (~42 GB): más lento, pero corre en cualquier GPU grande.
            from transformers import Mxfp4Config
            mk["quantization_config"] = Mxfp4Config(dequantize=True)
        self.model = load_model(repo, td, token, mk)
        gc = self.model.generation_config
        eos = gc.eos_token_id if gc.eos_token_id is not None else self.tok.eos_token_id
        eos = set(eos if isinstance(eos, (list, tuple)) else [eos]) - {None}
        if self.tok.eos_token_id is not None:
            eos.add(self.tok.eos_token_id)
        if not harmony:
            # fin de turno de cada familia, si el tokenizer lo tiene como token especial: algunos
            # generation_config no lo incluyen y el modelo seguiría escribiendo el turno siguiente
            special = set(self.tok.all_special_tokens) | set(self.tok.get_added_vocab())
            for t in END_TOKENS:
                if t in special:
                    i = self.tok.convert_tokens_to_ids(t)
                    if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                        eos.add(i)
        self.eos = sorted(eos)
        self.bos = gc.bos_token_id

    def max_batch(self, seq_len: int, cap: int) -> int:
        """Batch que cabe en la memoria libre de la GPU para secuencias de seq_len tokens (caché K/V en bf16,
        con margen). Sin GPU, o si la config no trae lo necesario, devuelve cap."""
        torch = self.torch
        if not torch.cuda.is_available():
            return cap
        per_tok = kv_bytes_per_token(getattr(self.model, "config", None))   # robusto a configs por capa (Gemma 4)
        if not per_tok:
            return cap
        free = min(torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count()))
        per_seq = per_tok * seq_len * 1.3
        return int(max(1, min(cap, 0.85 * free // per_seq)))

    def free(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def chat(self, prompt: str) -> str:
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.chat_kwargs)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, max_new_tokens: int, temperature: float, top_p: float, seed: int,
                 greedy: bool = False):
        import copy
        torch = self.torch
        texts = [self.chat(p) for p in prompts]
        enc = self.tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
        # muestreo idéntico para todos los generadores: sólo temperature y top-p (sin top-k, min-p, typical-p
        # ni penalización de repetición, aunque el generation_config del modelo traiga otros valores). Se parte
        # de su generation_config para conservar lo propio de la arquitectura (fin de secuencia, tipo de caché).
        gcfg = copy.deepcopy(self.model.generation_config)
        if greedy:      # v0.3.8, juez de falsos negativos: decodificación voraz (sin muestreo)
            gcfg.update(do_sample=False, temperature=None, top_p=None, top_k=None, min_p=None, typical_p=None,
                        repetition_penalty=1.0, no_repeat_ngram_size=0, num_beams=1,
                        max_new_tokens=int(max_new_tokens), pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos)
        else:
            gcfg.update(do_sample=True, temperature=temperature, top_p=top_p, top_k=0, min_p=None, typical_p=1.0,
                        repetition_penalty=1.0, no_repeat_ngram_size=0, num_beams=1,
                        max_new_tokens=int(max_new_tokens), pad_token_id=self.tok.pad_token_id, eos_token_id=self.eos)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        with torch.no_grad():
            out = self.model.generate(**enc, generation_config=gcfg)
        gen = out[:, enc["input_ids"].shape[1]:]
        if self.final_marker:
            full = self.tok.batch_decode(gen, skip_special_tokens=False)
            if self.tok.pad_token:
                full = [t.replace(self.tok.pad_token, "") for t in full]   # relleno tras el fin de cada salida
            self.last_raw = full
            texts = [final_text(t, self.final_marker, self.reasoning_start, self.control) for t in full]
        else:
            self.last_raw = None
            texts = self.tok.batch_decode(gen, skip_special_tokens=not self.harmony)
        ended = torch.isin(gen, torch.tensor(self.eos, device=gen.device)).any(1).tolist()
        # si se cortó por max_new_tokens (sin fin de secuencia), la última línea puede estar a medias: se descarta
        return [t if e else (t.rsplit("\n", 1)[0] if "\n" in t else "") for t, e in zip(texts, ended)]


class VLLMBackend:
    """Opcional; NO probado en el entorno de desarrollo (sin GPU)."""

    def __init__(self, repo: str, dtype: str, chat_kwargs: dict, harmony: bool, seed: int = 0,
                 final_marker: str | None = None, reasoning_start: str | None = None):
        from vllm import LLM
        self.harmony = harmony
        self.final_marker, self.reasoning_start = final_marker, reasoning_start
        self.last_raw = None
        self.chat_kwargs = dict(chat_kwargs or {})
        self.llm = LLM(model=repo, dtype=("auto" if dtype == "auto" else dtype), seed=seed,
                       max_model_len=int(os.environ.get("TGEN_VLLM_MAX_LEN", "6144")),
                       gpu_memory_utilization=float(os.environ.get("TGEN_VLLM_GPU_MEM", "0.85")))
        self.tok = self.llm.get_tokenizer()

    def chat(self, prompt):
        msgs = [{"role": "user", "content": prompt}]
        try:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **self.chat_kwargs)
        except TypeError:
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    def generate(self, prompts, max_new_tokens, temperature, top_p, seed, greedy: bool = False):
        from vllm import SamplingParams
        sp = SamplingParams(temperature=0.0 if greedy else temperature, top_p=1.0 if greedy else top_p, top_k=-1,
                            max_tokens=int(max_new_tokens), seed=seed,
                            skip_special_tokens=not (self.harmony or self.final_marker))
        outs = self.llm.generate([self.chat(p) for p in prompts], sp, use_tqdm=False)
        texts = [o.outputs[0].text for o in outs]
        if self.final_marker:
            self.last_raw = texts
            control = sorted({t for t in set(self.tok.all_special_tokens) | set(self.tok.get_added_vocab())
                              if len(t) > 2 and t.startswith("<") and t.endswith(">")}, key=len, reverse=True)
            texts = [final_text(t, self.final_marker, self.reasoning_start, control) for t in texts]
        return texts


class FakeBackend:
    """Salidas deterministas con los defectos típicos (preámbulos, numeración, duplicados, copias del oro,
    respuestas de menos) para probar parseo, reintentos y reanudación. Sólo pruebas."""

    def __init__(self, harmony: bool = False, short_every: int = 7, oom_above: int | None = None):
        self.harmony = harmony
        self.short_every = short_every
        self.oom_above = oom_above
        self.batches = []

    def generate(self, prompts, max_new_tokens, temperature, top_p, seed, greedy: bool = False):
        if self.oom_above is not None and len(prompts) > self.oom_above:
            raise RuntimeError(f"CUDA out of memory (simulado: batch {len(prompts)} > {self.oom_above})")
        self.batches.append(len(prompts))
        out = []
        for p in prompts:
            if JUDGE_MARK in p:                               # juez falso: etiquetas deterministas (sólo pruebas)
                n = p.count("\n[") or 10
                h = int(hashlib.sha1(p.encode()).hexdigest()[:8], 16)
                if h % 11 == 0 and "Remember:" not in p:      # a veces no respeta el formato (prueba el reintento)
                    out.append("I think candidate 2 is correct.")
                    continue
                out.append("\n".join(f"{j + 1}: {1 if (h >> j) % 13 == 0 else 0}" for j in range(n)))
                continue
            h = int(hashlib.sha1((p + str(seed)).encode()).hexdigest()[:8], 16)
            gold = p.split("Correct answer: ", 1)[1].split("\n", 1)[0] if "Correct answer: " in p else "x"
            k = 14 if h % self.short_every else 6            # a veces devuelve de menos (prueba los reintentos)
            lines = ["Here are 14 incorrect answers:"] if h % 3 == 0 else []
            for j in range(k):
                words = " ".join(f"w{(h >> (j % 16)) % 97}_{t}" for t in range(3 + (h + j) % 9))
                pre = f"{j + 1}. " if h % 2 else ""
                lines.append(f"{pre}Wrong claim {j} {words}")
            lines.append(gold)                                # copia del oro: el parser debe descartarla
            lines.append(lines[-2])                           # duplicado: también
            text = "\n".join(lines)
            if self.harmony:
                text = "<|channel|>analysis<|message|>thinking...<|end|><|start|>assistant<|channel|>final<|message|>" \
                       + text + "<|return|>"
            out.append(text)
        return out


# --------------------------------------------------------------------------------------
# v0.3.8: juez de falsos negativos (notebook pre-envío, bloque 1). Un prompt por ítem con sus candidatos; el juez
# marca 1 si un candidato que el set da por incorrecto es en realidad otra respuesta correcta. Decodificación voraz.
JUDGE_PROMPT_VERSION = "fn-judge-v1"
JUDGE_MARK = "You are auditing a question-answering benchmark for false negatives."
JUDGE_PROMPT = (
    JUDGE_MARK + "\n\n"
    "Question: {question}\n"
    "Reference answer (correct): {answer}\n\n"
    "The benchmark labels each candidate below as a WRONG answer to the question. For each candidate, decide whether "
    "that label is a mistake, that is, whether the candidate is in fact ALSO a correct answer to the question: it "
    "states the same fact as the reference answer (possibly with other words, more or less detail, or another "
    "format), or it is otherwise fully correct for this question. Answer 1 if the candidate is also correct and 0 if "
    "it is wrong, partly wrong, or answers something else.\n\n"
    "Candidates:\n{candidates}\n\n"
    "Reply with exactly {n} lines, one per candidate, in the format \"<number>: <0 or 1>\", and nothing else."
)
JUDGE_REMINDER = "\n\nRemember: reply with exactly {n} lines such as \"1: 0\", one per candidate, with no other text."


def judge_prompt(question: str, answer: str, candidates) -> str:
    cands = "\n".join(f"[{j + 1}] {' '.join(str(c).split())}" for j, c in enumerate(candidates))
    return JUDGE_PROMPT.format(question=" ".join(str(question).split()), answer=" ".join(str(answer).split()),
                               candidates=cands, n=len(candidates))


_JUDGE_LINE = None


def parse_judge(text: str, n: int):
    """Etiquetas 0/1 de los n candidatos ("1: 0", "[2] - 1", "3. 0", ...); None si falta alguno o hay basura."""
    import re
    global _JUDGE_LINE
    if _JUDGE_LINE is None:
        _JUDGE_LINE = re.compile(r"^\s*(?:candidate\s*)?\[?(\d{1,2})\]?\s*[:.)=\-]?\s*\[?([01])\]?\s*[.;,]?\s*$",
                                 re.IGNORECASE)
    if not text:
        return None
    got = {}
    for line in str(text).splitlines():
        s = line.strip().strip("`*").strip()
        if not s:
            continue
        m = _JUDGE_LINE.match(s)
        if not m:
            continue
        j = int(m.group(1))
        if 1 <= j <= n and j not in got:
            got[j] = int(m.group(2))
    if len(got) != n:
        return None
    return [got[j] for j in range(1, n + 1)]


def cmd_judge(args) -> int:
    """Juez: lee prompts {"i", "prompt", "n", "max_new_tokens"} y agrega a `out` {"i", "attempt", "raw", "labels"}.
    Reanudable: salta los ítems que ya tienen etiquetas válidas o que agotaron sus intentos. El segundo intento
    agrega un recordatorio del formato (con decodificación voraz, repetir el mismo prompt daría lo mismo)."""
    prompts = _read_jsonl(args.prompts)
    att, ok = {}, set()
    for r in _read_jsonl(args.out):
        att[r["i"]] = att.get(r["i"], 0) + 1
        if r.get("labels") is not None:
            ok.add(r["i"])
    todo_total = [p for p in prompts if p["i"] not in ok and att.get(p["i"], 0) < args.max_attempts]
    print(f"[genworker] juez: {len(prompts)} ítems; pendientes {len(todo_total)}", flush=True)
    if not todo_total:
        return 0
    t0 = time.time()
    backend = make_backend(args)
    print(f"[genworker] modelo cargado en {time.time() - t0:.0f} s", flush=True)
    done_items, t1 = 0, time.time()
    bs = args.batch
    for attempt in range(args.max_attempts):
        todo = [p for p in prompts if p["i"] not in ok and att.get(p["i"], 0) == attempt]
        if not todo:
            continue
        todo.sort(key=lambda p: len(p["prompt"]))
        b0 = 0
        while b0 < len(todo):
            nb = bs
            if hasattr(backend, "max_batch"):
                head = todo[b0:b0 + bs]
                seq = max(len(p["prompt"]) for p in head) // 3 + 160 + max(p["max_new_tokens"] for p in head)
                nb = backend.max_batch(seq, bs)
            chunk = todo[b0:b0 + nb]
            texts_in = [p["prompt"] + (JUDGE_REMINDER.format(n=p["n"]) if attempt else "") for p in chunk]
            try:
                texts = backend.generate(texts_in, max(p["max_new_tokens"] for p in chunk), 1.0, 1.0, args.seed,
                                         greedy=True)
            except Exception as e:  # noqa: BLE001
                if not is_oom(e) or len(chunk) == 1:
                    raise
                bs = max(1, len(chunk) // 2)
                if hasattr(backend, "free"):
                    backend.free()
                print(f"[genworker] memoria insuficiente con batch {len(chunk)}: sigue con {bs}", flush=True)
                continue
            recs = []
            for p, t in zip(chunk, texts):
                lab = parse_judge(t, int(p["n"]))
                recs.append({"i": p["i"], "attempt": attempt, "raw": t, "labels": lab})
                att[p["i"]] = att.get(p["i"], 0) + 1
                if lab is not None:
                    ok.add(p["i"])
            _append(args.out, recs)
            b0 += len(chunk)
            done_items += len(chunk)
            rate = done_items / max(1e-9, time.time() - t1) * 60
            print(f"[genworker] juez, intento {attempt + 1}: {b0}/{len(todo)} ítems · batch {len(chunk)} · "
                  f"{rate:.0f} ítems/min", flush=True)
    print(f"[genworker] juez terminado: {len(ok)}/{len(prompts)} ítems con etiquetas válidas en "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


def make_backend(args):
    ck = json.loads(args.chat_kwargs or "{}")
    if args.backend == "hf":
        return HFBackend(args.model, args.dtype, ck, args.harmony, json.loads(args.model_kwargs or "{}"),
                         args.final_marker, args.reasoning_start)
    if args.backend == "vllm":
        return VLLMBackend(args.model, args.dtype, ck, args.harmony, args.seed, args.final_marker, args.reasoning_start)
    if args.backend == "fake":
        return FakeBackend(args.harmony, oom_above=args.fake_oom_above)
    raise ValueError(args.backend)


# --------------------------------------------------------------------------------------
def cmd_generate(args) -> int:
    from clm_losses.generators import parse_negatives
    prompts = _read_jsonl(args.prompts)
    att, best = {}, {}
    for r in _read_jsonl(args.out):
        att[r["i"]] = att.get(r["i"], 0) + 1
        best[r["i"]] = max(best.get(r["i"], 0), len(r.get("negatives", [])))
    todo_total = [p for p in prompts if best.get(p["i"], 0) < args.k_keep and att.get(p["i"], 0) < args.max_attempts]
    print(f"[genworker] {len(prompts)} ítems; pendientes {len(todo_total)}", flush=True)
    if not todo_total:
        return 0
    t0 = time.time()
    backend = make_backend(args)
    print(f"[genworker] modelo cargado en {time.time() - t0:.0f} s", flush=True)
    done_items, t1 = 0, time.time()
    bs = args.batch                       # baja si falta memoria; no vuelve a subir en este proceso
    for attempt in range(args.max_attempts):
        todo = [p for p in prompts if best.get(p["i"], 0) < args.k_keep and att.get(p["i"], 0) == attempt]
        if not todo:
            continue
        todo.sort(key=lambda p: p["max_new_tokens"])          # menos relleno: largos parecidos en cada batch
        b0 = 0
        while b0 < len(todo):
            nb = bs
            if hasattr(backend, "max_batch"):
                head = todo[b0:b0 + bs]
                seq = max(len(p["prompt"]) for p in head) // 3 + 96 + max(p["max_new_tokens"] for p in head)
                nb = backend.max_batch(seq, bs)
            chunk = todo[b0:b0 + nb]
            seed = (args.seed * 1_000_003 + attempt * 10_007 + b0) % (2 ** 31)
            try:
                texts = backend.generate([p["prompt"] for p in chunk], max(p["max_new_tokens"] for p in chunk),
                                         args.temperature, args.top_p, seed)
            except Exception as e:  # noqa: BLE001
                if not is_oom(e) or len(chunk) == 1:
                    raise
                bs = max(1, len(chunk) // 2)
                if hasattr(backend, "free"):
                    backend.free()
                print(f"[genworker] memoria insuficiente con batch {len(chunk)}: sigue con {bs}", flush=True)
                continue
            raws = getattr(backend, "last_raw", None) or texts
            recs = []
            for p, t, r in zip(chunk, texts, raws):
                negs = parse_negatives(t, p["gold"], args.k, harmony=args.harmony)
                recs.append({"i": p["i"], "attempt": attempt, "raw": r, "negatives": negs})
                att[p["i"]] = att.get(p["i"], 0) + 1
                best[p["i"]] = max(best.get(p["i"], 0), len(negs))
            _append(args.out, recs)
            b0 += len(chunk)
            done_items += len(chunk)
            rate = done_items / max(1e-9, time.time() - t1) * 60
            print(f"[genworker] intento {attempt + 1}: {b0}/{len(todo)} ítems · batch {len(chunk)} · "
                  f"{rate:.0f} ítems/min", flush=True)
    complete = sum(1 for p in prompts if best.get(p["i"], 0) >= args.k_keep)
    print(f"[genworker] terminado: {complete}/{len(prompts)} ítems con ≥ {args.k_keep} respuestas en "
          f"{(time.time() - t0) / 60:.1f} min", flush=True)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="clm_losses.genworker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--prompts", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--backend", default="hf", choices=["hf", "vllm", "fake"])
    g.add_argument("--batch", type=int, default=64)
    g.add_argument("--temperature", type=float, default=0.8)
    g.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    g.add_argument("--k", type=int, default=14)
    g.add_argument("--k-keep", dest="k_keep", type=int, default=10)
    g.add_argument("--max-attempts", dest="max_attempts", type=int, default=2)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--dtype", default="bfloat16")
    g.add_argument("--chat-kwargs", dest="chat_kwargs", default="{}")
    g.add_argument("--model-kwargs", dest="model_kwargs", default="{}")
    g.add_argument("--harmony", action="store_true")
    g.add_argument("--final-marker", dest="final_marker", default=None)
    g.add_argument("--reasoning-start", dest="reasoning_start", default=None)
    g.add_argument("--fake-oom-above", dest="fake_oom_above", type=int, default=None, help=argparse.SUPPRESS)
    j = sub.add_parser("judge")            # v0.3.8: juez de falsos negativos (decodificación voraz)
    j.add_argument("--model", required=True)
    j.add_argument("--prompts", required=True)
    j.add_argument("--out", required=True)
    j.add_argument("--backend", default="hf", choices=["hf", "vllm", "fake"])
    j.add_argument("--batch", type=int, default=64)
    j.add_argument("--max-attempts", dest="max_attempts", type=int, default=2)
    j.add_argument("--seed", type=int, default=0)
    j.add_argument("--dtype", default="bfloat16")
    j.add_argument("--chat-kwargs", dest="chat_kwargs", default="{}")
    j.add_argument("--model-kwargs", dest="model_kwargs", default="{}")
    j.add_argument("--fake-oom-above", dest="fake_oom_above", type=int, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.cmd == "generate":
        return cmd_generate(args)
    if args.cmd == "judge":
        args.harmony, args.final_marker, args.reasoning_start = False, None, None
        return cmd_judge(args)
    return 2


if __name__ == "__main__":
    sys.exit(main())
