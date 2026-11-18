# Sensibilidad del panel T-gen en F3 (post hoc, descriptivo)

**Qué responde.** ¿Cuánto dependen C1 (parte T-gen) y C2 de cada generador del macro congelado? Motivo: en el test confirmatorio, Phi-4 cruzó a banda alta (0.312) y la ventaja de InfoNCE sobre VICReg es muy distinta según el generador (EXPERIMENT_LOG, EXP-011 y DP-08).

**Qué calcula.** Para los 5 generadores del macro (el control: tiene que reproducir el confirmatorio), sin cada uno y cada uno solo:

- C1 (parte T-gen): `C1_best_noncontrastive` del freeze − InfoNCE;
- C2 por par (InfoNCE+HN − InfoNCE y SigLIP+HN − SigLIP): Δ T-gem, Δ T-gen y C2.

Usa el mismo bootstrap pareado por documento que el confirmatorio (`stats.contrast`, B = 4000, semilla 0) y no lleva Holm: no es una prueba. No cambia ningún veredicto; el macro congelado se mantiene.

**Cómo correrlo.**

- **Colab:** `CLM_losses_Analisis_F3.ipynb`, Ejecutar todo (basta con CPU, ~5 min).
- **Local**, con el Drive del estudio montado:

```python
import importlib.util
from clm_losses.pipeline import setup
spec = importlib.util.spec_from_file_location("sp", "analisis/F3_sensibilidad/sensibilidad_panel.py")
sp = importlib.util.module_from_spec(spec); spec.loader.exec_module(sp)
study = setup("colab", drive_base="/content/drive/MyDrive/CLM_losses", cache_root="/content/cache")
res = sp.panel_sensitivity(study)
```

**Salida:** `results/F3/sensibilidad_panel/sensibilidad.json` y `SENSIBILIDAD.md`. Sólo lee las corridas.

**Pruebas:** `tests/test_sensibilidad.py`.

**Corrida real (2026-10-07; EXPERIMENT_LOG, EXP-012).**

- C2 no depende del panel: +18.0 a +19.7 pp en los 11 subconjuntos.
- C1 en T-gen mantiene la dirección sin cualquiera de los 5 generadores (−1.4 a −2.7 pp). Su tamaño depende del generador: de −4.7 pp con Phi-4 a +0.5 con Mistral Small.

**Nota sobre el control.** En esa corrida, el control imprimió "IC95 NO" aunque los puntos eran idénticos. Los IC difieren en ≤ 2.2e-7 porque el bootstrap multiplica en float32, y con valores no enteros (macro, C2) el IC depende del orden de suma (IMPLEMENTACION_02 §18.2). Una diferencia menor que 1e-6 en los IC cuenta como idéntica. El script no se cambió, para que el JSON guardado siga siendo reproducible.
