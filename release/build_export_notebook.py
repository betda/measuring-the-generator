"""Write release/Exportar_datos.ipynb, the Colab notebook that runs export_release_data.py on the study folder in Drive.

The notebook carries a copy of the script, so it runs without cloning this repository.
tests/test_export_release_data.py checks that the copy matches the script; run this file again after changing it.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "export_release_data.py")
OUT = os.path.join(HERE, "Exportar_datos.ipynb")
WRITEFILE = "%%writefile /content/export_release_data.py\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True)}


INTRO = """
# Exportar los datos del estudio sin textos

*Export the study's data without any text (CC BY 4.0).*

1. **Entorno de ejecución → Ejecutar todo.** Colab pide permiso para tu Google Drive: acéptalo con la cuenta donde está la carpeta `CLM_losses`.
2. Espera a que termine. Recorre todas las corridas y puede tardar de 10 a 30 minutos.
3. Al final quedan en tu Drive, en `CLM_losses`, las partes del archivo de datos (`release_data.tar.xz.001`, `.002`, ...), `release_data.SHA256SUMS` y el reporte `release_data_REPORT.md`. No tienes que descargar nada: avísale a Claude.

No se exporta ningún texto: ni las preguntas y respuestas de Nemotron-CC, ni los negativos de CLM, ni los negativos generados. Tampoco los checkpoints ni los embeddings. El script revisa cada archivo antes de copiarlo y deja un reporte (`REPORT.md`) de lo que copió y de lo que dejó fuera.

La celda 2 escribe una copia de `release/export_release_data.py`, el script del repositorio; una prueba del repositorio verifica que la copia sea idéntica.
"""

MOUNT = """
#@title 1 · Montar Drive y ubicar la carpeta del estudio
from google.colab import drive
drive.mount('/content/drive')
import os
BASE = '/content/drive/MyDrive/CLM_losses'
ROOT = next((p for p in (os.path.join(BASE, 'colab'), BASE)
             if all(os.path.isdir(os.path.join(p, s)) for s in ('runs', 'results'))), None)
assert ROOT, f'No encontré las carpetas runs/ y results/ en {BASE}/colab ni en {BASE}.'
print('Carpeta del estudio:', ROOT)
"""

RUN = """
#@title 3 · Exportar (sin textos; sin los arreglos de puntajes, para que el archivo sea liviano)
!python /content/export_release_data.py --root "{ROOT}" --out /content/release_data --no-scores
"""

SAVE = """
#@title 4 · Guardar las partes del archivo y el reporte en Drive
import glob, shutil
from IPython.display import Markdown, display
for old in glob.glob(os.path.join(BASE, 'release_data.tar.xz.[0-9][0-9][0-9]')):
    os.remove(old)   # partes de una corrida anterior de este cuaderno
partes = sorted(glob.glob('/content/release_data.tar.xz.[0-9][0-9][0-9]'))
for p in partes + ['/content/release_data.SHA256SUMS']:
    shutil.copy(p, os.path.join(BASE, os.path.basename(p)))
shutil.copy('/content/release_data/REPORT.md', os.path.join(BASE, 'release_data_REPORT.md'))
mb = os.path.getsize('/content/release_data.tar.xz') / 1e6
print(f'Listo: {len(partes)} partes ({mb:.1f} MB en total) en {BASE}. No tienes que descargar nada: avísale a Claude.')
display(Markdown(open('/content/release_data/REPORT.md', encoding='utf-8').read()))
"""


def build():
    script = open(SCRIPT, encoding="utf-8").read()
    copy = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": (WRITEFILE + script).splitlines(keepends=True)}
    nb = {"cells": [md(INTRO), code(MOUNT), copy, code(RUN), code(SAVE)],
          "metadata": {"colab": {"provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 0}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    return OUT


if __name__ == "__main__":
    print("wrote", build())
