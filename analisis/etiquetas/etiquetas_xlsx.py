"""Libros de Excel para las etiquetas a mano del bloque 1 (notebook pre-envío).

A partir de ETIQUETAR_200_items.csv (celda 1.2) arma:
  * ETIQUETADAS.xlsx             — las 200 preguntas (formato que lee preenvio.read_hand_labels);
  * ETIQUETAR_<Nombre>.xlsx      — diseño en anillo: 6 tramos de ~33 preguntas; el alumno K hace los tramos K y K+1,
                                    así cada pregunta la marcan dos alumnos por separado (uno la ve al principio de su
                                    archivo y el otro al final);
  * ADJUDICAR.xlsx               — sólo los candidatos en desacuerdo, para que decida el investigador.
No agrega nada sobre el juez ni sobre el set de cada ítem. Uso:
  python3 -I etiquetas_xlsx.py profesor ETIQUETAR_200_items.csv ETIQUETADAS.xlsx
  python3 -I etiquetas_xlsx.py alumnos  ETIQUETAR_200_items.csv carpeta_salida/ [Nombre1,Nombre2,...,Nombre6]
  python3 -I etiquetas_xlsx.py csv      ETIQUETADAS.xlsx ETIQUETAR_200_items.csv   (reconstruye el CSV desde el libro)
"""
import csv as _csv
import math
import os
import sys

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.cell.cell import MergedCell
from openpyxl.comments import Comment
from openpyxl.formatting.rule import FormulaRule
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.datavalidation import DataValidation

COLS = ["id", "pregunta", "oro", "n", "candidato", "es_correcta(0/1)", "notas"]
QC = "es_correcta(0/1)"
F = "Arial"
NAVY, AMBER, GRAY_TXT = "1F3864", "BF8F00", "808080"
BLOCK_FILLS = ("FFFFFF", "F4F7FB")
ORO_FILL = "DDEBF7"
thin = Side(style="thin", color="D9D9D9")
med = Side(style="medium", color="7F7F7F")
DRIVE = "CLM_losses › colab › results › PRE › b1_calidad › etiquetas"
N_ALUMNOS = 6

REGLA = [
    ("1 = el candidato dice lo mismo que el oro: idéntico, con otras palabras, en otro formato, o con más o menos "
     "detalle sin cambiar el dato que pide la pregunta. También es 1 si es correcto por otra razón, aunque casi nunca "
     "se puede saber sin el documento.", "E2EFDA"),
    ("0 = cambia algún dato (otro nombre, número, sustancia, sentido o grado), le falta parte de lo que pide la "
     "pregunta (por ejemplo, la pregunta pide dos cosas y el candidato da una) o responde otra cosa.", "EDEDED"),
    ("Juzgas contra el oro: no hace falta saber del tema ni ver el documento. Los textos vienen del dataset, en inglés.",
     None),
]
EX_Q = "What effect did the drug have on blood pressure?"
EX_O = "It lowered blood pressure by about 10 mmHg."
EX = [("Blood pressure dropped by roughly 10 mmHg.", 1, "Mismo dato con otras palabras."),
      ("It lowered blood pressure by about 10 mmHg.", 1, "Idéntico al oro."),
      ("It reduced it by approximately ten mmHg.", 1, "Otro formato (ten en vez de 10)."),
      ("It raised blood pressure by about 10 mmHg.", 0, "Cambia el sentido (sube en vez de bajar)."),
      ("It lowered blood pressure by about 25 mmHg.", 0, "Cambia el número."),
      ("It lowered heart rate by about 10 beats per minute.", 0, "Responde otra cosa.")]


def read_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    assert list(df.columns) == COLS, list(df.columns)
    assert len(df) == 2000 and df["id"].nunique() == 200, (len(df), df["id"].nunique())
    return df


def csv_from_workbook(xlsx, dst):
    """Reconstruye ETIQUETAR_200_items.csv desde un libro con las 200 preguntas (pestaña Etiquetar): id, n y candidato
    por fila; pregunta y oro desde la primera fila de cada bloque (celdas combinadas)."""
    ws = load_workbook(xlsx)["Etiquetar"]
    hdr = [ws.cell(row=1, column=j).value for j in range(1, 8)]
    assert hdr == COLS, hdr
    rows, q, o, cur = [], None, None, None
    for r in range(2, ws.max_row + 1):
        iid = ws.cell(row=r, column=1).value
        if iid is None:
            continue
        if iid != cur:
            cur, q, o = iid, ws.cell(row=r, column=2).value, ws.cell(row=r, column=3).value
        rows.append({"id": iid, "pregunta": q, "oro": o, "n": str(ws.cell(row=r, column=4).value),
                     "candidato": ws.cell(row=r, column=5).value, QC: "", "notas": ""})
    df = pd.DataFrame(rows, columns=COLS)
    df.to_csv(dst, index=False, encoding="utf-8-sig", quoting=_csv.QUOTE_MINIMAL)
    return read_csv(dst)


def ids_all(df):
    return list(dict.fromkeys(df["id"]))


def tramos(ids, k=N_ALUMNOS):
    """6 tramos contiguos de ids (33, 34, 33, 33, 34, 33 con 200 ids)."""
    n = len(ids)
    cuts = [round(i * n / k) for i in range(k + 1)]
    return [ids[cuts[i]:cuts[i + 1]] for i in range(k)]


def asignacion(ids, k=N_ALUMNOS):
    """{alumno: [ids en orden]}: el alumno a (1..k) hace su tramo a y luego el a+1 (el k hace el k y el 1)."""
    t = tramos(ids, k)
    return {a: t[a - 1] + t[a % k] for a in range(1, k + 1)}


def rango_texto(ids):
    """'e001–e066' o 'e167–e200 y e001–e033' (tramos contiguos en el orden dado)."""
    partes, ini, prev = [], ids[0], ids[0]
    for x in ids[1:]:
        if int(x[1:]) != int(prev[1:]) + 1:
            partes.append(f"{ini}–{prev}")
            ini = x
        prev = x
    partes.append(f"{ini}–{prev}")
    return " y ".join(partes)


def _lines(text, width, k):
    return max(1, math.ceil(len(str(text)) / (width * k)))


def _set_text(cell, value):
    cell.value = value
    if isinstance(value, str) and value.startswith("="):
        cell.data_type = "s"          # texto, nunca fórmula


def build_workbook(rows: pd.DataFrame, dst: str, txt: dict) -> dict:
    """rows: columnas COLS, en el orden en que se muestran; bloques = filas seguidas con el mismo id (cualquier tamaño)."""
    assert list(rows.columns) == COLS
    rows = rows.reset_index(drop=True)
    wb = Workbook()
    wb._named_styles["Normal"].font = Font(name=F, size=10)
    ws = wb.active
    ws.title = "Etiquetar"
    W = {"A": 7, "B": 40, "C": 34, "D": 4.5, "E": 58, "F": 17, "G": 22}
    for col, w in W.items():
        ws.column_dimensions[col].width = w
    for j, name in enumerate(COLS, 1):
        c = ws.cell(row=1, column=j, value=name)
        c.font = Font(name=F, size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=AMBER if name == QC else NAVY)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        c.border = Border(bottom=med)
    ws.row_dimensions[1].height = 30
    ws["F1"].comment = Comment("Aquí escribes: 1 si este candidato TAMBIÉN responde bien la pregunta (dice lo mismo que "
                               "el oro); 0 si no. Las celdas amarillas son las que faltan. Reglas y ejemplo: pestaña "
                               "Instrucciones.", "Instrucciones", width=300, height=110)
    blocks = []
    for iid in dict.fromkeys(rows["id"]):
        g = rows[rows["id"] == iid]
        assert g.index.max() - g.index.min() + 1 == len(g), f"{iid}: filas no contiguas"
        assert (g["pregunta"].nunique(), g["oro"].nunique()) == (1, 1), iid
        blocks.append(g.reset_index(drop=True))
    r = 2
    full10 = all(len(g) == 10 for g in blocks)
    for b, g in enumerate(blocks):
        fill = PatternFill("solid", fgColor=BLOCK_FILLS[b % 2])
        top, heights = r, []
        for k, row in g.iterrows():
            bottom = med if k == len(g) - 1 else thin
            a = ws.cell(row=r, column=1, value=row["id"])
            a.font = Font(name=F, size=10, bold=(k == 0), color=None if k == 0 else GRAY_TXT)
            a.alignment = Alignment(horizontal="center", vertical="top")
            nn = ws.cell(row=r, column=4, value=int(row["n"]))
            nn.font = Font(name=F, size=10, color=GRAY_TXT)
            nn.alignment = Alignment(horizontal="center", vertical="top")
            e = ws.cell(row=r, column=5)
            _set_text(e, row["candidato"])
            e.font = Font(name=F, size=10)
            e.alignment = Alignment(vertical="top", wrap_text=True)
            f = ws.cell(row=r, column=6)
            f.font = Font(name=F, size=12, bold=True)
            f.alignment = Alignment(horizontal="center", vertical="center")
            f.fill = PatternFill("solid", fgColor="FFF2CC")
            gc = ws.cell(row=r, column=7)
            gc.font = Font(name=F, size=10)
            gc.alignment = Alignment(vertical="top", wrap_text=True)
            for col in (1, 4, 5, 7):
                ws.cell(row=r, column=col).fill = fill
            for col in (1, 4, 5, 6, 7):
                ws.cell(row=r, column=col).border = Border(left=thin, right=thin, top=thin, bottom=bottom)
            heights.append(13 * _lines(row["candidato"], W["E"], 1.25) + 4)
            r += 1
        q, o = g["pregunta"][0], g["oro"][0]
        bq = ws.cell(row=top, column=2)
        _set_text(bq, q)
        bq.font = Font(name=F, size=10)
        bq.fill = fill
        bq.alignment = Alignment(vertical="top", wrap_text=True)
        bq.border = Border(left=thin, right=thin, top=thin, bottom=med)
        bo = ws.cell(row=top, column=3)
        _set_text(bo, o)
        bo.font = Font(name=F, size=10, bold=True)
        bo.fill = PatternFill("solid", fgColor=ORO_FILL)
        bo.alignment = Alignment(vertical="top", wrap_text=True)
        bo.border = Border(left=thin, right=thin, top=thin, bottom=med)
        need = max(13 * _lines(q, W["B"], 1.15), 13 * _lines(o, W["C"], 1.0)) + 8
        if sum(heights) < need:
            heights[-1] += need - sum(heights)
        for k, h in enumerate(heights):
            ws.row_dimensions[top + k].height = h
        if len(g) > 1:
            ws.merge_cells(start_row=top, start_column=2, end_row=r - 1, end_column=2)
            ws.merge_cells(start_row=top, start_column=3, end_row=r - 1, end_column=3)
    last = r - 1
    rng = f"F2:F{last}"
    dv = DataValidation(type="list", formula1='"0,1"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Sólo 0 o 1", error="Escribe 1 si el candidato también es correcto y 0 si no.",
                        showInputMessage=False)
    dv.add(rng)
    ws.add_data_validation(dv)
    ws.conditional_formatting.add(rng, FormulaRule(formula=['F2&""="1"'], stopIfTrue=True,
                                                   fill=PatternFill("solid", fgColor="C6EFCE", bgColor="C6EFCE"),
                                                   font=Font(color="006100", bold=True)))
    ws.conditional_formatting.add(rng, FormulaRule(formula=['F2&""="0"'], stopIfTrue=True,
                                                   fill=PatternFill("solid", fgColor="EDEDED", bgColor="EDEDED"),
                                                   font=Font(color="404040", bold=True)))
    ws.freeze_panes = "A2"
    ws.sheet_properties.tabColor = AMBER
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "1:1"
    ws.sheet_view.selection[0].activeCell = "F2"
    ws.sheet_view.selection[0].sqref = "F2"

    # --- Instrucciones ----------------------------------------------------------------------------------------
    ins = wb.create_sheet("Instrucciones")
    ins.sheet_properties.tabColor = NAVY
    ins.sheet_view.showGridLines = False
    IW = {"A": 2, "B": 30, "C": 30, "D": 4.5, "E": 40, "F": 17, "G": 34}
    for col, w in IW.items():
        ins.column_dimensions[col].width = w
    TOT = sum(IW[c] for c in "BCDEFG")
    row = [1]

    def para(text, size=10, bold=False, color=None, fill=None, italic=False):
        rr = row[0]
        ins.merge_cells(start_row=rr, start_column=2, end_row=rr, end_column=7)
        c = ins.cell(row=rr, column=2, value=text)
        c.font = Font(name=F, size=size, bold=bold, color=color, italic=italic)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        if fill:
            c.fill = PatternFill("solid", fgColor=fill)
        lh = 13 if size <= 10 else round(size * 1.45)
        ins.row_dimensions[rr].height = lh * _lines(text, TOT, 1.1 if size <= 10 else 0.9) + 4
        row[0] += 1

    def blank(h=8):
        ins.row_dimensions[row[0]].height = h
        row[0] += 1

    def head(text):
        para(text, size=11, bold=True, color=NAVY)

    n_rows, n_q = last - 1, len(blocks)
    blank(10)
    para(txt["titulo"].format(n_q=n_q, n_rows=n_rows), size=14, bold=True, color=NAVY)
    para(txt["intro"])
    blank()
    head("Qué haces")
    for t in txt["pasos"]:
        para(t)
    blank()
    head("La regla (es la misma que sigue el juez)")
    for t, fl in REGLA:
        para(t, fill=fl)
    blank()
    head("Ejemplo inventado (no está en tus datos): así se ve una pregunta ya marcada")
    rr = row[0]
    for j, name in enumerate(["pregunta", "oro", "n", "candidato", QC, "por qué"], 2):
        c = ins.cell(row=rr, column=j, value=name)
        c.font = Font(name=F, size=10, bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor=AMBER if name == QC else NAVY)
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ins.row_dimensions[rr].height = 28
    row[0] += 1
    ex_top = row[0]
    for k, (cand, lab, why) in enumerate(EX, 1):
        rr = row[0]
        for col, v in {4: k, 5: cand, 6: lab, 7: why}.items():
            c = ins.cell(row=rr, column=col, value=v)
            c.font = Font(name=F, size=12 if col == 6 else 10, bold=(col == 6),
                          color=("006100" if lab == 1 else "404040") if col == 6 else None)
            c.alignment = Alignment(horizontal="center" if col in (4, 6) else None, vertical="top", wrap_text=True)
            if col == 6:
                c.fill = PatternFill("solid", fgColor="C6EFCE" if lab == 1 else "EDEDED")
            c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
        ins.row_dimensions[rr].height = 13 * max(_lines(cand, IW["E"], 1.15), _lines(why, IW["G"], 1.15)) + 6
        row[0] += 1
    for col, v, bold, fillc in ((2, EX_Q, False, None), (3, EX_O, True, ORO_FILL)):
        c = ins.cell(row=ex_top, column=col, value=v)
        c.font = Font(name=F, size=10, bold=bold)
        c.alignment = Alignment(vertical="top", wrap_text=True)
        c.border = Border(left=thin, right=thin, top=thin, bottom=thin)
        if fillc:
            c.fill = PatternFill("solid", fgColor=fillc)
        ins.merge_cells(start_row=ex_top, start_column=col, end_row=row[0] - 1, end_column=col)
    blank()
    head("Progreso (se actualiza solo)")
    FR, AR = f"Etiquetar!$F$2:$F${last}", f"Etiquetar!$A$2:$A${last}"
    m_row = row[0]
    prog = [(f"Filas marcadas, de {n_rows:,}", f'=COUNTIF({FR},"0")+COUNTIF({FR},"1")'),
            ("Filas sin marcar", f"=ROWS({FR})-C{m_row}")]
    if full10:
        prog.append((f"Preguntas completas (sus 10 candidatos marcados), de {n_q}",
                     f'=ROUND(SUMPRODUCT((COUNTIFS({AR},{AR},{FR},"<>")=10)/10),0)'))
    prog.append(("Celdas con algo distinto de 0 o 1 (tiene que ser 0)", f"=COUNTA({FR})-C{m_row}"))
    for label, formula in prog:
        rr = row[0]
        a = ins.cell(row=rr, column=2, value=label)
        a.font = Font(name=F, size=10)
        a.alignment = Alignment(vertical="center", wrap_text=True)
        v = ins.cell(row=rr, column=3, value=formula)
        v.font = Font(name=F, size=11, bold=True, color="000000")
        v.alignment = Alignment(horizontal="left", vertical="center")
        v.number_format = "#,##0"
        for col in (2, 3):
            ins.cell(row=rr, column=col).border = Border(left=thin, right=thin, top=thin, bottom=thin)
        ins.row_dimensions[rr].height = 13 * _lines(label, IW["B"], 1.15) + 6
        row[0] += 1
    blank()
    head("Al terminar")
    for t in txt["al_terminar"]:
        para(t)
    blank()
    para(txt["pie"].format(n_q=n_q, n_rows=n_rows), color=GRAY_TXT, italic=True)
    ins.page_setup.orientation = "landscape"
    ins.page_setup.fitToWidth = 1
    ins.page_setup.fitToHeight = 0
    ins.sheet_properties.pageSetUpPr.fitToPage = True
    for c in (c for rw in ins.iter_rows(min_row=1, max_row=ins.max_row, min_col=1, max_col=7) for c in rw):
        if not isinstance(c, MergedCell) and c.font.name != F:
            c.font = Font(name=F, size=c.font.sz or 10, bold=c.font.b, italic=c.font.i, color=c.font.color)
    wb.active = wb.sheetnames.index("Instrucciones")
    for s in wb.worksheets:
        s.sheet_view.tabSelected = s.title == "Instrucciones"
    wb.properties.creator = "CLM losses · notebook pre-envío"
    wb.properties.title = txt.get("doc_title", "Etiquetas a mano (bloque 1)")
    wb.save(dst)
    return {"dst": dst, "filas": n_rows, "preguntas": n_q, "progreso": (m_row, m_row + len(prog) - 1)}


# --- textos -----------------------------------------------------------------------------------------------------
PASOS_COMUNES = [
    "1. Ve a la pestaña «Etiquetar». Cada bloque es una pregunta: a la izquierda, la pregunta y el oro (en azul); "
    "a la derecha, sus candidatos.",
    "2. Para cada candidato, escribe en la columna es_correcta(0/1): 1 si también responde bien la pregunta; 0 si no. "
    "Las celdas amarillas son las que te faltan.",
]


def textos_profesor():
    return {
        "titulo": "Etiquetas a mano: {n_q} preguntas, 10 candidatos cada una ({n_rows:,} filas)",
        "intro": "Cada pregunta trae su respuesta correcta (el oro) y 10 candidatos que el test da por incorrectos. Tu "
                 "trabajo es marcar cuáles de esos candidatos en realidad también son correctos. Con tus marcas "
                 "medimos qué tan bien trabaja el juez automático, que revisó todo el test.",
        "pasos": PASOS_COMUNES + [
            "3. Marca los 10 candidatos de cada pregunta: una pregunta a medias no cuenta.",
            "4. Si dudas, decide igual (0 o 1) y escribe «duda» en notas.",
            "5. Sólo escribes en es_correcta(0/1) y en notas. No cambies id, n ni los encabezados.",
            "6. No abras la carpeta juez ni clave.json hasta terminar: tus marcas tienen que ser ciegas.",
            "7. Guarda seguido."],
        "al_terminar": [
            "1. Guarda este archivo como libro de Excel con el mismo nombre: ETIQUETADAS.xlsx. Si lo llenaste en Google "
            "Sheets, descárgalo como Microsoft Excel (.xlsx) y ponle ese nombre.",
            f"2. Súbelo a Google Drive, a la carpeta {DRIVE} (la misma donde está ETIQUETAR_200_items.csv).",
            "3. Cuando el notebook haya terminado, en Colab: Entorno de ejecución → Ejecutar todo. Se salta lo ya hecho "
            "y sólo calcula la comparación con el juez (unos minutos)."],
        "pie": "Este libro trae los mismos 2,000 renglones que ETIQUETAR_200_items.csv, en el mismo orden. El notebook "
               "lee sólo la primera pestaña (Etiquetar) y sus columnas id, n y es_correcta(0/1).",
    }


def textos_alumno(a, rango, nombre=None):
    quien = nombre or f"alumno {a}"
    archivo = f"ETIQUETAR_{nombre}.xlsx" if nombre else f"ETIQUETAR_alumno{a}.xlsx"
    return {
        "doc_title": f"Etiquetas a mano · {quien}",
        "titulo": f"Etiquetas a mano · {quien}: " + "{n_q} preguntas, 10 candidatos cada una ({n_rows:,} filas)",
        "intro": "Cada pregunta trae su respuesta correcta (el oro) y 10 candidatos que el test da por incorrectos. Tu "
                 "trabajo es marcar cuáles de esos candidatos en realidad también son correctos. Cada pregunta la "
                 "revisan dos personas por separado; con esas marcas medimos qué tan bien trabaja un juez automático. "
                 f"Tus preguntas: {rango}.",
        "pasos": PASOS_COMUNES + [
            "3. Marca los 10 candidatos de cada pregunta: una pregunta a medias no cuenta.",
            "4. Si dudas, decide igual (0 o 1) y escribe «duda» en notas.",
            "5. Sólo escribes en es_correcta(0/1) y en notas. No cambies id, n ni los encabezados.",
            "6. Trabaja solo o sola: no compares tus marcas con tus compañeros ni le preguntes a ChatGPT, Claude u "
            "otra IA. Si no entiendes una palabra, búscala en un diccionario. Otra persona revisa las mismas "
            "preguntas por separado: tus marcas tienen que ser sólo tuyas.",
            "7. Guarda seguido. Es más o menos una hora; puedes hacerlo en varias sesiones."],
        "al_terminar": [
            (f"1. Guárdalo con el mismo nombre: {archivo}. Si lo llenaste en Google Sheets, descárgalo como Microsoft "
             "Excel (.xlsx) con ese nombre." if nombre else
             f"1. Guarda el archivo agregando tu nombre al final, por ejemplo ETIQUETAR_alumno{a}_Ana.xlsx. Si lo "
             "llenaste en Google Sheets, descárgalo como Microsoft Excel (.xlsx)."),
            "2. Mándaselo a tu profesor. No lo compartas con nadie más ni lo subas a otro lado."],
        "pie": "Este archivo trae {n_q} de las 200 preguntas del estudio.",
    }


def textos_adjudicar():
    return {
        "doc_title": "Etiquetas a mano · desacuerdos",
        "titulo": "Desacuerdos para decidir: {n_rows:,} candidatos de {n_q} preguntas",
        "intro": "Estos candidatos los marcaron distinto los dos alumnos que revisaron su pregunta. Decide tú con la "
                 "misma regla, sin ver lo que dijo el juez. Sólo aparecen los candidatos en desacuerdo, no los 10.",
        "pasos": PASOS_COMUNES + [
            "3. Si dudas, decide igual (0 o 1) y escribe «duda» en notas.",
            "4. No cambies id, n ni los encabezados. No abras la carpeta juez ni clave.json."],
        "al_terminar": ["1. Guarda el archivo como ADJUDICADAS.xlsx y súbelo a la conversación con Claude: con eso "
                        "arma el ETIQUETADAS.xlsx final para Drive."],
        "pie": "Las marcas de los alumnos y la combinación están en REPORTE_ACUERDO.md.",
    }


def make_profesor(csv, dst):
    return build_workbook(read_csv(csv), dst, textos_profesor())


def make_alumnos(csv, outdir, nombres=None):
    """nombres: lista de 6 nombres (el 1.º es el alumno 1, etc.); sin nombres, ETIQUETAR_alumnoK.xlsx."""
    df = read_csv(csv)
    os.makedirs(outdir, exist_ok=True)
    out = {}
    if nombres is not None:
        assert len(nombres) == N_ALUMNOS and len(set(nombres)) == N_ALUMNOS, nombres
    for a, ids in asignacion(ids_all(df)).items():
        sub = pd.concat([df[df["id"] == i] for i in ids], ignore_index=True)
        rango = rango_texto(ids)
        nombre = nombres[a - 1] if nombres else None
        fn = f"ETIQUETAR_{nombre}.xlsx" if nombre else f"ETIQUETAR_alumno{a}.xlsx"
        info = build_workbook(sub, os.path.join(outdir, fn), textos_alumno(a, rango, nombre))
        out[a] = {**info, "rango": rango, "nombre": nombre}
    return out


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "profesor":
        print(make_profesor(sys.argv[2], sys.argv[3]))
    elif mode == "alumnos":
        nombres = sys.argv[4].split(",") if len(sys.argv) > 4 else None
        for a, v in make_alumnos(sys.argv[2], sys.argv[3], nombres).items():
            print(a, v)
    elif mode == "csv":
        df = csv_from_workbook(sys.argv[2], sys.argv[3])
        print("CSV reconstruido:", sys.argv[3], len(df), "filas,", df["id"].nunique(), "preguntas")
    else:
        raise SystemExit("modo: profesor | alumnos | csv")
