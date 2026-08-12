#!/usr/bin/env python3
"""CP5: insert the new Results subsection 3.5 (Site-Held-Out Performance under
Leave-One-Site-Out Evaluation), including Table 6, into the unpacked working
copy's word/document.xml. Reads Table 6 values from loso_table6_source.csv
(itself derived from canonical, frozen loso_static_v1 outputs in CP4); does
not hard-code scientific figures beyond what that CSV contains. Table style
(borders, shading, font size) is cloned from the existing Table 4 in the
manuscript for visual consistency, not invented.
"""
from __future__ import annotations

import html
import os
import re
from pathlib import Path

import pandas as pd

PHASE_DIR = Path(__file__).resolve().parents[1]
UNPACKED_DIR = PHASE_DIR / "work" / "unpacked"
DOC_XML = UNPACKED_DIR / "word" / "document.xml"

SITE_ORDER = ["NYU", "Peking", "NeuroIMAGE", "OHSU"]


def new_id() -> str:
    return format(int.from_bytes(os.urandom(4), "big") & 0x7FFFFFFF, "X").zfill(8)


def fmt_cell_no_pct(point: float, lo: float, hi: float) -> str:
    return f"{point * 100:.1f} [{lo * 100:.1f}, {hi * 100:.1f}]"


def build_table_xml(table6: pd.DataFrame) -> str:
    col_widths = [2400, 1600, 1600, 1600, 1630]
    headers = [
        "Held-out site",
        "BNN-12 AUC [95% CI]",
        "LOSO Logistic-12",
        "BNN-116",
        "LOSO Logistic-116",
    ]

    def tc(text, width, bold=False, shaded=False, center=True):
        shd = '<w:shd w:val="clear" w:color="auto" w:fill="EDEDED"/>' if shaded else ""
        jc = '<w:pPr><w:jc w:val="center"/></w:pPr>' if center else ""
        b = "<w:b/><w:bCs/>" if bold else ""
        return (
            f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>{shd}'
            '<w:tcMar><w:top w:w="40" w:type="dxa"/><w:left w:w="80" w:type="dxa"/>'
            '<w:bottom w:w="40" w:type="dxa"/><w:right w:w="80" w:type="dxa"/></w:tcMar>'
            '<w:vAlign w:val="center"/></w:tcPr>'
            f'<w:p w14:paraId="{new_id()}" w14:textId="77777777" w:rsidR="0059799F" w:rsidRDefault="00000000">{jc}'
            f'<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/>{b}'
            f'<w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr><w:t xml:space="preserve">{html.escape(str(text))}</w:t></w:r></w:p></w:tc>'
        )

    header_row = (
        f'<w:tr w:rsidR="0090502D" w14:paraId="{new_id()}" w14:textId="77777777">'
        '<w:trPr><w:cantSplit/><w:tblHeader/></w:trPr>'
        + "".join(tc(h, w, bold=True, shaded=True) for h, w in zip(headers, col_widths))
        + "</w:tr>"
    )

    data_rows = []
    for _, r in table6.iterrows():
        label = f'{r["held_out_site"]} (held-out n={int(r["held_out_n"])})'
        cells = [
            tc(label, col_widths[0], center=False),
            tc(fmt_cell_no_pct(r["bnn12_auc_point"], r["bnn12_auc_ci_low"], r["bnn12_auc_ci_high"]), col_widths[1]),
            tc(fmt_cell_no_pct(r["logreg12_auc_point"], r["logreg12_auc_ci_low"], r["logreg12_auc_ci_high"]), col_widths[2]),
            tc(fmt_cell_no_pct(r["bnn116_auc_point"], r["bnn116_auc_ci_low"], r["bnn116_auc_ci_high"]), col_widths[3]),
            tc(fmt_cell_no_pct(r["logreg116_auc_point"], r["logreg116_auc_ci_low"], r["logreg116_auc_ci_high"]), col_widths[4]),
        ]
        data_rows.append(f'<w:tr w:rsidR="0090502D" w14:paraId="{new_id()}" w14:textId="77777777"><w:trPr><w:cantSplit/></w:trPr>' + "".join(cells) + "</w:tr>")

    tbl_pr = (
        '<w:tblPr><w:tblW w:w="8830" w:type="dxa"/><w:tblBorders>'
        '<w:top w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:left w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:right w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="auto"/></w:tblBorders>'
        '<w:tblCellMar><w:left w:w="10" w:type="dxa"/><w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
        '<w:tblLook w:val="04A0" w:firstRow="1" w:lastRow="0" w:firstColumn="1" w:lastColumn="0" w:noHBand="0" w:noVBand="1"/></w:tblPr>'
    )
    tbl_grid = "<w:tblGrid>" + "".join(f'<w:gridCol w:w="{w}"/>' for w in col_widths) + "</w:tblGrid>"

    return f"<w:tbl>{tbl_pr}{tbl_grid}{header_row}{''.join(data_rows)}</w:tbl>"


def body_paragraph(text: str) -> str:
    pid, tid = new_id(), new_id()
    return (
        f'<w:p w14:paraId="{pid}" w14:textId="{tid}" w:rsidR="0059799F" w:rsidRDefault="000962EB">'
        '<w:pPr><w:pStyle w:val="Prrafodelista"/><w:spacing w:before="240" w:line="360" w:lineRule="auto"/><w:ind w:left="0"/><w:jc w:val="both"/>'
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:sz w:val="24"/></w:rPr>'
        f'<w:t xml:space="preserve">{html.escape(text, quote=False)}</w:t></w:r></w:p>'
    )


def caption_paragraph(text: str) -> str:
    """Bold caption paragraph, matching the style of existing table captions
    (e.g. 'Table 5. Paired differences ...') found elsewhere in the document:
    List Paragraph, non-justified, bold run."""
    pid, tid = new_id(), new_id()
    return (
        f'<w:p w14:paraId="{pid}" w14:textId="{tid}" w:rsidR="0059799F" w:rsidRDefault="000962EB">'
        '<w:pPr><w:pStyle w:val="Prrafodelista"/><w:spacing w:before="120" w:after="240" w:line="360" w:lineRule="auto"/><w:ind w:left="0"/>'
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:b/><w:bCs/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman"/><w:b/><w:bCs/><w:sz w:val="22"/><w:szCs w:val="22"/></w:rPr>'
        f'<w:t xml:space="preserve">{html.escape(text, quote=False)}</w:t></w:r></w:p>'
    )


def heading_paragraph(text: str) -> str:
    pid, tid = new_id(), new_id()
    return (
        f'<w:p w14:paraId="{pid}" w14:textId="{tid}" w:rsidR="0059799F" w:rsidRDefault="00000000">'
        '<w:pPr><w:pStyle w:val="Prrafodelista"/><w:numPr><w:ilvl w:val="1"/><w:numId w:val="7"/></w:numPr>'
        '<w:spacing w:before="240" w:after="120" w:line="360" w:lineRule="auto"/><w:ind w:left="0"/><w:jc w:val="both"/>'
        '<w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr></w:pPr>'
        '<w:r><w:rPr><w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:cs="Times New Roman"/><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>'
        f'<w:t xml:space="preserve">{html.escape(text, quote=False)}</w:t></w:r></w:p>'
    )


def main() -> None:
    table6 = pd.read_csv(PHASE_DIR / "loso_table6_source.csv")
    table6 = table6.set_index("held_out_site").loc[SITE_ORDER].reset_index()

    para1 = Path("/tmp/results_para1_v2.txt").read_text().strip()
    para2 = Path("/tmp/results_para2.txt").read_text().strip()
    para3 = Path("/tmp/results_para3.txt").read_text().strip()
    para4 = Path("/tmp/results_para4.txt").read_text().strip()

    caption_text = (
        "Table 6. Leave-one-site-out AUC by held-out site, ROI dimensionality, and model "
        "across the four observed acquisition sites. Held-out sample sizes are shown in the "
        "site labels. Values are AUC [95% CI]. Confidence intervals were obtained from 10,000 "
        "class-stratified participant-bootstrap resamples within each held-out site and are "
        "pointwise, unadjusted, and conditional on the fixed source-site composition and frozen "
        "LOSO predictions for that rotation. BrainNetCNN point estimates are means of five "
        "seed-specific AUCs; logistic-regression estimates are from a single deterministic fit. "
        "No pooled estimate across sites was calculated. For BrainNetCNN, NYU had historical "
        "exposure during configuration development and its held-out-site result is therefore not "
        "development-independent (see Methods). LOSO partition sizes are detailed in "
        "Supplementary Table S_LOSO_Design."
    )

    heading_xml = heading_paragraph(
        "Site-Held-Out Performance under Leave-One-Site-Out Evaluation"
    )
    para1_xml = body_paragraph(para1)
    table_xml = build_table_xml(table6)
    caption_xml = caption_paragraph(caption_text)
    para2_xml = body_paragraph(para2)
    para3_xml = body_paragraph(para3)
    para4_xml = body_paragraph(para4)

    insertion = heading_xml + para1_xml + table_xml + caption_xml + para2_xml + para3_xml + para4_xml

    data = DOC_XML.read_text(encoding="utf-8")
    anchor = (
        "The diagonal dashed line indicates chance-level discrimination (AUC = 0.50)."
        "</w:t></w:r></w:p>"
    )
    count = data.count(anchor)
    if count != 1:
        raise SystemExit(f"STOP: anchor found {count} times, expected 1.")
    data = data.replace(anchor, anchor + insertion, 1)
    DOC_XML.write_text(data, encoding="utf-8")
    print("Inserted Results section 3.5 (heading + 4 paragraphs + Table 6 + caption).")


if __name__ == "__main__":
    main()
