"""Profile Online Retail II workbook without requiring openpyxl.

This fallback parser reads the xlsx package XML directly. It exists because
the current environment has pandas but not openpyxl, and package installation
is unavailable.
"""

from __future__ import annotations

import json
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


XLSX_PATH = Path(r"C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_II.xlsx")


def cell_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - 64
    return index - 1


def main() -> None:
    with zipfile.ZipFile(XLSX_PATH) as workbook:
        shared_strings = []
        for _, elem in ET.iterparse(workbook.open("xl/sharedStrings.xml"), events=("end",)):
            if elem.tag.endswith("}si"):
                text_parts = [node.text for node in elem.iter() if node.tag.endswith("}t") and node.text]
                shared_strings.append("".join(text_parts))
                elem.clear()

        sheets = [
            ("Year 2009-2010", "xl/worksheets/sheet1.xml"),
            ("Year 2010-2011", "xl/worksheets/sheet2.xml"),
        ]
        headers = None
        rows = 0
        non_missing = None
        cancel_rows = 0
        countries = set()

        for _, sheet_path in sheets:
            row_number = 0
            for _, row in ET.iterparse(workbook.open(sheet_path), events=("end",)):
                if not row.tag.endswith("}row"):
                    continue
                values = {}
                for cell in row:
                    if not cell.tag.endswith("}c"):
                        continue
                    value_node = cell.find("{http://schemas.openxmlformats.org/spreadsheetml/2006/main}v")
                    if value_node is None:
                        continue
                    value = value_node.text
                    if cell.attrib.get("t") == "s":
                        value = shared_strings[int(value)]
                    values[cell_col_index(cell.attrib.get("r", ""))] = value

                if row_number == 0 and headers is None:
                    headers = [values.get(i, "") for i in range(max(values) + 1)]
                    non_missing = [0] * len(headers)
                elif row_number > 0:
                    rows += 1
                    for index in range(len(headers)):
                        if values.get(index) not in (None, ""):
                            non_missing[index] += 1
                    invoice = values.get(0)
                    if isinstance(invoice, str) and invoice.upper().startswith("C"):
                        cancel_rows += 1
                    if values.get(7):
                        countries.add(values[7])
                row_number += 1
                row.clear()

    missing = {name: rows - count for name, count in zip(headers, non_missing) if rows - count}
    print(json.dumps({
        "rows": rows,
        "cols": len(headers),
        "columns": headers,
        "missing": missing,
        "cancel_rows": cancel_rows,
        "countries": len(countries),
    }, indent=2))


if __name__ == "__main__":
    main()
