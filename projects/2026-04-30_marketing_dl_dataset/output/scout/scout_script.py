"""Create Scout outputs from the real Online Retail II workbook.

The project input folder currently has a Windows ACL deny rule that can block
copying the workbook into input/. To keep Scout reproducible, this script first
looks in input/ and then falls back to the known staging location.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "input" / "online_retail_II.xlsx"
STAGING_INPUT = Path(r"C:\Users\amorn\.codex\memories\online_retail_ii_work\online_retail_II.xlsx")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "scout"
MAIN_NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def cell_col_index(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha())
    index = 0
    for ch in letters:
        index = index * 26 + ord(ch.upper()) - 64
    return index - 1


def resolve_input(path_arg: str) -> tuple[Path, str]:
    candidates = []
    if path_arg:
        candidates.append(Path(path_arg))
    candidates.extend([DEFAULT_INPUT, STAGING_INPUT])

    for candidate in candidates:
        if candidate.exists():
            source = "project_input" if candidate == DEFAULT_INPUT else "staging_fallback"
            return candidate, source

    searched = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Online Retail II workbook not found. Searched: {searched}")


def load_shared_strings(workbook: zipfile.ZipFile) -> list[str]:
    shared_strings = []
    with workbook.open("xl/sharedStrings.xml") as strings_file:
        for _, elem in ET.iterparse(strings_file, events=("end",)):
            if elem.tag.endswith("}si"):
                text_parts = [node.text for node in elem.iter() if node.tag.endswith("}t") and node.text]
                shared_strings.append("".join(text_parts))
                elem.clear()
    return shared_strings


def profile_workbook(xlsx_path: Path) -> dict[str, object]:
    with zipfile.ZipFile(xlsx_path) as workbook:
        shared_strings = load_shared_strings(workbook)
        sheets = [
            ("Year 2009-2010", "xl/worksheets/sheet1.xml"),
            ("Year 2010-2011", "xl/worksheets/sheet2.xml"),
        ]

        headers: list[str] | None = None
        rows = 0
        non_missing: list[int] | None = None
        cancel_rows = 0
        countries = set()
        sheet_rows = {}

        for sheet_name, sheet_path in sheets:
            data_rows_in_sheet = 0
            with workbook.open(sheet_path) as sheet_file:
                for _, row in ET.iterparse(sheet_file, events=("end",)):
                    if not row.tag.endswith("}row"):
                        continue

                    row_number = int(row.attrib.get("r", "0"))
                    values = {}
                    for cell in row:
                        if not cell.tag.endswith("}c"):
                            continue
                        value_node = cell.find(f"{MAIN_NS}v")
                        if value_node is None:
                            continue
                        value = value_node.text
                        if cell.attrib.get("t") == "s":
                            value = shared_strings[int(value)]
                        values[cell_col_index(cell.attrib.get("r", ""))] = value

                    if row_number == 1 and headers is None:
                        headers = [values.get(i, "") for i in range(max(values) + 1)]
                        non_missing = [0] * len(headers)
                    elif row_number > 1 and headers is not None and non_missing is not None:
                        rows += 1
                        data_rows_in_sheet += 1
                        for index in range(len(headers)):
                            if values.get(index) not in (None, ""):
                                non_missing[index] += 1
                        invoice = values.get(0)
                        if isinstance(invoice, str) and invoice.upper().startswith("C"):
                            cancel_rows += 1
                        if values.get(7):
                            countries.add(values[7])

                    row.clear()

            sheet_rows[sheet_name] = data_rows_in_sheet

    if headers is None or non_missing is None:
        raise ValueError(f"No worksheet headers found in {xlsx_path}")

    missing = {name: rows - count for name, count in zip(headers, non_missing) if rows - count}
    return {
        "dataset": "Online Retail II",
        "source": "https://archive.ics.uci.edu/dataset/502/online+retail+ii",
        "domain": "retail/customer_behavior",
        "target_options": "Customer ID CLV, basket analysis, Quantity/Price regression",
        "rows": rows,
        "cols": len(headers),
        "columns": headers,
        "sheet_rows": sheet_rows,
        "missing": missing,
        "cancel_rows": cancel_rows,
        "countries": len(countries),
        "dl_ready": rows >= 50000,
    }


def row_values(row: ET.Element, shared_strings: list[str]) -> dict[int, str]:
    values = {}
    for cell in row:
        if not cell.tag.endswith("}c"):
            continue

        cell_type = cell.attrib.get("t")
        value = ""

        if cell_type == "inlineStr":
            text_nodes = [node.text for node in cell.iter() if node.tag.endswith("}t") and node.text]
            value = "".join(text_nodes)
        else:
            value_node = cell.find(f"{MAIN_NS}v")
            if value_node is None:
                continue
            value = value_node.text or ""
            if cell_type == "s":
                value = shared_strings[int(value)]

        values[cell_col_index(cell.attrib.get("r", ""))] = value
    return values


def export_workbook_csv(xlsx_path: Path, csv_path: Path) -> int:
    sheets = [
        "xl/worksheets/sheet1.xml",
        "xl/worksheets/sheet2.xml",
    ]

    rows_written = 0
    headers: list[str] | None = None

    with zipfile.ZipFile(xlsx_path) as workbook:
        shared_strings = load_shared_strings(workbook)

        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer: csv.writer | None = None

            for sheet_path in sheets:
                with workbook.open(sheet_path) as sheet_file:
                    for _, row in ET.iterparse(sheet_file, events=("end",)):
                        if not row.tag.endswith("}row"):
                            continue

                        row_number = int(row.attrib.get("r", "0"))
                        values = row_values(row, shared_strings)

                        if row_number == 1:
                            if headers is None:
                                headers = [values.get(i, "") for i in range(max(values) + 1)]
                                writer = csv.writer(csv_file)
                                writer.writerow(headers)
                            row.clear()
                            continue

                        if headers is None or writer is None:
                            raise ValueError(f"No worksheet headers found before data rows in {sheet_path}")

                        writer.writerow([values.get(i, "") for i in range(len(headers))])
                        rows_written += 1
                        row.clear()

    if headers is None:
        raise ValueError(f"No worksheet headers found in {xlsx_path}")

    return rows_written


def write_outputs(profile: dict[str, object], input_path: Path, input_source: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "scout_output.csv"
    rows_written = export_workbook_csv(input_path, csv_path)
    if rows_written != profile["rows"]:
        raise ValueError(f"Exported {rows_written:,} rows, but profile counted {profile['rows']:,} rows")

    print(f"[STATUS] CSV saved: {csv_path}")
    print(f"[STATUS] CSV rows written: {rows_written:,}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args, _ = parser.parse_known_args()

    input_path, input_source = resolve_input(args.input)
    output_dir = Path(args.output_dir) if args.output_dir else DEFAULT_OUTPUT_DIR
    profile = profile_workbook(input_path)
    write_outputs(profile, input_path, input_source, output_dir)
    print("[STATUS] Script completed successfully")


if __name__ == "__main__":
    main()
