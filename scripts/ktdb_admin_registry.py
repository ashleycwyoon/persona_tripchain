#!/usr/bin/env python3
"""Read the KTDB 2021 administrative-code workbook without an Excel dependency.

Numeric codes are retained only as internal join keys.  Model-facing labels and
whitelists are deterministic English renderings of the workbook names.
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterator
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tripchain.paths import SHARED_DATA_ROOT, TRIPCHAIN_1015_DATA as DATA_ROOT  # noqa: E402

from korean_admin_english import romanize_name


DEFAULT_WORKBOOK = SHARED_DATA_ROOT / "행정동코드_20210726(말소코드포함).xlsx"
AS_OF_DATE = "20210726"
_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
_NSMAP = {"m": _NS}


def normalize_code(value: object) -> str:
    text = str(value or "").strip()
    return text[:-2] if text.endswith(".0") and text[:-2].isdigit() else text


def _xlsx_rows(path: Path) -> Iterator[dict[str, str]]:
    with zipfile.ZipFile(path) as archive:
        strings_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        strings = [
            "".join(node.text or "" for node in item.iterfind(".//m:t", _NSMAP))
            for item in strings_root.findall("m:si", _NSMAP)
        ]
        with archive.open("xl/worksheets/sheet1.xml") as sheet:
            for _, row in ET.iterparse(sheet, events=("end",)):
                if row.tag != f"{{{_NS}}}row":
                    continue
                values: dict[str, str] = {}
                for cell in row.findall("m:c", _NSMAP):
                    column = re.match(r"[A-Z]+", cell.attrib["r"]).group()
                    value_node = cell.find("m:v", _NSMAP)
                    value = "" if value_node is None else value_node.text or ""
                    if cell.attrib.get("t") == "s" and value:
                        value = strings[int(value)]
                    values[column] = value.strip()
                yield values
                row.clear()


def load_admin_registry(path: Path = DEFAULT_WORKBOOK, as_of: str = AS_OF_DATE) -> dict[str, dict[str, str]]:
    """Return active administrative-dong code -> Korean/English label metadata."""
    registry: dict[str, dict[str, str]] = {}
    for row_number, row in enumerate(_xlsx_rows(Path(path)), 1):
        if row_number == 1:
            expected = {
                "A": "행정동코드",
                "B": "시도명",
                "C": "시군구명",
                "D": "읍면동명",
                "G": "생성일자",
                "H": "말소일자",
            }
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"unexpected administrative workbook header: {row}")
            continue
        code = normalize_code(row.get("A"))
        sido, sigungu, dong = row.get("B", ""), row.get("C", ""), row.get("D", "")
        # Sejong has no lower si/gun/gu column in this workbook.  Use the
        # special self-governing city itself as its external-zone label.
        sigungu = sigungu or sido
        created, deleted = row.get("G", ""), row.get("H", "")
        if not (code and sido and sigungu and dong):
            continue
        if created and created > as_of:
            continue
        if deleted and deleted <= as_of:
            continue
        record = {
            "sido_ko": sido,
            "sigungu_ko": sigungu,
            "dong_ko": dong,
            "sido": romanize_name(sido),
            "sigungu": romanize_name(sigungu),
            "administrative_dong_name": romanize_name(dong),
        }
        previous = registry.get(code)
        if previous is not None and previous != record:
            raise ValueError(f"conflicting active labels for administrative code {code}")
        registry[code] = record
    if not registry:
        raise ValueError(f"no active administrative dongs found in {path}")
    return registry


def build_model_whitelist(registry: dict[str, dict[str, str]]) -> dict:
    seoul: dict[str, set[str]] = {}
    external: set[tuple[str, str]] = set()
    for item in registry.values():
        if item["sido_ko"] == "서울특별시":
            seoul.setdefault(item["sigungu"], set()).add(item["administrative_dong_name"])
        else:
            external.add((item["sido"], item["sigungu"]))
    return {
        "source": str(DEFAULT_WORKBOOK),
        "as_of": AS_OF_DATE,
        "seoul": {gu: sorted(dongs) for gu, dongs in sorted(seoul.items())},
        "external_sigungu": [{"sido": sido, "sigungu": sigungu} for sido, sigungu in sorted(external)],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--as-of", default=AS_OF_DATE)
    parser.add_argument("--out", type=Path, default=DATA_ROOT / "processed/ktdb_location_whitelist.json")
    args = parser.parse_args()
    registry = load_admin_registry(args.workbook, args.as_of)
    whitelist = build_model_whitelist(registry)
    whitelist["source"] = str(args.workbook)
    whitelist["as_of"] = args.as_of
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(whitelist, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "active_admin_dongs": len(registry),
                "seoul_gu": len(whitelist["seoul"]),
                "seoul_admin_dongs": sum(map(len, whitelist["seoul"].values())),
                "external_sigungu": len(whitelist["external_sigungu"]),
                "output": str(args.out),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
