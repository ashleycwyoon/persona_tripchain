"""Deterministic ASCII rendering of Korean administrative-area names.

The conversion follows Revised Romanization spelling rules closely enough for
stable model inputs. Province-level names use their established English forms.
The original administrative code remains the canonical location identifier.
"""
from __future__ import annotations

import re

SIDO_EN = {
    "서울": "Seoul",
    "서울특별시": "Seoul",
    "부산광역시": "Busan",
    "대구광역시": "Daegu",
    "인천광역시": "Incheon",
    "광주광역시": "Gwangju",
    "대전광역시": "Daejeon",
    "울산광역시": "Ulsan",
    "세종특별자치시": "Sejong",
    "경기도": "Gyeonggi-do",
    "강원도": "Gangwon-do",
    "강원특별자치도": "Gangwon-do",
    "충청북도": "Chungcheongbuk-do",
    "충청남도": "Chungcheongnam-do",
    "전라북도": "Jeollabuk-do",
    "전북특별자치도": "Jeonbuk-do",
    "전라남도": "Jeollanam-do",
    "경상북도": "Gyeongsangbuk-do",
    "경상남도": "Gyeongsangnam-do",
    "제주특별자치도": "Jeju-do",
}

_INITIAL = ("g", "kk", "n", "d", "tt", "r", "m", "b", "pp", "s", "ss", "", "j", "jj", "ch", "k", "t", "p", "h")
_VOWEL = ("a", "ae", "ya", "yae", "eo", "e", "yeo", "ye", "o", "wa", "wae", "oe", "yo", "u", "wo", "we", "wi", "yu", "eu", "ui", "i")
_FINAL = ("", "k", "k", "k", "n", "n", "n", "t", "l", "k", "m", "p", "l", "l", "p", "l", "m", "p", "p", "t", "t", "ng", "t", "t", "k", "t", "p", "t")
_FINAL_JAMO = ("", "g", "kk", "gs", "n", "nj", "nh", "d", "r", "rg", "rm", "rb", "rs", "rt", "rp", "rh", "m", "b", "bs", "s", "ss", "ng", "j", "ch", "k", "t", "p", "h")


def _syllable_parts(character: str) -> tuple[int, int, int] | None:
    code = ord(character) - 0xAC00
    if not 0 <= code < 11172:
        return None
    return code // 588, (code % 588) // 28, code % 28


def _romanize_hangul_word(word: str) -> str:
    result: list[str] = []
    for index, character in enumerate(word):
        parts = _syllable_parts(character)
        if parts is None:
            result.append(character if character.isascii() else "-")
            continue
        initial, vowel, final = parts
        next_parts = _syllable_parts(word[index + 1]) if index + 1 < len(word) else None
        onset = _INITIAL[initial]
        coda = _FINAL[final]

        # Liaison before a vowel-initial syllable: 역이 -> yeogi, not yeoki.
        if final and next_parts and next_parts[0] == 11 and final != 21:
            movable = {1: "g", 4: "n", 7: "d", 8: "r", 16: "m", 17: "b", 19: "s", 22: "j", 23: "ch", 24: "k", 25: "t", 26: "p", 27: "h"}
            if final in movable:
                coda = ""
                # The next silent onset is handled on its own iteration; append
                # the carried consonant here so its following vowel joins it.
                result.append(onset + _VOWEL[vowel] + movable[final])
                continue

        # Common liquid/nasal assimilations used in official place spellings.
        if next_parts:
            next_initial = next_parts[0]
            if final == 4 and next_initial == 5:  # ㄴ + ㄹ -> ll (신림 -> Sillim)
                coda = "l"
            elif final == 8 and next_initial == 2:  # ㄹ + ㄴ -> ll
                coda = "l"
            elif next_initial == 5 and final not in (0, 8):  # 종로 -> Jongno
                pass
        result.append(onset + _VOWEL[vowel] + coda)

    rendered = "".join(result)
    # Cross-syllable pronunciation spellings frequent in Korean place names.
    rendered = re.sub(r"ngro", "ngno", rendered)
    rendered = re.sub(r"lri", "lli", rendered)
    rendered = re.sub(r"lr(a|eo|e|o|u)", lambda m: "ll" + m.group(1), rendered)
    rendered = re.sub(r"nri", "lli", rendered)
    rendered = re.sub(r"nra", "lla", rendered)
    rendered = re.sub(r"nr(eo|e|o|u)", lambda m: "ll" + m.group(1), rendered)
    rendered = re.sub(r"mri", "mni", rendered)
    rendered = re.sub(r"pri", "mni", rendered)
    return rendered


def romanize_name(value: str | None) -> str | None:
    if value is None or str(value).strip() in {"", "None", "nan"}:
        return None
    source = str(value).strip()
    if source in SIDO_EN:
        return SIDO_EN[source]

    # Administrative 제 before a number is a numbering marker, not part of the
    # proper name: 자양제4동 -> Jayang 4-dong.
    source = re.sub(r"제(?=\d)", "", source)
    suffixes = {"동": "dong", "읍": "eup", "면": "myeon", "리": "ri", "가": "ga", "구": "gu", "군": "gun", "시": "si", "도": "do"}

    pieces: list[str] = []
    for token in source.split():
        suffix = None
        if token and token[-1] in suffixes:
            suffix = suffixes[token[-1]]
            token = token[:-1]
        numbered_ga = re.match(r"^(.*?)(\d+(?:\.\d+)*)(가)$", token)
        match = re.match(r"^(.*?)(\d+(?:\.\d+)*)$", token)
        if numbered_ga:
            base, number, _ = numbered_ga.groups()
            rendered = _romanize_hangul_word(base).strip("-")
            rendered = f"{rendered} {number}-ga" if rendered else f"{number}-ga"
        elif match:
            base, number = match.groups()
            rendered = _romanize_hangul_word(base).strip("-")
            rendered = f"{rendered} {number}" if rendered else number
        else:
            rendered = _romanize_hangul_word(token).strip("-")
        if suffix:
            rendered = f"{rendered}-{suffix}" if rendered else suffix
        pieces.append(rendered[:1].upper() + rendered[1:] if rendered else rendered)

    text = " ".join(pieces)
    text = re.sub(r"[^A-Za-z0-9. -]+", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -")
    return text or None


def english_location(location: dict) -> dict:
    """Return a location object with no Korean region-name text."""
    return {
        **location,
        "administrative_dong_name": romanize_name(location.get("administrative_dong_name")),
        "gu": romanize_name(location.get("gu")),
        "sido": romanize_name(location.get("sido")),
    }
