"""Brand words from OCR readings, and word-based similarity between product crops."""
from __future__ import annotations

import difflib
import re
from collections import Counter

import numpy as np

# Generic packaging, category and marketing words. They are never treated as brand names.
GENERIC = set("""
SHAMPOO CONDITIONER EXTRA FREE NEW PACK PACKS SOFT ULTRA SUPER MINI MINIS MAXI NIGHT DAY WITH FOR AND THE NET WT
ML GMS KG OFFER SAVE PRICE MRP BUY GET BEST FRESH NATURAL ORIGINAL CLASSIC FLAVOUR FLAVOR CHOCOLATE BISCUIT BISCUITS
COOKIES CREAM CREME SANDWICH PANTS DIAPERS DIAPER PADS WINGS REGULAR LARGE SMALL MEDIUM SIZE TISSUE TISSUES FACIAL
WASH FACE HAIR BODY LOTION OIL POWDER DETERGENT LIQUID BACON SAUSAGE RANCH ITALIAN DRESSING CAESAR FRENCH VINEGAR
WHISKEY WHISKY BOURBON VODKA WINE RED WHITE BLUE GREEN BLACK GOLD PINK STRENGTH KIDS BABY CARE PROTECTION HERBAL
COTTON STRONG THICK CUT SMOKED PREMIUM SPECIAL VALUE FAMILY JUMBO COUNT SHEETS PLY ROLLS INSTANT MIX DRINK MILK
HEALTH NUTRITION GROWTH BITES DISKON GRATIS HARGA PRODUCT PRODUCTS FORMULA DAILY PLUS PRO MAX LIGHT GENTLE
AIRY DRY SKIN CLEAN CLEAR ANTI GERM HAND SOAP ADULT LADY MAN MEN WOMEN EDITION LIMITED
""".split())


def brand_tokens(reads: list[dict], min_len: int, min_conf: float, prominence: float) -> set[str]:
    """Words from the largest text on the pack (brand names are usually the biggest print)."""
    reads = [r for r in reads if r["conf"] >= min_conf and r["text"].strip()]
    if not reads:
        return set()
    tallest = max(r["height"] for r in reads)
    toks = set()
    for r in reads:
        if r["height"] < prominence * tallest:
            continue
        for w in re.findall(r"[A-Za-z0-9&']+", r["text"].upper()):
            w = w.replace("'", "")
            if len(w) >= min_len and sum(ch.isalpha() for ch in w) >= 0.6 * len(w) and w not in GENERIC:
                toks.add(w)
    return toks


def tokens_match(a: str, b: str, fuzzy: float) -> bool:
    return a == b or (min(len(a), len(b)) >= 5 and difflib.SequenceMatcher(None, a, b).ratio() >= fuzzy)


def propagate_tokens(tokens: list[set[str]], V: np.ndarray, min_sim: float) -> list[set[str]]:
    """An unreadable crop inherits the words of crops that look practically identical to it (one hop only)."""
    if min_sim <= 0:
        return tokens
    read = [i for i, t in enumerate(tokens) if t]
    out = [set(t) for t in tokens]
    for i, t in enumerate(tokens):
        if not t:
            for j in read:
                if V[i, j] >= min_sim:
                    out[i] |= tokens[j]
    return out


def text_similarity(tokens: list[set[str]], fuzzy: float) -> tuple[np.ndarray, np.ndarray]:
    """T[i, j] = matched words / size of the smaller word set; has[i] = crop i has brand words."""
    n = len(tokens)
    has = np.array([bool(t) for t in tokens], bool)
    T = np.zeros((n, n))
    idx = np.flatnonzero(has)
    for a in range(len(idx)):
        i = idx[a]
        T[i, i] = 1.0
        for b in range(a + 1, len(idx)):
            j = idx[b]
            matched = sum(1 for x in tokens[i] if any(tokens_match(x, y, fuzzy) for y in tokens[j]))
            T[i, j] = T[j, i] = matched / min(len(tokens[i]), len(tokens[j]))
    return T, has


def group_labels(labels: np.ndarray, tokens: list[set[str]]) -> dict[int, str | None]:
    """Most common brand word per group (shown next to the group id), or None."""
    out = {}
    for g in range(int(labels.max()) + 1 if len(labels) else 0):
        c = Counter(t for i in np.flatnonzero(labels == g) for t in tokens[i])
        out[g] = c.most_common(1)[0][0] if c else None
    return out
