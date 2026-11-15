"""Field-level scoring for invoice-extraction predictions.

Usage:
    python score_invoices.py test.jsonl base_preds.jsonl finetuned_preds.jsonl

test.jsonl        : your test file (original {"input","output"} or cleaned "conversations" format)
invoice_schema.json must sit next to this script (strict schema check).
*_preds.jsonl     : one line per test row, same order, {"prediction": "<raw model text>"}

Generate predictions with the SAME prompt template you used for training,
greedy decoding (do_sample=False) and max_new_tokens ~1200.
"""
import json, re, sys, os
from datetime import datetime
import jsonschema  # pip install jsonschema

_SCHEMA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "invoice_schema.json")
VALIDATOR = jsonschema.Draft202012Validator(json.load(open(_SCHEMA))) if os.path.exists(_SCHEMA) else None


def parse(text):
    """Try to pull a JSON object out of raw model output."""
    if isinstance(text, dict):
        return text
    text = str(text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)  # model added text around the JSON
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return None


def norm_text(v):
    return re.sub(r"[^a-z0-9]", "", str(v).lower()) if v not in (None, "") else None


def norm_num(v):
    if v in (None, ""):
        return None
    try:
        return round(abs(float(str(v).replace(",", "").replace("$", ""))), 2)
    except ValueError:
        return None


def norm_date(v):
    if not v:
        return None
    for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%m/%d/%y", "%d.%m.%Y", "%B %d, %Y"):
        try:
            return datetime.strptime(str(v).strip(), fmt).date().isoformat()
        except ValueError:
            pass
    return norm_text(v)


def get(o, path):
    for k in path.split("."):
        o = o.get(k) if isinstance(o, dict) else None
    return o


FIELDS = {
    "invoice_number": norm_text,
    "invoice_date": norm_date,
    "due_date": norm_date,
    "total_amount": norm_num,
    "subtotal": norm_num,
    "tax_amount": norm_num,
    "currency": norm_text,
    "vendor.name": norm_text,
    "customer.name": norm_text,
    "purchase_order": norm_text,
}


def score(gold_rows, preds):
    stats = {"valid_json": 0, "schema_valid": 0, "n_items_match": 0}
    hits = {f: [0, 0] for f in FIELDS}  # [correct, counted]
    for g, p in zip(gold_rows, preds):
        gold = json.loads(g["output"]) if "output" in g else json.loads(g["conversations"][-1]["content"])
        raw = p.get("prediction", p.get("output"))
        pred = parse(raw)
        if pred is not None:
            stats["valid_json"] += 1
        # strict: the raw text itself must be one JSON object that passes the schema
        try:
            strict = json.loads(str(raw).strip())
            if VALIDATOR is not None and not list(VALIDATOR.iter_errors(strict)):
                stats["schema_valid"] += 1
        except Exception:
            pass
        pred = pred or {}
        if len(gold.get("line_items") or []) == len(pred.get("line_items") or []):
            stats["n_items_match"] += 1
        for f, fn in FIELDS.items():
            gv = fn(get(gold, f))
            if gv is None:
                continue  # only score fields that have a gold value
            hits[f][1] += 1
            hits[f][0] += fn(get(pred, f)) == gv
    n = len(gold_rows)
    out = {k: v / n for k, v in stats.items()}
    out.update({f: (c / t if t else float("nan")) for f, (c, t) in hits.items()})
    out["MEAN_FIELD_ACC"] = sum(c / t for c, t in hits.values() if t) / sum(1 for _, t in hits.values() if t)
    return out


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


if __name__ == "__main__":
    gold = load(sys.argv[1])
    results = {}
    for path in sys.argv[2:]:
        preds = load(path)
        assert len(preds) == len(gold), f"{path}: {len(preds)} preds vs {len(gold)} test rows"
        results[path] = score(gold, preds)
    names = list(results)
    print(f"{'metric':<18}" + "".join(f"{n[-22:]:>24}" for n in names))
    for m in results[names[0]]:
        print(f"{m:<18}" + "".join(f"{results[n][m]:>24.1%}" for n in names))
