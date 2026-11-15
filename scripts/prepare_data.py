"""Clean raw {"input","output"} invoice data and convert it to chat format for Unsloth.

Usage: python prepare_data.py raw_train.jsonl train.jsonl raw_val.jsonl validation.jsonl raw_test.jsonl test.jsonl

What it does (see README "Labelling and cleaning"):
  - one fixed top-level schema: the same 17 keys in the same order, null when missing, [] for no line items
  - drops extraction_confidence and rare top-level keys
  - flips deduction line items (LESS / DISCOUNT / CREDIT ...) to negative ONLY when that makes the
    line items add up to the total; unclear cases are left unchanged and counted
  - wraps each example as system / user / assistant messages
"""
import json, sys, collections
KEYS = ["vendor","customer","invoice_number","invoice_date","due_date","purchase_order",
        "payment_terms","payment_method","currency","line_items","subtotal","discount",
        "tax_amount","total_amount","amount_paid","balance_due","notes"]
DEDUCT = ("LESS","DISCOUNT","CREDIT","DEDUCT","REBATE")
SYSTEM = ("You extract data from OCR text of invoices. Reply with only a JSON object with exactly these keys, "
          "in this order: " + ", ".join(KEYS) + ". Use null for anything not present. "
          "Dates as YYYY-MM-DD, amounts as numbers, deductions as negative line-item amounts.")
def num(x): return isinstance(x,(int,float)) and not isinstance(x,bool)
def fix_signs(o, st):
    items = o.get("line_items") or []; t = o.get("total_amount")
    if not num(t) or not all(num(i.get("amount")) for i in items): return
    cand = [i for i in items if i["amount"]>0 and any(w in str(i.get("description","")).upper() for w in DEDUCT)]
    if not cand: return
    before = sum(i["amount"] for i in items); after = before - 2*sum(i["amount"] for i in cand)
    targets = [t] + [o[k] for k in ("subtotal",) if num(o.get(k))]
    if any(abs(after-x)<0.01 for x in targets) and not any(abs(before-x)<0.01 for x in targets):
        for i in cand: i["amount"] = -i["amount"]
        st["signs fixed"] += len(cand)
    else: st["deduction left as is (unclear)"] += len(cand)
def clean(row, st):
    o = json.loads(row["output"])
    dropped = [k for k in o if k not in KEYS]; st["rows w/ rare keys dropped"] += bool(dropped)
    fix_signs(o, st)
    out = {k: (o.get(k) if o.get(k) not in ("", []) else None) for k in KEYS}
    if out["line_items"] is None: out["line_items"] = []
    return {"conversations":[{"role":"system","content":SYSTEM},
                             {"role":"user","content":row["input"]},
                             {"role":"assistant","content":json.dumps(out, ensure_ascii=False)}]}
for src, dst in zip(sys.argv[1::2], sys.argv[2::2]):
    st = collections.Counter(); n = 0; mx = 0
    with open(dst,"w",encoding="utf-8") as f:
        for l in open(src, encoding="utf-8"):
            if not l.strip(): continue
            r = clean(json.loads(l), st); n += 1
            mx = max(mx, sum(len(m["content"]) for m in r["conversations"]))
            f.write(json.dumps(r, ensure_ascii=False)+"\n")
    print(dst, n, "rows |", dict(st), "| longest example ~", mx//3, "tokens (rough)")
