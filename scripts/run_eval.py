"""Run the test set through any OpenAI-compatible endpoint and record predictions + latency.

Works for: your vLLM server (adapter or base model) and frontier APIs with an OpenAI-compatible endpoint.

Examples
  # fine-tuned adapter served by vLLM
  python run_eval.py --test test.jsonl --model invoice --out preds_adapter.jsonl
  # base model, no adapter
  python run_eval.py --test test.jsonl --model unsloth/Llama-3.2-1B-Instruct --out preds_base.jsonl
  # base model + schema-constrained decoding (no fine-tune at all)
  python run_eval.py --test test.jsonl --model unsloth/Llama-3.2-1B-Instruct --guided --out preds_base_guided.jsonl
  # frontier baseline (set OPENAI_API_KEY / base URL for your provider)
  python run_eval.py --test test.jsonl --base-url https://api.openai.com/v1 --model gpt-5.6-luna --reasoning --max-tokens 6000 --out preds_frontier.jsonl

Then: python score_invoices.py test.jsonl preds_base.jsonl preds_adapter.jsonl preds_frontier.jsonl
"""
import argparse, asyncio, json, os, time
import numpy as np
from openai import AsyncOpenAI  # pip install openai


def load_rows(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "conversations" in r:  # cleaned format: system, user, assistant
            msgs = [m for m in r["conversations"] if m["role"] != "assistant"]
        else:  # original format: put the same system prompt in front
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": r["input"]}]
        rows.append(msgs)
    return rows


SYSTEM = ("You extract data from OCR text of invoices. Reply with only a JSON object with exactly these keys, "
          "in this order: vendor, customer, invoice_number, invoice_date, due_date, purchase_order, payment_terms, "
          "payment_method, currency, line_items, subtotal, discount, tax_amount, total_amount, amount_paid, "
          "balance_due, notes. Use null for anything not present. Dates as YYYY-MM-DD, amounts as numbers, "
          "deductions as negative line-item amounts.")


async def one(client, sem, args, i, msgs, schema):
    async with sem:
        if args.reasoning:  # o-series / gpt-5 style: no temperature, max_completion_tokens incl. hidden reasoning
            kw = dict(model=args.model, messages=msgs, max_completion_tokens=args.max_tokens,
                      reasoning_effort=args.reasoning_effort)
        else:
            kw = dict(model=args.model, messages=msgs, temperature=0, max_tokens=args.max_tokens)
        if args.guided:
            kw["response_format"] = {"type": "json_schema",
                                     "json_schema": {"name": "invoice", "schema": schema}}
        t0 = time.perf_counter()
        try:
            r = await client.chat.completions.create(**kw)
            text = r.choices[0].message.content or ""
            usage = r.usage.model_dump() if r.usage else {}
            err = None
        except Exception as e:  # keep going; a failed call counts as a wrong prediction
            text, usage, err = "", {}, str(e)[:300]
        return i, {"prediction": text, "latency_s": time.perf_counter() - t0, "usage": usage, "error": err}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    ap.add_argument("--concurrency", type=int, default=16, help="parallel requests (use 1 for single-user latency)")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--reasoning", action="store_true",
                    help="reasoning model (e.g. gpt-5-mini): drops temperature, uses max_completion_tokens")
    ap.add_argument("--reasoning-effort", default="low", help="passed to the API as-is, e.g. low, medium, high")
    ap.add_argument("--guided", action="store_true", help="schema-constrained decoding via response_format")
    ap.add_argument("--schema", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "invoice_schema.json"))
    args = ap.parse_args()

    rows = load_rows(args.test)
    schema = json.load(open(args.schema)) if args.guided else None
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key, timeout=300)
    sem = asyncio.Semaphore(args.concurrency)

    t0 = time.perf_counter()
    results = await asyncio.gather(*(one(client, sem, args, i, m, schema) for i, m in enumerate(rows)))
    wall = time.perf_counter() - t0

    results = [r for _, r in sorted(results, key=lambda x: x[0])]
    with open(args.out, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    lat = np.array([r["latency_s"] for r in results])
    tin = sum(r["usage"].get("prompt_tokens", 0) for r in results)
    tout = sum(r["usage"].get("completion_tokens", 0) for r in results)
    reas = sum(((r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens") or 0) for r in results)
    empty = sum(1 for r in results if r["error"] is None and not r["prediction"].strip())
    errs = sum(r["error"] is not None for r in results)
    n = len(results)
    print(f"model={args.model} guided={args.guided} n={n} errors={errs} concurrency={args.concurrency}")
    print(f"latency  p50={np.percentile(lat,50):.2f}s  p95={np.percentile(lat,95):.2f}s  max={lat.max():.2f}s")
    print(f"wall={wall:.1f}s  throughput={n/wall*3600:,.0f} invoices/hour")
    print(f"tokens   avg_in={tin/n:,.0f}  avg_out={tout/n:,.0f} (of which reasoning={reas/n:,.0f})  empty_answers={empty}")
    print("cost per 1,000 invoices:")
    print(f"  self-hosted = GPU $/hour / {n/wall*3600:,.0f} * 1000")
    print(f"  API         = ({tin/n:,.0f} * $in_per_token + {tout/n:,.0f} * $out_per_token) * 1000")


if __name__ == "__main__":
    asyncio.run(main())
