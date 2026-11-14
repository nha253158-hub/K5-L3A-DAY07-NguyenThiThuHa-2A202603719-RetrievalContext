import argparse, json, statistics, sys, time
sys.path[:0] = ["starter", "src"]
from agentic import run_agent
from baseline import hit
from retrievers import DenseRetriever, BM25Retriever, HybridRetriever
from openrouter_controller import OpenRouterController

PRICE_IN, PRICE_OUT = 0.05, 0.20   # USD / 1M token, deepseek-v4.1-flash trên OpenRouter

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev")
ap.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="dense")
ap.add_argument("--top-k", type=int, default=3)
ap.add_argument("--budget", type=int, default=2000)
ap.add_argument("--max-searches", type=int, default=3)
ap.add_argument("--use-filter", action="store_true")
ap.add_argument("--tag", required=True)            # tên cấu hình, vd: dense-k3-b2000
ap.add_argument("--context-order", choices=["first", "latest"], default="first")
ap.add_argument("--force-final", action="store_true")
a = ap.parse_args()


qs = [q for q in json.load(open("data/converted/questions.json", encoding="utf-8")) if q["split"] == a.split]
qs += [q for q in json.load(open("data/converted/unanswerable.json", encoding="utf-8")) if q["split"] == a.split]
ret = {"dense": DenseRetriever, "bm25": BM25Retriever,
       "hybrid": lambda: HybridRetriever(DenseRetriever(), BM25Retriever())}[a.method]()
ctl = OpenRouterController(force_final_after=a.max_searches if a.force_final else None)

traces = []
for q in qs:
    f = {q["filter"]["field"]: q["filter"]["value"]} if a.use_filter and q.get("filter") else None
    n0, t0 = len(ctl.usage), time.perf_counter()
    try:
        r = run_agent(q["question"], ctl, ret, filters=f, max_searches=a.max_searches, top_k=a.top_k, budget=a.budget,context_order=a.context_order)
    except Exception as e:                           # lỗi thật -> ghi lại, không thay bằng kết quả giả
        r = {"status": "error", "answer": "", "citations": [], "events": [], "error": repr(e)}
    seen = {c["id"]: c for e in r["events"] if e["event"] == "search" for c in e["results"]}
    searches = [e for e in r["events"] if e["event"] == "search"]
    final_ids = searches[-1]["context_ids"] if searches else []
    u = ctl.usage[n0:]
    traces.append({**r, "query_id": q["id"], "answerable": "gold" in q, "filter": f,
                   "n_searches": len(searches), "seconds": round(time.perf_counter() - t0, 2),
                   "tokens_in": sum(x["in"] for x in u), "tokens_out": sum(x["out"] for x in u),
                   "context_hit": any(hit(seen[i], q["gold"]) for i in final_ids) if "gold" in q else None})
    print(q["id"], r["status"], f"{len(searches)} search", traces[-1]["seconds"], "s", flush=True)

ans = [t for t in traces if t["answerable"]]
tin, tout = sum(t["tokens_in"] for t in traces), sum(t["tokens_out"] for t in traces)
summary = {"tag": a.tag, "split": a.split, "config": vars(a), "model": ctl.model, "n": len(traces),
           "status": {s: sum(t["status"] == s for t in traces) for s in ("answered", "abstain", "error")},
           "tool_use": f"{sum(t['n_searches'] > 0 for t in traces)}/{sum(t['status'] != 'error' for t in traces)}",
           "searches_dist": {k: sum(t["n_searches"] == k for t in traces) for k in range(a.max_searches + 1)},
           "context_evidence": f"{sum(bool(t['context_hit']) for t in ans)}/{len(ans)}",
           "tokens_in": tin, "tokens_out": tout, "cost_usd": round((tin * PRICE_IN + tout * PRICE_OUT) / 1e6, 5),
           "median_seconds": statistics.median(t["seconds"] for t in traces)}
out = f"starter/results/agent-{a.tag}-{a.split}"
import os; os.makedirs(out, exist_ok=True)
json.dump(traces, open(f"{out}/traces.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
json.dump(summary, open(f"{out}/summary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(summary, ensure_ascii=False, indent=1))