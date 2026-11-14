import argparse, json, sys, time
sys.path.insert(0, "starter"); sys.path.insert(0, "src")
from baseline import hit
from retrievers import DenseRetriever, BM25Retriever, HybridRetriever   # ← (1)

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev")
ap.add_argument("--use-filter", action="store_true")
ap.add_argument("--method", choices=["dense", "bm25", "hybrid"], default="dense")   # ← (2)
args = ap.parse_args()

qs = [q for q in json.load(open("data/converted/questions.json", encoding="utf-8")) if q["split"] == args.split]
ret = {"dense": DenseRetriever, "bm25": BM25Retriever,                  # ← (3)
       "hybrid": lambda: HybridRetriever(DenseRetriever(), BM25Retriever())}[args.method]()
traces, t0 = [], time.perf_counter()
for q in qs:
    f = {q["filter"]["field"]: q["filter"]["value"]} if args.use_filter and q.get("filter") else None
    res = ret.search(q["question"], filters=f, top_k=3)
    traces.append({"query_id": q["id"], "filter": f,
                   "top3": [(r["id"], round(r["score"], 3)) for r in res],
                   "hit1": hit(res[0], q["gold"]), "hit3": any(hit(r, q["gold"]) for r in res)})
ms = (time.perf_counter() - t0) / len(qs) * 1000
print(f"[{args.method}] hit1 {sum(t['hit1'] for t in traces)}/{len(qs)}  hit3 {sum(t['hit3'] for t in traces)}/{len(qs)}  {ms:.0f} ms/q")
json.dump(traces, open(f"starter/results/{args.method}-{args.split}{'-filter' if args.use_filter else ''}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)   # ← (4)