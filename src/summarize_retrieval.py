"""Tổng hợp Hit@1, Evidence Hit@3, Context Evidence Rate cho các file kết quả retrieval.

Context được dựng lại từ top-3 đã lưu bằng đúng hàm pack_context của agent (cùng budget),
nên không cần chạy lại vector store. Dùng: python src/summarize_retrieval.py --split dev --budget 2000
"""
import argparse, json, os, sys
sys.path.insert(0, "starter")
from baseline import hit, pack_context

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="dev")
ap.add_argument("--budget", type=int, default=2000)
a = ap.parse_args()

C = {c["id"]: c for c in json.load(open("data/converted/chunks.json", encoding="utf-8"))}
Q = {q["id"]: q for q in json.load(open("data/converted/questions.json", encoding="utf-8")) if q["split"] == a.split}
R = "starter/results"
files = {"tfidf": f"{R}/{a.split}-tfidf/traces.json"}
for m in ("dense", "bm25", "hybrid"):
    files[m] = f"{R}/{m}-{a.split}.json"
    files[m + "+filter"] = f"{R}/{m}-{a.split}-filter.json"

rows = []
for name, path in files.items():
    if not os.path.exists(path):
        continue
    T = json.load(open(path, encoding="utf-8"))
    h1 = h3 = ctx = 0
    ms = [t["elapsed_ms"] for t in T if "elapsed_ms" in t] or [t["ms"] for t in T if "ms" in t]
    for t in T:
        ids = [r["id"] for r in t["results"]][:3] if "results" in t else [i for i, _ in t["top3"]]
        res = [C[i] for i in ids]
        gold = Q[t["query_id"]]["gold"]
        _, cids = pack_context(res, budget=a.budget)
        h1 += bool(res) and hit(res[0], gold)
        h3 += any(hit(r, gold) for r in res)
        ctx += any(hit(C[i], gold) for i in cids)
    n = len(T)
    rows.append({"method": name, "n": n, "hit1": h1, "hit3": h3, "context": ctx,
                 "median_ms": round(sorted(ms)[len(ms) // 2], 1) if ms else None})
print(f"| Phương pháp | Hit@1 | Evidence Hit@3 | Context Evidence Rate (top-3, {a.budget} ký tự) | ms/query |")
print("|---|---|---|---|---|")
for r in rows:
    f = lambda k: f"{r[k]}/{r['n']} ({r[k] / r['n']:.0%})"
    print(f"| {r['method']} | {f('hit1')} | {f('hit3')} | {f('context')} | {r['median_ms'] if r['median_ms'] is not None else 'n/a'} |")
json.dump(rows, open(f"{R}/retrieval_summary-{a.split}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
