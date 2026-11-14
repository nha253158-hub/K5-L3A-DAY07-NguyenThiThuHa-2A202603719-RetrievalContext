import csv, json, sys
tag, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "dev")
d = f"starter/results/agent-{tag}-{split}"
out = sys.argv[3] if len(sys.argv) > 3 else f"{d}/review_sheet.csv"
T = json.load(open(f"{d}/traces.json", encoding="utf-8"))
Q = {q["id"]: q for f in ("questions.json", "unanswerable.json")
     for q in json.load(open(f"data/converted/{f}", encoding="utf-8"))}

with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.writer(f)
    w.writerow(["query_id", "answerable", "status", "question", "expected_or_reason", "answer", "citations",
                "cited_text", "answer_grade", "citation_support", "abstention_correct", "note"])
    for t in T:
        q = Q[t["query_id"]]
        seen = {c["id"]: c["text"] for e in t["events"] if e["event"] == "search" for c in e["results"]}
        cited = "\n---\n".join(f"[{i}] {seen.get(i, '')[:500]}" for i in t["citations"])
        w.writerow([t["query_id"], t["answerable"], t["status"], q["question"],
                    q.get("expected_answer", q.get("absence_reason", "")), t["answer"],
                    " ".join(t["citations"]), cited, "", "", "", ""])
print("Đã ghi", out)