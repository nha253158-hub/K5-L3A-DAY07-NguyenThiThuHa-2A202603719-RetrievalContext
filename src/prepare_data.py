import json
from pathlib import Path

OUT = Path("data/converted"); OUT.mkdir(parents=True, exist_ok=True)

units = {u["unit_id"]: u for u in map(json.loads, open("data/day6/units.jsonl", encoding="utf-8"))}
chunks = [json.loads(l) for l in open("data/day6/chunks_B.jsonl", encoding="utf-8")]
out_c = [{"id": c["chunk_id"], "text": c["text"], "source_id": c["doc_id"],
          "source": units[c["unit_id"]]["source"], "page": c["unit_id"],
          "metadata": {"guide_section": units[c["unit_id"]]["guide_section"],
                       "section_heading": c["section_heading"]}} for c in chunks]

qs = json.load(open("data/day6/questions.json", encoding="utf-8"))["questions"]
out_q = [{**{k: q[k] for k in ("id", "split", "question", "type", "filter", "expected_answer")},
          "gold": [{"source_id": g["gold_source_id"], "page": g["gold_page"],
                    "evidence": g["evidence_quote"]} for g in q["gold"]]} for q in qs]

for name, data in (("chunks.json", out_c), ("questions.json", out_q)):
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(out_c), "chunks,", len(out_q), "questions")