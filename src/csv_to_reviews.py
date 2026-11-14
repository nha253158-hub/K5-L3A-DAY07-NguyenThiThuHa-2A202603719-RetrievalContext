import csv, io, json, re, sys
tag, split = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "dev")
d = f"starter/results/agent-{tag}-{split}"

# cases.json: id, split, answerable (dùng chung cho mọi cấu hình)
qs = [q for f in ("questions.json", "unanswerable.json")
      for q in json.load(open(f"data/converted/{f}", encoding="utf-8"))]
json.dump([{"id": q["id"], "split": q["split"], "answerable": "gold" in q} for q in qs],
          open("data/converted/cases.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# Excel có thể lưu UTF-8 hoặc ANSI
sheet = sys.argv[3] if len(sys.argv) > 3 else f"{d}/review_sheet.csv"
data = open(sheet, "rb").read()
for enc in ("utf-8-sig", "cp1258", "cp1252"):
    try:
        raw = data.decode(enc); break
    except UnicodeDecodeError:
        pass
# Chỉ dò dấu phân cách ở dòng tiêu đề; Sniffer đoán sai doublequote khi ô có dấu "
header = raw.split("\n")[0]
delim = max(",;\t", key=header.count)

reviews = []
for r in csv.DictReader(io.StringIO(raw), delimiter=delim):
    g = lambda k: (r.get(k) or "").strip()
    if g("status") == "error":
        continue
    rv = {"query_id": g("query_id"), "note": g("note"),
          "citation_support": [x == "1" for x in re.findall(r"[01]", g("citation_support"))]}
    if g("answerable").lower() == "true":
        if not g("answer_grade"):
            sys.exit(f"{g('query_id')}: thiếu answer_grade")
        rv["answer_grade"] = int(float(g("answer_grade")))
    else:
        rv["abstention_correct"] = g("abstention_correct") == "1"
    reviews.append(rv)
json.dump(reviews, open(f"{d}/reviews.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(len(reviews), "nhãn ->", f"{d}/reviews.json")