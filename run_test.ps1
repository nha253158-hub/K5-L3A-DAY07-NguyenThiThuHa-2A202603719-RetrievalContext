# Run TEST evaluation after the config is frozen (report/individual/selection_criteria.md).
# Run from repo root with the Python env activated and OPENROUTER_API_KEY available.
$ErrorActionPreference = "Stop"

# 1) Retrieval on test (no API cost): TF-IDF, dense, BM25, hybrid, with/without filter
python starter/baseline.py --chunks data/converted/chunks.json --questions data/converted/questions.json --split test --output starter/results/test-tfidf
foreach ($m in "dense", "bm25", "hybrid") {
    python src/eval_retrieval.py --split test --method $m
    python src/eval_retrieval.py --split test --method $m --use-filter
}
python src/summarize_retrieval.py --split test --budget 2000

# 2) Agent on test: baseline and the chosen config (C1+C2)
python src/run_agent_bench.py --split test --method dense --top-k 3 --budget 2000 --max-searches 3 --tag dense-k3-b2000
python src/run_agent_bench.py --split test --method dense --top-k 3 --budget 2000 --max-searches 3 --context-order latest --force-final --tag C1C2

# 3) Review sheets for manual grading
python src/make_review_sheet.py dense-k3-b2000 test
python src/make_review_sheet.py C1C2 test
Write-Host 'Grade starter/results/agent-*-test/review_sheet.csv (save as CSV UTF-8), then for each TAG in dense-k3-b2000, C1C2:'
Write-Host '  python src/csv_to_reviews.py TAG test'
Write-Host '  python starter/agent_metrics.py --cases data/converted/cases.json --traces starter/results/agent-TAG-test/traces.json --reviews starter/results/agent-TAG-test/reviews.json --split test --output starter/results/agent-TAG-test/agent_metrics.json'
