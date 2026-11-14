# Code Day 7

| File | Công dụng |
|---|---|
| `prepare_data.py` | Chuyển dữ liệu Day 6 (`data/day6/`) sang `data/converted/` (chunk, câu hỏi; test Day 6 chuyển thành dev, thêm `data/new_test.json`). |
| `retrievers.py` | 3 phương pháp: `DenseRetriever` (Chroma Day 6 + e5-small), `BM25Retriever`, `HybridRetriever` (RRF k=60, pool 20). |
| `eval_retrieval.py` | Hit@1, Hit@3 từng phương pháp, có/không filter; lưu top-3, score và ms/query. |
| `summarize_retrieval.py` | Bảng Hit@1, Evidence Hit@3, Context Evidence Rate (dựng lại context bằng `pack_context`). |
| `openrouter_controller.py` | Controller gọi model thật qua OpenRouter với 2 tool `search`, `answer`; retry 429/5xx; `force_final_after`. |
| `run_agent_bench.py` | Chạy agent trên một split, lưu trace, token, chi phí, thời gian, context hit. |
| `make_review_sheet.py` / `csv_to_reviews.py` | Tạo sheet chấm tay → `reviews.json` cho `starter/agent_metrics.py`. |

## Cài đặt

```bash
pip install -r requirements.txt
# .env: OPENROUTER_API_KEY=..., OPENROUTER_MODEL=deepseek/deepseek-v4.1-flash, CHROMA_PATH=<đường dẫn chroma_db Day 6>
python src/prepare_data.py
python -m unittest discover -s tests -v
```

Chroma DB Day 6 không nằm trong repo (mặc định `../day6/results/chroma_db`, đặt lại bằng `CHROMA_PATH`). Tái tạo bằng pipeline Day 6 với chunker B và `intfloat/multilingual-e5-small` (384 chiều, prefix `passage: `/`query: `).

## Chạy dev

```bash
python starter/baseline.py --chunks data/converted/chunks.json --questions data/converted/questions.json --split dev --output starter/results/dev-tfidf
python src/eval_retrieval.py --split dev --method dense   # bm25 | hybrid, thêm --use-filter
python src/summarize_retrieval.py --split dev --budget 2000
python src/run_agent_bench.py --split dev --tag dense-k3-b2000                                  # baseline
python src/run_agent_bench.py --split dev --context-order latest --tag C1-latest                # cải thiện 1
python src/run_agent_bench.py --split dev --force-final --tag C2-forcefinal                     # cải thiện 2
python src/run_agent_bench.py --split dev --context-order latest --force-final --tag C1C2       # kết hợp (đã chọn)
python src/make_review_sheet.py <tag> dev    # chấm tay, rồi:
python src/csv_to_reviews.py <tag> dev
python starter/agent_metrics.py --cases data/converted/cases.json --traces starter/results/agent-<tag>-dev/traces.json --reviews starter/results/agent-<tag>-dev/reviews.json --split dev --output starter/results/agent-<tag>-dev/agent_metrics.json
```

Khi chấm trong Excel, lưu bằng **CSV UTF-8** (không phải "CSV (Comma delimited)") để giữ dấu tiếng Việt.

## Chạy test

Sau khi chốt cấu hình ([selection_criteria.md](../report/individual/selection_criteria.md)): `powershell -ExecutionPolicy Bypass -File run_test.ps1`.
