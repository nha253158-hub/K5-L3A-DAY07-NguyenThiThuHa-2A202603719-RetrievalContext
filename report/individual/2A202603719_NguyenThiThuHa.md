# Báo cáo Day 7 — Nguyễn Thị Thu Hà (2A202603719)

> Trình tự: chạy và chấm dev → chốt tiêu chí và cấu hình ([selection_criteria.md](selection_criteria.md)) → chạy test một lần bằng `run_test.ps1` (baseline và cấu hình đã chọn), không chỉnh gì sau khi xem test.

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:** scikit-learn 1.9.1 User Guide (`doc/modules/*.rst`, GitHub scikit-learn, giấy phép BSD-3-Clause), 40 tài liệu / 288 trang (unit), thuộc 5 mục User Guide. Chunker B của Day 6: **1828 chunk**, median 494 ký tự, tối đa 999 ký tự. Chuyển đổi: `src/prepare_data.py` → `data/converted/chunks.json` (id, text, source_id, source, page = unit_id, metadata `guide_section`, `section_heading`).
- **Vector store, embedding model, phiên bản và số chiều:** ChromaDB persistent, collection `day06_B` (index Day 6, không index lại); `intfloat/multilingual-e5-small`, 384 chiều, chuẩn hóa L2, prefix `query: ` cho câu hỏi (tương thích `passage: ` lúc index); chạy CPU. Score dense = 1 − distance trả về từ Chroma.
- **Agent model, cách chạy và phân chia dev/test:** `deepseek/deepseek-v4.1-flash` qua OpenRouter, temperature 0, `max_tokens=1024`, tắt reasoning, tool calling bắt buộc (`search`, `answer`). Dev = D01–D16 (dev Day 6) + T01–T06 (test Day 6, chuyển sang dev vì T02 đã bị xem khi sửa chunker ở Day 6) = **22 câu có đáp án + 2 câu thiếu evidence (D17, D18)**. Test mới = **6 câu có đáp án (N01–N06) + 2 câu thiếu evidence (N07, N08)**, lấy từ các tài liệu không có trong dev (sgd, classification_threshold, semi_supervised, neural_networks_supervised, mixture, compose). Tổng 32 câu.
- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:** câu có đáp án ghi `gold = {source_id, page, evidence}`; evidence trích nguyên văn tiếng Anh. Hit = chunk đúng `page` và chứa trọn evidence sau khi gộp khoảng trắng; evidence bị tách qua 2 chunk tính là không hit. D15 có 2 evidence thay thế (tree:008 hoặc ensemble:001). Câu thiếu evidence được kiểm bằng grep toàn corpus (vd. "MNIST", "CUDA", "device=" xuất hiện 0 lần; N07 không có "profit/gain/revenue"; N08 không có "size/limit/disk") và ghi `absence_reason`. Câu hỏi và evidence do LLM soạn ở Day 6 từ việc đọc từng trang; script kiểm tra evidence tồn tại nguyên văn trong nguồn. Filter (`guide_section`) chỉ có ở câu loại `filter`, coi như người dùng cung cấp trước, giá trị lấy từ toctree User Guide, không suy từ gold.

## Retrieval và thử nghiệm

### So sánh phương pháp retrieval (dev, 22 câu có đáp án, query gốc, top-3)

Lệnh: `src/eval_retrieval.py`, `starter/baseline.py`, bảng tổng hợp `src/summarize_retrieval.py --budget 2000` (context dựng lại từ top-3 bằng đúng `pack_context` của agent, budget 2000 ký tự gồm cả header nguồn).

| Phương pháp | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|
| TF-IDF (starter, lexical) | 8/22 (36%) | 12/22 (55%) | 11/22 (50%) | 1,5 ms/q, 0 USD |
| **Dense** (Chroma Day 6, e5-small) | **13/22 (59%)** | 16/22 (73%) | **16/22 (73%)** | ~40 ms/q CPU*, 0 USD |
| Dense + filter `guide_section` | 13/22 (59%) | 17/22 (77%) | 16/22 (73%) | ~54 ms/q*, 0 USD |
| BM25 (`rank_bm25`, tách từ `\w+`) | 7/22 (32%) | 15/22 (68%) | 10/22 (45%) | ~16 ms/q*, 0 USD |
| Hybrid RRF (dense + BM25, k=60, pool 20) | 9/22 (41%) | 17/22 (77%) | 15/22 (68%) | ~56 ms/q*, 0 USD |
| Hybrid RRF + filter | 10/22 (45%) | **18/22 (82%)** | 16/22 (73%) | ~73 ms/q*, 0 USD |

Nhận xét: dense có Hit@1 cao nhất; hybrid kéo thêm 1–2 câu vào top-3 nhờ BM25 bắt tên class/tham số chính xác, nhưng đẩy evidence xuống hạng 2–3 nên Hit@1 giảm và Context Evidence Rate không tăng (chunk dài bị cắt khỏi budget). Filter chỉ áp dụng cho vài câu loại `filter` nên tác động nhỏ. Agent dùng dense vì Hit@1 cao nhất và context rate ngang hybrid+filter với 1 retriever đơn giản hơn. \* Thời gian median đo ở lần chạy test (cùng retriever, cùng máy, không tính thời gian nạp model); lần chạy dev bản cũ chỉ in ra màn hình nên không lưu.

Retrieval trên **test** (6 câu, chạy sau khi chốt, chỉ để báo cáo): TF-IDF 1/6 · 3/6 · 2/6; dense 2/6 · 3/6 · 3/6; dense+filter 2/6 · 3/6 · 3/6; BM25 0/6 · 4/6 · 3/6; hybrid 2/6 · 3/6 · 3/6; hybrid+filter 2/6 · 3/6 · 3/6 (Hit@1 · Hit@3 · Context). Test khó hơn dev với query gốc tiếng Việt: evidence thường ở hạng 2–3 hoặc ngoài top-3; dense, hybrid và filter ngang nhau (2/6 · 3/6 · 3/6); BM25 có Hit@3 cao nhất (4/6) nhưng Hit@1 bằng 0, giống xu hướng ở dev (BM25 kéo được evidence vào top-3 nhưng xếp hạng kém).

### Thử nghiệm agent (dense, top_k=3, budget 2000 ký tự, tối đa 3 search)

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline `dense-k3-b2000` | dense, k=3, không filter, không rerank, 2000 ký tự, ≤3 search, context giữ chunk lượt đầu | — | dev | 13/22 | 16/22 | 17/22 (77%) | median 8,66 s/câu; 63,5k in + 8,4k out token; 0,00485 USD/24 câu |
| Phương pháp 2: BM25 | như trên, chỉ đo retrieval | lexical bắt tên tham số | dev | 7/22 | 15/22 | 10/22 | 0 USD |
| Phương pháp 3: Hybrid RRF | như trên, chỉ đo retrieval | gộp dense + BM25 | dev | 9/22 | 17/22 | 15/22 | 0 USD |
| Cải thiện 1 `C1-latest` | baseline + `context_order=latest` | Evidence tìm được ở lượt 2–3 bị loại vì budget đã đầy bởi chunk lượt 1 → ưu tiên kết quả mới nhất | dev | 13/22 | 16/22 | **21/22 (95%)** | 7,18 s; 0,00457 USD |
| Cải thiện 2 `C2-forcefinal` | baseline + `force_final` (sau 3 search, bắt buộc gọi `answer`) | Model đã có evidence nhưng vẫn search tới giới hạn → bị ép từ chối | dev | 13/22 | 16/22 | 16/22 (73%) | 7,57 s; 0,00486 USD |
| **Đã chọn `C1C2`** | baseline + latest + force_final | Kết hợp 2 sửa lỗi độc lập | dev | 13/22 | 16/22 | **21/22 (95%)** | 8,79 s; 0,00486 USD |
| Cấu hình đã chọn | C1C2 | — | test | 2/6 | 3/6 | **5/6 (83%)** | median 9,06 s; 24,0k in + 2,8k out; 0,00177 USD/8 câu |
| Baseline | dense-k3-b2000 | — | test | 2/6 | 3/6 | 4/6 (67%) | median 8,70 s; 21,8k in + 2,4k out; 0,00157 USD/8 câu |

Hit@1/Hit@3 là của lượt search với query gốc (giống nhau giữa các cấu hình agent vì cùng retriever); Context Evidence Rate là context cuối gửi model, gom từ nhiều lượt search do model tự viết query, nên có thể cao hơn Hit@3. Chi phí theo giá OpenRouter 0,05 USD/1M token vào, 0,20 USD/1M token ra; embedding chạy local.

### Hai lỗi dev, nguyên nhân và cách thử

1. **Evidence bị đẩy khỏi context (D13, D14, D06 ở baseline).** D13: lượt search 3 trả `clustering:001:002` (chứa evidence "convex and isotropic") ở hạng 2, nhưng context vẫn là 3 chunk của lượt 1 vì `pack_context` duyệt theo thứ tự cũ và budget 2000 đã đầy → model không thấy evidence, hết lượt, từ chối. Giả thuyết: đặt kết quả lượt mới nhất lên đầu. **C1** tăng Context Evidence Rate 17/22 → 21/22 và từ chối nhầm 8/22 → 3/22; D13 trả lời đúng với `[B:clustering:001:002]`.
2. **Hết lượt search dù context đã đủ (D02, D05, T01 ở baseline).** D05: `svm:003:000` (có evidence class_weight → C * value) nằm trong context từ lượt 1, nhưng model vẫn search thêm 2 lần; lượt thứ 4 model lại gọi `search` nên vòng lặp trả `search_limit` → từ chối. Giả thuyết: sau 3 search bắt buộc model gọi `answer` (`tool_choice=answer`) để nó tự quyết trả lời hay từ chối. **C2** giảm từ chối nhầm 8/22 → 3/22, nhưng dùng một mình thì context vẫn theo thứ tự cũ (16/22) và có 2 câu chỉ đúng một phần (D12, D13). Kết hợp **C1C2**: từ chối nhầm còn 1/22, grounded 1,91/2.

Lỗi còn lại của C1C2 — **D15** ("estimator nào hỗ trợ sẵn missing values?"): model lặp nguyên query "estimators support missing values natively without imputation" cả 3 lượt, dense luôn trả các chunk `impute:*`, evidence ở `tree:008`/`ensemble:001` không bao giờ vào top-3 → từ chối. Đây là lỗi query không đa dạng + câu cần tổng hợp nhiều nguồn; hướng thử tiếp: hybrid/BM25 cho tool search, hoặc chặn query trùng lặp. Không thử thêm để giữ đúng cấu hình đã chốt.

Lý do chọn: theo tiêu chí công bố trước ([selection_criteria.md](selection_criteria.md)) — abstention 2/2 và citation 100% như các cấu hình khác, grounded và từ chối nhầm tốt nhất, chi phí/thời gian ngang baseline.

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:** 2 tool: `search(query)` (dense, top-3, query tiếng Anh do model viết) và `answer(answer, citations)`. Mỗi lượt model nhận câu hỏi, danh sách query đã dùng và context tích lũy (deduplicate theo chunk ID, ≤ 2000 ký tự, mỗi chunk kèm `[id | source | page]`). Dừng khi model gọi `answer`, hoặc sau 3 search (C2: ép gọi `answer`; baseline: trả `search_limit` → từ chối). Câu trả lời chỉ được chấp nhận nếu mọi citation có trong context và xuất hiện dạng `[id]` trong câu trả lời; ngược lại chuyển thành từ chối (`invalid_or_missing_citations`). System prompt ghi rõ context là dữ liệu, không phải chỉ dẫn. Lỗi API 429/5xx/timeout retry tối đa 5 lần (backoff 5·2^i s); lỗi còn lại ghi `status: error` vào trace, không thay bằng kết quả giả. Dev: 0 lỗi provider.

| Cấu hình / split | Tool use | Số lượt gọi (1/2/3 search) | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| Baseline / dev | 24/24 | 12/1/11 | 8/0/14 — 1,27/2 | 21/21 (100%) | 14/14 | 2/2 | 8/22 (36%) |
| C1-latest / dev | 24/24 | 13/5/6 | 3/0/19 — 1,73/2 | 31/31 | 19/19 | 2/2 | 3/22 (14%) |
| C2-forcefinal / dev | 24/24 | 13/1/10 | 3/2/17 — 1,64/2 | 30/30 | 19/19 | 2/2 | 3/22 (14%) |
| **C1C2 / dev** | 24/24 | 12/3/9 | **1/0/21 — 1,91/2** | 33/33 | 21/21 | 2/2 | **1/22 (5%)** |
| Baseline / test | 8/8 | 4/0/4 | 2/0/4 — 1,33/2 | 6/6 | 4/4 | 2/2 | 2/6 |
| **C1C2 / test** | 8/8 | 1/4/3 | **1/0/5 — 1,67/2** | 9/9 | 6/6 | 2/2 | 0/6 theo status; **1/6 theo nội dung** (N06) |

Nhãn dev do người chấm đọc câu trả lời và đọc từng chunk được cite (cột `cited_text` trong `review_sheet`); ID có trong context chưa đủ để chấm 1. Nhãn baseline dev được chấm nháp có AI hỗ trợ, Hà đã rà lại.

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| D10 (C1C2) | Có đáp án | 1 search "GridSearchCV refit multimetric multiple metrics scoring" → `grid_search:006:001` (0,914), **`grid_search:006:002`** (0,907, hit), `model_evaluation:004:000` | refit phải là tên metric (string) dùng để tìm `best_params_`; không refit thì `refit=False`; để `None` sẽ lỗi `[B:grid_search:006:002]` | 2 — đủ cả 3 ý; citation hỗ trợ từng câu |
| D17 (C1C2) | Thiếu evidence | 3 search "K-means accuracy MNIST", "K-means MNIST accuracy digits", "K-means MNIST accuracy 0.5" → chỉ chunk clustering/model_evaluation, không có MNIST | "Không đủ bằng chứng để trả lời." (citations rỗng; bị ép `answer` sau 3 search) | Abstention đúng — không bịa con số |
| D13 (baseline → C1) | Có đáp án, lỗi | Baseline: lượt 3 có `clustering:001:002` (hit) nhưng context giữ 3 chunk lượt 1 → `search_limit` | Baseline: từ chối. C1: "convex and isotropic… phản ứng kém với cụm kéo dài/manifold bất thường" `[B:clustering:001:002]` | Baseline 0 (từ chối nhầm); C1 2 |
| D15 (C1C2) | Có đáp án, lỗi còn lại | 3 lần cùng query → `impute:000:000`, `impute:000:002`, `impute:004:002`; không có tree:008/ensemble:001 | Từ chối | 0 — evidence chưa bao giờ được truy xuất |
| N01 (C1C2, test) | Có đáp án | 2 search; "SGD convergence number of training samples max_iter" → **`sgd:004:003`** hạng 1 (0,936) | ~10^6 mẫu; max_iter = np.ceil(10**6 / n) `[B:sgd:004:003]` | 2 — khớp nguyên văn evidence |
| N04 (baseline → C1C2, test) | Có đáp án | Lượt 2 trả `neural_networks_supervised:004:002` (hit, hạng 2–3). Baseline: context vẫn là chunk lượt 1 → `search_limit`. C1C2: chunk mới lên đầu context | Baseline: từ chối. C1C2: L-BFGS cho dữ liệu nhỏ, Adam cho dữ liệu tương đối lớn `[B:neural_networks_supervised:004:002]` | Baseline 0, C1C2 2 — lỗi dev số 1 lặp lại trên test và C1 sửa được |
| N06 (C1C2, test) | Có đáp án, lỗi | 3 search về "Pipeline memory caching final step…" → luôn `compose:002:003/001/002`; evidence ở `compose:002:000` không vào top-3 | "Không đủ bằng chứng…" nhưng kèm `[B:compose:002:001]`, `[B:compose:002:003]` → code ghi `status=answered` | 0 (từ chối); 2 citation đúng cho ý phụ. Lỗi retrieval + lỗi đo status |
| N07 (C1C2, test) | Thiếu evidence | 3 search "insurance company profit…" → `classification_threshold:001:001` (có ví dụ bảo hiểm, không có số tiền) | Bị ép `answer` → "Không đủ bằng chứng", citations rỗng | Abstention đúng — không bịa số tiền |
| N08 (C1C2, test) | Thiếu evidence | 3 search, có "joblib.Memory cache size limit bytes" → `compose:002:001`, chunk svm/neighbors không liên quan | Từ chối, citations rỗng | Abstention đúng |

Chấm context test (đọc context cuối, thang 0/1/2): baseline N01–N06 = 2, 2, 2, 0, 2, 0 → **4 câu mức 2, 0 câu mức 1, 2 câu mức 0**; C1C2 = 2, 2, 2, 2, 2, 0 → **5 / 0 / 1**. Câu mức 0: N04 baseline (evidence bị đẩy khỏi context), N06 cả hai (không truy xuất được). Nhãn test được chấm nháp có AI hỗ trợ, Hà đã rà lại.

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:** `starter/results/agent-<tag>-<split>/` gồm `traces.json` (mọi tool call, query, filter, kết quả kèm score/rank, context, context_ids, câu trả lời, citation, token, giây), `summary.json`, `review_sheet*.csv` (chấm tay), `reviews.json`, `agent_metrics.json`. Retrieval: `starter/results/{dense,bm25,hybrid}-dev[-filter].json`, `dev-tfidf/`, `retrieval_summary-dev.json`. Lệnh: xem [src/README.md](../../src/README.md); test: `run_test.ps1`. Tổng chi phí 4 lần chạy agent dev ≈ 0,019 USD. `starter/results/` bị `.gitignore`, nộp bằng `git add -f starter/results`.
- **Kết quả test (chạy 1 lần sau khi chốt):** C1C2 tốt hơn baseline trên test như ở dev — grounded 1,33 → 1,67/2, Context Evidence Rate 4/6 → 5/6, abstention vẫn 2/2, citation 100%; chi phí +13% (0,00157 → 0,00177 USD/8 câu) và median +0,36 s vì nhiều câu dùng thêm 1 lượt search. Chênh lệch chỉ dựa trên 1 câu (N04), nên chưa đủ để kết luận có ý nghĩa thống kê. Lỗi mới phát hiện trên test: agent có thể "từ chối mềm" (viết "Không đủ bằng chứng" nhưng vẫn kèm citation), bị `run_agent` đếm là `answered` — nên metric từ chối nhầm theo status (0/6) thấp hơn thực tế (1/6); cần đọc nội dung, không chỉ dựa vào status.
- **Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn (dev):** C1C2 tăng grounded 1,27 → 1,91/2 và giảm từ chối nhầm 36% → 5% so với baseline mà không làm hỏng abstention (2/2) hay citation (100%); chi phí gần như không đổi (+0,2%), median chậm hơn ~0,1 s vì nhiều câu dùng đủ 3 search hơn. Giới hạn: (1) dev nhỏ (22 + 2 câu), test chỉ 6 + 2 câu nên chênh 1 câu = 17 điểm %; (1b) phát hiện từ chối dựa vào citation rỗng, bỏ sót "từ chối mềm" có citation (N06); (2) chỉ chạy agent với dense, chưa chạy agent với hybrid dù hybrid+filter có Hit@3 cao nhất; (3) câu hỏi do LLM soạn, có thể thiên về cách diễn đạt gần tài liệu; (4) Hit định nghĩa theo khớp nguyên văn trong một chunk nên câu có evidence bị cắt ngang chunk luôn tính miss; (5) chỉ 4 câu thiếu evidence; (6) chấm do một người, chưa có người chấm thứ hai.

## Thử thách nâng cao (nếu thực hiện)

Không thực hiện.
