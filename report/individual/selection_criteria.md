# Tiêu chí chọn cấu hình — chốt trước khi chạy test (14/11/2026)

Chỉ dùng kết quả **dev** (22 câu có đáp án + 2 câu thiếu evidence). Chưa chạy agent trên test tại thời điểm chốt.

## Tiêu chí (theo thứ tự ưu tiên)

1. **Không bịa với câu thiếu evidence:** abstention đúng phải 2/2. Cấu hình làm hỏng điều này bị loại.
2. **Citation hợp lệ:** mọi citation phải có trong context và hỗ trợ câu trả lời (100%).
3. **Grounded answer trung bình /2** cao nhất trên câu có đáp án (từ chối nhầm tính 0).
4. **Từ chối nhầm** (câu có đáp án mà agent từ chối) thấp nhất.
5. Hòa thì chọn cấu hình có **chi phí/thời gian** thấp hơn; chênh < 2 s median và < 0,001 USD/24 câu coi là ngang nhau.

## Kết quả dev dùng để chọn

| Cấu hình | Grounded 0/1/2 | Trung bình /2 | Từ chối nhầm | Abstention | Citation hợp lệ | Median s/câu | USD/24 câu |
|---|---|---|---|---|---|---|---|
| Baseline `dense-k3-b2000` | 8/0/14 | 1,27 | 8/22 | 2/2 | 21/21 | 8,66 | 0,00485 |
| C1 `context_order=latest` | 3/0/19 | 1,73 | 3/22 | 2/2 | 31/31 | 7,18 | 0,00457 |
| C2 `force_final` | 3/2/17 | 1,64 | 3/22 | 2/2 | 30/30 | 7,57 | 0,00486 |
| **C1+C2** | **1/0/21** | **1,91** | **1/22** | **2/2** | **33/33** | 8,79 | 0,00486 |

## Cấu hình đã chốt

**C1+C2:** dense retrieval (Chroma `day06_B`, `intfloat/multilingual-e5-small`), `top_k=3`, context budget 2000 ký tự,
tối đa 3 lượt search, `context_order=latest`, `force_final=True`, model `deepseek/deepseek-v4.1-flash` (temperature 0).

Lý do: đạt tiêu chí 1–2 như các cấu hình khác, tốt nhất ở tiêu chí 3–4; thời gian và chi phí ngang baseline theo tiêu chí 5.
Trên test sẽ chạy **baseline và C1+C2** với cùng câu hỏi, không chỉnh thêm sau khi xem kết quả test.
