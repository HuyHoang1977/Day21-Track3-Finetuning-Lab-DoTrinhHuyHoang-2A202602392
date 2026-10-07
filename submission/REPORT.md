# Lab 21 — Evaluation Report

**Họ tên**: Đỗ Trịnh Huy Hoàng  **MSSV**: 2A202602392  **Ngày**: 07/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 — 14.6 GB khả dụng`
**Định dạng nộp**: Option B — [GitHub](https://github.com/HuyHoang1977/Day21-Track3-Finetuning-Lab) ·
[Adapter trên HuggingFace Hub](https://huggingface.co/HuyHoang1977/lab21-qwen35-triage-vi) · [`LINKS.md`](../LINKS.md)

> Mọi con số dưới đây lấy từ `results/` sau lần chạy full-eval (50 mẫu target, 15 mẫu
> regression). Grader kiểm tra chéo.

---

## 1. Setup

| | |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (corpus mặc định của lab) |
| Train / val | 225 / 25 (seed 42) |
| `max_length` | **1024** — p95 đo được là **98 token** *(results/token_stats.json)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 → **30 optimizer steps** |
| Precision | **fp16** (T4 không có bfloat16 — banner của `labkit.device` cảnh báo trước) |

**Template có giữ khối `<think>` không?** **Có** — *(results/template_check.json)* cho
`"reasoning preserved — safe to train on traces"`. Chuỗi render thực tế từ NB1:

```
<|im_start|>user
2+2?<|im_end|>
<|im_start|>assistant
<think>
buoc 1: kiem tra. buoc 2: tra loi.
</think>

4<|im_end|>
```

**Về `max_length`:** NB1 cảnh báo p95 = 98 token nên gợi ý `max_length=256`, nhưng tier
T4 đặt 1024. Tôi **giữ 1024** và giải thích vì sao: đây là điểm đo được, không phải
điểm đoán — nhưng tier 1024 phải giữ được cả những mẫu có reasoning trace dài hơn.
Với corpus mặc định thì 256 là đủ và tiết kiệm thời gian; tôi giữ 1024 để phép so
sánh với NB3–NB5 chạy ở cấu hình tier, và vì `max_length` không phải biến tôi đang
nghiên cứu. Nếu tối ưu production, tôi sẽ đặt 256.

---

## 2. Mask proof (NB1)

| | |
|---|---|
| `supervised_fraction` | **0.4149** (39/94 token) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

Đoạn thực sự được tính loss (giải mã ngược chuỗi sau `apply_chat_template`):

```
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Đối chiếu với chế độ `everything` — cùng mẫu đó, loss tính lên **toàn bộ 94 token**:

```
<|im_start|>system
Phân loại ticket sau.<|im_end|>
<|im_start|>user
Alo shop, mình đặt balo laptop mã đơn VN411453...<|im_end|>
<|im_start|>assistant
...
```

Tôi **không** chấp nhận `supervised_fraction = 1.0`. Mất trắng điểm 1.1 nếu con số này
≥0.95 — nghĩa là đang tính loss cả trên prompt, tức là bắt model học thuộc lại câu hỏi
thay vì học trả lời. Ở đây 41.5% là con số đúng.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3310.8 |
| (b) base + optimized prompt | **0.765** | 0.791 | 1.000 | 1046.7 |
| (c) LoRA fine-tune | **0.965** | 0.522 | 1.000 | 1392.9 |

**Mốc đóng băng**: `optimized_prompt_sha = 719e74d3b6232053`.

**(b) có thật sự mạnh hơn (a) không?** Có, rất rõ: **0.000 → 0.765**. Prompt naive
("Phân loại ticket sau.") khiến model không sinh được JSON nào — format 0.000 nghĩa là
không parse được mẫu nào. Prompt tối ưu chứa schema 4 khoá, tập giá trị hợp lệ và một ví
dụ mẫu, nên format lên 1.000.

**Tôi có sửa `OPTIMIZED_PROMPT` không?** Không. SHA của prompt (b) được `verify` kiểm
tra và ghi vào `baselines_frozen.json`. Đây là điểm tôi muốn nói rõ: **làm yếu prompt
(b) để fine-tune trông thắng là gian lận, không phải tối ưu.** Giữ nguyên prompt có
nghĩa là mọi kết luận dưới đây đều chịu được sự kiểm chứng.

Đây cũng chính là lý do baseline (b) phải tồn tại: nếu chỉ so fine-tune với (a) = 0.000,
tôi sẽ kết luận "LoRA thắng với chênh lệch 0.965" — một con số trông rất ấn tượng và
hoàn toàn vô nghĩa, vì nó chỉ chứng minh prompt của tôi tệ.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear (12 module) | 16 | 32,464,896 | 1e-4 | 0.6257 | 0.965 | 391.2 | 8.78 |
| `attn_only` | q,v (2 module) | **283** | 32,456,704 | 1e-4 | **0.5381** | **0.970** | 261.0 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | **1e-5** | **1.5702** | **0.000** | 391.8 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | 0.940 | 460.6 | **3.86** |

Cả bốn run dùng **cùng 30 optimizer step** — `verify` đọc `runs.csv` và xác nhận điều
này. Rank của `attn_only` được nâng tới 283 bằng `matched_rank()` để khớp ngân sách tham
số (lệch 0.025%), nên đây là so **vị trí**, không phải so ngân sách.

**Về thứ tự hai cột.** Xếp theo `train_loss`: `attn_only` < `correct` < `qlora` <
`wrong_lr`. Xếp theo `target`: `attn_only` > `correct` > `qlora` > `wrong_lr`. Ở đây
hai cột **cùng thứ tự** — nhưng sự trùng khớp đó có tính chất may mắn, không phải quy
luật. Nếu tôi chỉ nhìn loss tôi sẽ kết luận sai rằng `wrong_lr` "chỉ hơi kém" (1.57
so với 0.54 — hơn gấp đôi), trong khi trên thang đo tác vụ nó **bằng không hoàn toàn**.

**4.1 — `attn_only` cùng số tham số với `correct`, và nó thắng.**

Trên tập target, `attn_only` được 0.970 so với 0.965 của `correct` — thắng, nhưng chỉ
+0.005, tức 0.25 điểm phần trăm trên 50 mẫu. Chênh lệch này nhỏ hơn nhiều so với độ nhiễu
mà một thay đổi seed hay một mẫu eval duy nhất có thể tạo ra, nên tôi không coi đây là
bằng chứng `attn_only` thật sự mạnh hơn. Điều đáng nói là **rank không phải đòn bẩy**:
khi đã cân bằng số tham số, việc chỉ gắn adapter vào `q,v` với r=283 không tệ hơn việc
gắn vào toàn bộ 12 module linear với r=16. Thứ tự theo train loss cũng giống, nên loss
và target cùng chỉ về một hướng ở lần chạy này.

Nhưng có một điều khiến tôi bất ngờ và tôi muốn nói thẳng: `attn_only` dùng **r=283** — một
con số rất lớn, gần bằng toàn bộ hạng của ma trận — lại **train nhanh hơn** `correct`
(261s so với 391s) với cùng số tham số huấn luyện và cùng VRAM đỉnh. Điều đó gợi ý chi
phí thời gian ở đây không nằm ở số lượng tham số mà nằm ở **số module bị chặn backprop**.
Nếu tôi phải chọn một kết luận cho report này thì đây là kết luận đáng giá nhất: ở quy mô
225 mẫu / 30 step, đổi vị trí gắn adapter không cho thấy tác dụng, và đó là giới hạn thật
của phép thí nghiệm này chứ không phải kết luận về LoRA nói chung.

**4.2 — `wrong_lr` chỉ khác đúng một con số.**

Khác biệt duy nhất là LR: 1e-4 → 1e-5, tức xuống đúng thang learning rate của full
fine-tune. Đường loss vì thế tách ngay từ step đầu. Ở `correct`, loss rơi từ 2.163 xuống
0.1387 ngay sau một vòng epoch (token accuracy 0.6559 → 0.9587); ở `wrong_lr`, sau cùng
mốc đó loss chỉ còn 1.606 và token accuracy 0.7176, cuối cùng đứng lại ở **1.119** với
accuracy 0.7909 — model vẫn chưa học được cấu trúc JSON sau 30 step.

Nếu chỉ nhìn loss mà không biết LR, tôi sẽ kết luận sai hai điều. Một: rằng cấu hình
này "chậm học" và chỉ cần thêm step. Không phải — ở 30 step nó vẫn còn đang bám lại ở
ngưỡng học cấu trúc, và LR sai 10× là nguyên nhân, không phải thiếu thời gian. Hai: rằng
`wrong_lr` chỉ kém `correct` một chút. Trên thang đo tác vụ nó là **0.000 target, 0.000
format** — không sinh được mẫu JSON nào parse được. Và latency của nó là 5449 ms, gấp
3.9 lần so với 1393 ms của `correct`, vì model không học được token kết thúc lẫn dấu
`<|im_end|>` nên quá trình decode chạy hết ngân sách token. Tức là LR sai không chỉ làm
giảm điểm, nó làm **hỏng cơ chế dừng** — một lỗi không nhìn thấy được nếu chỉ đọc loss.

**4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì?**

VRAM đỉnh: **3.86 GB so với 8.78 GB — giảm 56%**. Đổi lại: target 0.940 so với 0.965
(−0.025), train loss 0.7058 so với 0.6257, và **chậm hơn 460.6s so với 391.2s (+18%)**.
Latency suy ra 1790 ms so với 1393 ms.

Số đo này **không ủng hộ mạnh** khuyến nghị "không dùng QLoRA cho dòng model này" của
nhà cung cấp. Mất 0.025 target là mất rất nhỏ, và đổi lại tôi có thể train trên GPU 8 GB
thay vì 16 GB. Cái tôi **không** chấp nhận là phần thời gian: QLoRA chậm hơn 18% trong khi
dùng ít VRAM hơn, nghĩa là đường lợi không nằm ở tốc độ. Kết luận thật của tôi: với
tính nghiệm vụ lấy nhãn JSON chuẩn, 4-bit gần như miễn phí — nhưng chỉ đổi lấy bộ nhớ,
không đổi lấy tốc độ, và ở T4 14.6 GB mà tôi đang dùng thì 8.78 GB vẫn vừa, nên
QLoRA không tạo ra lý do để đánh đổi.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: **`FAILED`**
`target Δ = +0.200` · `regression Δ = -0.269` · `valid_trace_rate = 0.00`

Lý do, theo đúng thông điệp của cổng:

> general capability regressed by 0.269 (tolerance 0.020). See deck §6.3 — add 1-5% replay data.

**Diễn giải.** Bản fine-tune thắng mục tiêu rõ ràng: target 0.765 → 0.965, tức +0.200 trên
50 mẫu, và format giữ nguyên 1.000. Nhưng regression nhảy từ 0.791 xuống 0.522 trên 15
câu hỏi kiến thức/chỉ dẫn phổ thông — mất khoảng 4/15 câu. Ngưỡng cổng là 0.020, tức tối
đa được mất 0.3 câu; tôi mất 4 câu, vượt ngưỡng hơn 13 lần. Đây không phải nhiễu: ở lần
chạy smoke với `EVAL_LIMIT=8`, cùng hiện tượng này chỉ lộ ra ở mức −0.125, tức đúng 1 mẫu,
và tôi đã không kết luận gì từ con số đó. Khi chạy full 15 mẫu, nó lộ ra là tổn thất thật.

Nguyên nhân tôi cho là đủ rõ để nêu: `valid_trace_rate = 0.00`. Template giữ khối `<think>`,
nhưng cả 250 câu trả lời trong corpus mặc định đều là JSON trần — không có trace suy luận nào
để học. Mô hình vì thế học được một quy tắc hẹp là "cứ phát ra request thì trả JSON, không
suy nghĩ", và quy tắc đó xung đột trực tiếp với hành vi trả lời câu hỏi kiến thức chung.
Đây là mẫu hình quen thuộc: một tập huấn luyện quá đồng nhất, đủ để thắng một tác vụ và đủ
để xoá một hành vi.

Điều này nói lên điều gì về bài toán của tôi? Nó nói rằng **bài toán không chỉ là "train cho
thắng"**, mà là phải trade-off giữa độ chính xác tác vụ và giữ nguyên năng lực chung — và
phép đo đúng phải đo cả hai. Nếu tôi chỉ nhìn target, tôi sẽ kết luận đây là một
fine-tune thành công với chênh lệch +20 điểm và sẽ ship nó. Verdict FAILED là thứ ngăn
điều đó xảy ra.

**Tôi không nới lỏng cổng.** Ngưỡng 0.020 có vẻ chặt khi nhìn từ phía "target đã tăng
mạnh", nhưng việc nới nó để kết quả thành PASS chính là loại gian lận mà lab này dạy cách
nhận diện. Cách sửa đúng là ở phía dữ liệu (trộn 1–5% replay), không phải ở phía ngưỡng
đo — và tôi chưa thử, vì thời gian của lab đã hết.

---

## 6. Định tính

| # | Ticket (rút gọn) | `ft_score` | Nhận xét |
|---|---|---|---|
| 1 | ốp lưng điện thoại DH936478 — shipper khô... | **1.00** | ✅ FT thắng hoàn toàn (4/4 trường) |
| 2 | ốp lưng điện thoại DH734695 — giá bao nhiêu | **1.00** | ✅ FT thắng — intent `hoi_thong_tin` đúng |
| 3 | ốp lưng điện thoại VN833689 — sai màu, sớm nhất | **1.00** | ✅ FT thắng — phân biệt được `san_pham_loi` |
| 4 | bình giữ nhiệt VN804124 — "chưa thấy tiền" | **0.75** | ❌ **FT thua** — đoán `hoan_tien`, đáng lẽ là `van_chuyen` |
| 5 | nồi chiên không dầu DH249548 — "thiếu phụ kiện" | **0.75** | ❌ **FT thua** — đoán `san_pham_loi`, đáng lẽ là `van_chuyen` |
| 6 | áo khoác gió VN613097 — "bị lỗi, khi nào tiện" | **0.75** | ❌ **FT thua** — đoán `san_pham_loi`, đáng lẽ là `hoi_thong_tin` |

Ba ca thắng đều là ticket có **dấu hiệu rõ ràng** (khô hàng, hỏi giá, sai màu + "sớm nhất"),
và cả ba đều dùng cùng một sản phẩm "ốp lưng điện thoại" — mẫu trong tập train gần như
giống hệt.

Ba ca thua tụng quanh **một mẫu chung**: người viết mô tả triệu chứng bằng ngôn ngữ tự nhiên
nhưng không có từ khoá quyết định. "Chưa thấy tiền" là vấn đề vận chuyển chứ không phải hoàn
tiền; "thiếu phụ kiện" là vận chuyển; "bị lỗi, khi nào tiện" là hỏi thông tin. Model đọc
chúng thành `san_pham_loi` vì đó là lớp lỗi phổ biến nhất trong corpus train — nên tập 225
mẫu không đủ để dạy nó rằng ba cụm từ đó thuộc lớp khác. Cả ba ca thua đều là ca **thiếu
từ khoá nghiệp vụ**, không phải ca model "ngu đi"; đây là giới hạn của dữ liệu, không phải
lỗi của cấu hình train.

*(Ghi chú: NB5 lưu `qualitative.json` gồm 50 dòng đầy đủ, không chỉ 6 dòng này.)*

---

## 7. Kết luận & điều tôi học được

**Kết luận. Tôi không nên deploy bản fine-tune này.** Lý do không phải là nó không đạt
mục tiêu — nó đạt rất tốt, target 0.965 là con số tốt nhất mà tôi đo được. Lý do là cái giá
phải trả: 0.269 regression trên 15 câu hỏi kiến thức chung, tức mất khoảng 4 câu, trong
khi ngưỡng cho phép mất dưới 1 câu. Với một hệ thống CSKH thật, đó là một sự đánh đổi
không chấp nhận được: model phải trả lời được khách hàng hỏi "sản phẩm này dùng được bao
 lâu" hay "chính sách đổi trả thế nào", và một model chỉ biết điền JSON sẽ không trả lời
được. Hơn nữa tôi chưa sửa được nguyên nhân gốc — tôi mới chỉ quan sát được nó. Nếu ở vị
trí quyết định ship, tôi sẽ chặn và yêu cầu chạy lại sau khi trộn replay data.

**Đòn bẩy thật sự trong lab này không phải là LR, cũng không phải là rank.** Nó là **chất
lượng và độ đa dạng của dữ liệu**. Bằng chứng: run `wrong_lr` sập hoàn toàn (target 0.000)
chỉ vì học sai thang LR — nhưng đó là lỗi cấu hình, sửa được bằng cách đổi một con số.
Còn bản `correct` đã cấu hình đúng hoàn toàn vẫn hỏng regression, vì 250 câu trả lời JSON
trần không chứa bất kỳ trace suy luận nào. Một lỗi cấu hình thì không có trong bản deploy;
một tập huấn luyện đồng nhất thì có, và nó nằm trong trọng số. Nếu được thêm 2 giờ, tôi
sẽ dàng phần đó cho replay data chứ không cho sweep siêu tham số.

**Ba điều tôi học được:**

1. **Phải có baseline (b) thì con số +0.200 mới có nghĩa.** Cùng một lần chạy đó, nếu chỉ
   so với prompt naive (a) = 0.000, tôi sẽ khoe "LoRA tăng 96.5 điểm". Chênh lệch thật là
   20 điểm. Phần lớn cái tôi tưởng fine-tune mang lại thực ra đến từ việc tôi viết một
   prompt có schema rõ ràng — và phần đó không cần GPU nào cả.

2. **Train loss là chỉ số thay thế, và tôi đã suýt tin nó.** Ở `wrong_lr`, loss 1.57 nghe
   như "kém một chút" so với 0.54. Trên thang đo tác vụ nó là **0.000 target, 0.000
   format**, và latency cao gấp 3.9 lần vì cơ chế dừng bị hỏng. Nếu tôi báo cáo bằng loss,
   tôi sẽ bỏ sót đúng lỗi nghiêm trọng nhất của cả lab.

3. **Sự phân biệt giữa "nhiễu" và "tín hiệu" đòi hỏi kích thước mẫu đủ lớn.** Ở
   `EVAL_LIMIT=8`, regression tụt 0.125 và tôi đã suýt bác bỏ nó như nhiễu. Chạy đủ 15 mẫu,
   cùng hiện tượng đó là −0.269, tức mất 4 câu. Tôi không sai khi nghi ngờ con số nhỏ; tôi
   sai nếu dừng lại ở đó. Ngưỡng 0.125 trên n=8 là 1 mẫu — không đủ để kết luận, nhưng
   cũng đủ để đặt nghi vấn, và đặt nghi vấn rẻ hơn nhiều so với bỏ qua.

**Nếu có thêm 2 giờ nữa, tôi sẽ thử** (theo thứ tự): (1) trộn 3–5% replay data từ
`eval_regression` vào tập train và chạy lại NB3+NB5 để kiểm chứng giả thuyết nhân quả ở
mục 5 — nếu regression hồi phục, nguyên nhân là độ đồng nhất của dữ liệu chứ không phải LoRA;
(2) thay corpus bằng 250 câu trả lời có reasoning trace thật để `MASK_MODE` không còn là
no-op và chạy đúng đối chiếu §13.5; (3) sweep rank ở vị trí cố định `text-linear` để trả lời
câu hỏi mà lần chạy này chưa trả lời được — liệu `attn_only` thắng 0.005 là thật hay nhiễu.

---

## Phụ lục — thưởng đã làm

- [ ] B1 NB6 merge + hot-swap — *chưa chạy, hết thời gian*
- [ ] B2 dataset miền riêng — *không dùng, giữ corpus mặc định để phép so sánh công bằng với lớp*
- [ ] B3 reasoning-trace collapse — *không làm được: corpus mặc định `valid_trace_rate=0`, `MASK_MODE` là no-op*
- [ ] B4 quét rank có kiểm soát — *chưa làm, đã nêu ở mục "nếu có thêm 2 giờ"*
- [x] **B5 HuggingFace Hub (+2)** — https://huggingface.co/HuyHoang1977/lab21-qwen35-triage-vi
  Adapter `correct` dạng PEFT (không merge), 32.5M tham số, cùng cấu hình với bảng mục 4.
  Model card ghi rõ verdict FAILED và lý do không nên deploy.
- [x] **B1 GitHub + Hub (Option B)** — link đầy đủ trong [`LINKS.md`](../LINKS.md)

---

## Ghi chú về quá trình chạy

Ba lần chạy, mỗi lần dạy một điều:

1. **Lần 1** (`EVAL_LIMIT=8`): chạy trọn 36.8 phút, `verify` báo 2 FAIL. Tôi học được
   chế độ smoke không dùng được để nộp.
2. **Lần 2** (chạy lại NB2+NB5 full): NB5 **bị ngắt giữa chừng** ở run `qlora` (batch
   8/13). `verify` vẫn báo artefact "ok" vì nó chỉ kiểm tra *file tồn tại*, không kiểm tra
   *file có cùng kích thước eval với nhau*. `autopsy.json` và `qualitative.json` còn là số
   của lần smoke (n=8) trong khi `verdict.json` đã là n=50 — hai file mâu thuẫn nhau trong cùng
   một thư mục, và không có kiểm tra nào bắt được. Tôi phải tự so sánh `n` trong từng file
   mới phát hiện.
3. **Lần 3** (chạy lại NB5): 675s, đủ 4 run ở n=50, artefact nhất quán.

Bài học ở lần 2 là lý do tôi nghĩ lab này còn một điều đáng học: **một pipeline có thể báo
"xanh" trong khi nó đang trộn kết quả của hai lần chạy khác nhau.** Thang đo ở đây có
tính chất khách quan, nhưng nó không tự bảo vệ mình khỏi sự nhất quán nội bộ.
