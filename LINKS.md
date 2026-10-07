# Links — Lab 21 (2A202602392)

**Họ tên**: Đỗ Trịnh Huy Hoàng · **MSSV**: 2A202602392

| | |
|---|---|
| **Repo code + results** | https://github.com/HuyHoang1977/Day21-Track3-Finetuning-Lab |
| **Adapter (HuggingFace Hub)** | https://huggingface.co/HuyHoang1977/lab21-qwen35-triage-vi |
| **Base model** | https://huggingface.co/unsloth/Qwen3.5-4B |
| **Notebook Colab đã chạy** | https://colab.research.google.com/drive/1QMIa3cyNmbWAPkp22BKTwWuhsvDXGQZa |
| **Commit** | `d27c1c0` |
| **Phần cứng** | Colab Free · Tesla T4 · 14.6 GB khả dụng · fp16 |

## Cấu hình adapter

| | |
|---|---|
| Base | `unsloth/Qwen3.5-4B` |
| Vị trí | `text-linear` — 12 module |
| Rank / alpha | 16 / 32 |
| Learning rate | 1e-4 (cosine, warmup 3) |
| Effective batch | 16 (1 × grad-accum 16) |
| Optimizer steps | 30 (2 epoch trên 225 mẫu, seed 42) |
| Tham số huấn luyện | 32,464,896 |
| Peak VRAM | 8.78 GB |

## Kết quả chính

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.000 | 0.791 | 0.000 | 3310.8 |
| (b) base + optimized prompt | 0.765 | 0.791 | 1.000 | 1046.7 |
| (c) LoRA fine-tune | **0.965** | **0.522** | 1.000 | 1392.9 |

**Verdict: FAILED** — target Δ +0.200 nhưng regression Δ −0.269 (ngưỡng 0.020).
Chi tiết và phân tích nhân quả: [`submission/REPORT.md`](submission/REPORT.md).

> Adapter này **không nên deploy**. Nguyên nhân tụt regression đã được xác định:
> `valid_trace_rate = 0` — corpus 250 mẫu chỉ có câu trả lời JSON trần, không có
> trace suy luận nào để học.
