# 🎯 Recommender System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Model](https://img.shields.io/badge/Model-NeuMF%20%7C%20SeqNeuMF-purple)

> Hệ thống gợi ý hiện đại kết hợp **NeuMF** (Neural Matrix Factorization) và **SeqNeuMF** (Sequential Neural Matrix Factorization), mô hình hóa cả sở thích tĩnh lẫn hành vi tuần tự của người dùng.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc mô hình](#-kiến-trúc-mô-hình)
- [Tính năng](#-tính-năng)
- [Cài đặt](#-cài-đặt)
- [Sử dụng](#-sử-dụng)
- [Dataset](#-dataset)
- [Kết quả](#-kết-quả)
- [Cấu trúc dự án](#-cấu-trúc-dự-án)
- [Đóng góp](#-đóng-góp)
- [Tác giả](#-tác-giả)

---

## 📖 Giới thiệu

Dự án triển khai hai mô hình deep learning cho hệ thống gợi ý:

| Mô hình | Ý tưởng cốt lõi | Điểm mạnh |
|---|---|---|
| **NeuMF** | Kết hợp GMF (Generalized Matrix Factorization) + MLP để học tương tác user–item phi tuyến | Nắm bắt sở thích dài hạn của người dùng |
| **SeqNeuMF** | Mở rộng NeuMF tích hợp mô hình hóa chuỗi hành vi (LSTM/GRU/Attention) | Nắm bắt xu hướng tương tác ngắn hạn và thứ tự hành vi |

> **Tóm tắt:** NeuMF học "bạn thích gì", SeqNeuMF học thêm "bạn vừa làm gì gần đây".

---

## 🏗️ Kiến trúc mô hình

### NeuMF

```
  User ID ──► [Embedding GMF] ──┐           ┌── GMF Layer (element-wise ×)
                                 ├── Concat ──┤
  Item ID ──► [Embedding GMF] ──┘           │
                                             ├──► [NeuMF Layer] ──► ŷ (score)
  User ID ──► [Embedding MLP] ──┐           │
                                 ├── Concat ──┤
  Item ID ──► [Embedding MLP] ──┘           └── MLP Layers (FC → ReLU → Dropout)
```

### SeqNeuMF

```
  User ID ──► [Embedding] ──────────────────────────────────────────┐
                                                                     │
  Item ID ──► [Embedding] ──────────────────────────────────────────┤
                                                                     ▼
  Interaction                                               ┌──────────────┐
  Sequence ──► [Embedding] ──► [LSTM / GRU / Attention] ──►│    Concat    │──► [MLP] ──► ŷ
  (i₁,i₂,...,iₜ)                  Sequential Encoder        └──────────────┘
```

---

## ✨ Tính năng

- ✅ **NeuMF** — Neural Matrix Factorization (GMF + MLP)
- ✅ **SeqNeuMF** — NeuMF + Sequential Encoder (LSTM / GRU / Self-Attention)
- ✅ Training với **BPR Loss** hoặc **Binary Cross-Entropy**
- ✅ Negative sampling tự động trong quá trình huấn luyện
- ✅ Đánh giá với **HR@K**, **NDCG@K**, **Recall@K**
- ✅ Hỗ trợ **early stopping** và **learning rate scheduler**
- ✅ Lưu và tải lại checkpoint mô hình
- ✅ Visualize loss curve và metrics qua các epoch

---

## ⚙️ Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- CUDA (khuyến nghị, không bắt buộc)

### Clone dự án

```bash
git clone https://github.com/duzktdtc-gif/Recommender-system.git
cd Recommender-system
```

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install torch numpy pandas scikit-learn scipy tqdm matplotlib
```

---

## 🚀 Sử dụng

### 1. NeuMF

```python
from src.neumf import NeuMF

# Khởi tạo mô hình
model = NeuMF(
    num_users=943,
    num_items=1682,
    emb_size=64,
    mlp_layers=[128, 64, 32],
    dropout=0.2
)

# Huấn luyện
model.fit(
    train_data,
    epochs=50,
    batch_size=256,
    lr=0.001,
    loss='bce'          # 'bce' hoặc 'bpr'
)

# Gợi ý Top-10 cho user_id = 1
recommendations = model.recommend(user_id=1, top_n=10)
print(recommendations)
```

### 2. SeqNeuMF

```python
from src.seqneumf import SeqNeuMF

# Khởi tạo mô hình
model = SeqNeuMF(
    num_users=943,
    num_items=1682,
    emb_size=64,
    seq_len=10,                  # Độ dài chuỗi lịch sử
    encoder_type='attention',    # 'lstm', 'gru', hoặc 'attention'
    mlp_layers=[128, 64],
    dropout=0.2
)

# Huấn luyện
model.fit(
    train_data,
    epochs=50,
    batch_size=256,
    lr=0.001
)

# Gợi ý dựa trên chuỗi hành vi gần đây
recommendations = model.recommend(
    user_id=1,
    interaction_seq=[31, 1029, 1061, 2105],   # Các item đã tương tác gần đây
    top_n=10
)
print(recommendations)
```

### 3. Chạy từ command line

```bash
# Huấn luyện NeuMF
python main.py --model neumf --dataset movielens --epochs 50

# Huấn luyện SeqNeuMF với GRU
python main.py --model seqneumf --encoder gru --seq_len 10 --dataset movielens

# Huấn luyện SeqNeuMF với Self-Attention
python main.py --model seqneumf --encoder attention --seq_len 20 --dataset movielens
```

### 4. Đánh giá mô hình

```bash
python evaluate.py --model neumf --checkpoint checkpoints/neumf_best.pt
python evaluate.py --model seqneumf --checkpoint checkpoints/seqneumf_best.pt
```

---

## 📊 Dataset

| Dataset | Users | Items | Interactions | Link |
|---|---|---|---|---|
| **MovieLens 100K** | 943 | 1,682 | 100,000 | [Download](https://grouplens.org/datasets/movielens/100k/) |
| **MovieLens 1M** | 6,040 | 3,706 | 1,000,209 | [Download](https://grouplens.org/datasets/movielens/1m/) |
| **Amazon Reviews** | — | — | — | [Download](https://jmcauley.ucsd.edu/data/amazon/) |

### Cấu trúc dữ liệu

```
data/
├── raw/
│   ├── ratings.csv           # user_id, item_id, rating, timestamp
│   └── items.csv             # item_id, title, genre, ...
└── processed/
    ├── train.pkl             # Tập huấn luyện
    ├── val.pkl               # Tập validation
    ├── test.pkl              # Tập kiểm tra
    └── user_sequences.pkl    # Chuỗi tương tác theo thứ tự thời gian (cho SeqNeuMF)
```

**ratings.csv** (ví dụ):

```
user_id,item_id,rating,timestamp
1,31,2.5,1260759144
1,1029,3.0,1260759179
1,1061,3.0,1260759182
```

> ⚠️ SeqNeuMF yêu cầu dữ liệu được **sắp xếp theo `timestamp`** để xây dựng chuỗi tương tác đúng thứ tự.

---

## 📈 Kết quả

Kết quả đánh giá trên **MovieLens 100K** (leave-one-out evaluation, Top-10):

| Mô hình | HR@10 | NDCG@10 | Recall@10 |
|---|---|---|---|
| NeuMF (GMF + MLP) | 0.684 | 0.412 | 0.389 |
| SeqNeuMF (LSTM) | 0.701 | 0.431 | 0.408 |
| SeqNeuMF (GRU) | 0.715 | 0.445 | 0.421 |
| **SeqNeuMF (Attention)** | **0.729** | **0.461** | **0.437** |

> 📌 *SeqNeuMF với Self-Attention cho kết quả tốt nhất nhờ khả năng nắm bắt mối quan hệ xa trong chuỗi hành vi.*

---

## 📁 Cấu trúc dự án

```
Recommender-system/
│
├── data/                         # Dữ liệu thô và đã xử lý
│   ├── raw/
│   └── processed/
│
├── src/                          # Source code chính
│   ├── neumf.py                  # Mô hình NeuMF
│   ├── seqneumf.py               # Mô hình SeqNeuMF
│   ├── encoders.py               # LSTM / GRU / Self-Attention encoder
│   ├── dataset.py                # Dataset & DataLoader
│   ├── loss.py                   # BCE Loss, BPR Loss
│   ├── metrics.py                # HR@K, NDCG@K, Recall@K
│   ├── preprocessing.py          # Tiền xử lý & xây dựng chuỗi
│   └── utils.py                  # Hàm tiện ích, checkpoint
│
├── notebooks/                    # Jupyter notebooks demo
│   ├── 01_EDA.ipynb
│   ├── 02_NeuMF.ipynb
│   └── 03_SeqNeuMF.ipynb
│
├── checkpoints/                  # Lưu model weights
│   ├── neumf_best.pt
│   └── seqneumf_best.pt
│
├── tests/                        # Unit tests
│   └── test_models.py
│
├── main.py                       # Entry point training
├── evaluate.py                   # Đánh giá mô hình
├── requirements.txt
└── README.md
```

---

## 🤝 Đóng góp

Mọi đóng góp đều được chào đón! Để đóng góp:

1. **Fork** repository này
2. Tạo branch mới: `git checkout -b feature/ten-tinh-nang`
3. Commit thay đổi: `git commit -m 'feat: thêm tính năng X'`
4. Push lên branch: `git push origin feature/ten-tinh-nang`
5. Tạo **Pull Request**

### Quy tắc commit

```
feat:     thêm tính năng mới
fix:      sửa lỗi
docs:     cập nhật tài liệu
refactor: cải thiện code
test:     thêm/sửa test
perf:     cải thiện hiệu năng mô hình
```

---

## 📚 Tài liệu tham khảo

- He, X. et al. (2017). [Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031). *WWW 2017*.
- Kang, W. et al. (2018). [Self-Attentive Sequential Recommendation](https://arxiv.org/abs/1808.09781). *ICDM 2018*.

---

## 👤 Tác giả

**duzktdtc-gif**

- GitHub: [@duzktdtc-gif](https://github.com/duzktdtc-gif)

---

## 📄 License

Dự án này được cấp phép theo [MIT License](LICENSE).

---

<p align="center">
  Made with ❤️ by duzktdtc-gif
</p>
