# Recommender System
## Giới thiệu 

Dự án xây dựng hệ thống gợi ý video ngắn dựa trên bộ dữ liệu MicroLens.  
Hệ thống sử dụng dữ liệu tương tác giữa người dùng và video, kết hợp với đặc trưng hình ảnh của video để đưa ra danh sách gợi ý Top-K.
Project hỗ trợ hai mô hình chính:

- **Multimodal NeuMF**: kết hợp Neural Matrix Factorization với visual embeddings.
- **SeqNeuMF**: mở rộng NeuMF bằng cách khai thác chuỗi lịch sử tương tác của người dùng.

Ngoài phần huấn luyện mô hình, project còn có giao diện Streamlit để khám phá dữ liệu, xem lịch sử người dùng và demo kết quả gợi ý.

---



---

## Tính năng

-  **NeuMF** — Neural Matrix Factorization (GMF + MLP)
-  **SeqNeuMF** — NeuMF + Sequential Encoder (LSTM / GRU / Self-Attention)
-  Training với **BPR Loss** hoặc **Binary Cross-Entropy**
-  Negative sampling tự động trong quá trình huấn luyện
-  Đánh giá với **HR@K**, **NDCG@K**, **Recall@K**
-  Hỗ trợ **early stopping** và **learning rate scheduler**
-  Lưu và tải lại checkpoint mô hình
-  Visualize loss curve và metrics qua các epoch

---

## Cài đặt

### Yêu cầu

- Python 3.8+
- PyTorch 2.0+
- CUDA (khuyến nghị, không bắt buộc)

### Cài đặt thư viện

```bash
pip install -r requirements.txt
```

Hoặc cài thủ công:

```bash
pip install torch numpy pandas scikit-learn scipy tqdm matplotlib
```

---

## Sử dụng

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

## Kết quả

Kết quả đánh giá trên **MicroLens** theo phương pháp leave-one-out evaluation với Top-10 recommendation:

| Mô hình | HR@10 | NDCG@10 |
|---|---:|---:|
| M-NeuMF | 0.3840 | 0.2219 |
| M-SeqNeuMF | 0.5206 | 0.3196 |

Như vậy, M-SeqNeuMF cho kết quả tốt hơn vì mô hình khai thác thêm chuỗi lịch sử tương tác của người dùng.

## Cấu trúc dự án

```
Recommender-system/
├── data/
│   └── microlens-5k/
├── src/
│   ├── components/      # Reusable UI components
│   ├── pages/           # Streamlit page definitions
│   ├── services/        # Backend logic for recommendations
│   ├── state/           # Session state management
│   ├── app.py           # Streamlit entrypoint
│   ├── data_utils.py    # Data loading utilities
│   ├── engine.py        # Training engine
│   ├── inference.py     # Inference script
│   ├── metrics.py       # Evaluation metrics
│   ├── neumf.py         # NeuMF model definition
│   ├── seqneumf.py      # Sequential NeuMF model definition
│   └── train.py         # Training script
├── utils/
│   └── download_data.py
└── README.md
```


<p align="center">
</p>
