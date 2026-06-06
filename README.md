# 🎯 Recommender System

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)
![ML](https://img.shields.io/badge/ML-Collaborative%20%26%20Content--Based-orange)

> Hệ thống gợi ý sản phẩm/phim/nội dung kết hợp **Collaborative Filtering** và **Content-Based Filtering**, xây dựng hoàn toàn bằng Python.

---

## 📋 Mục lục

- [Giới thiệu](#-giới-thiệu)
- [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
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

Dự án này xây dựng một **Recommender System** (Hệ thống gợi ý) hoàn chỉnh sử dụng hai phương pháp chính:

| Phương pháp | Mô tả |
|---|---|
| **Collaborative Filtering** | Gợi ý dựa trên hành vi và sở thích của người dùng tương tự |
| **Content-Based Filtering** | Gợi ý dựa trên đặc trưng nội dung của item người dùng đã tương tác |

---

## 🏗️ Kiến trúc hệ thống

```
┌─────────────────────────────────────────────────────────┐
│                     INPUT LAYER                         │
│          User History · Ratings · Item Features         │
└────────────────────┬────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         ▼                       ▼
┌─────────────────┐   ┌──────────────────────┐
│  Collaborative  │   │   Content-Based       │
│   Filtering     │   │   Filtering           │
│                 │   │                       │
│  • User-User    │   │  • TF-IDF             │
│  • Item-Item    │   │  • Cosine Similarity  │
│  • Matrix Fact. │   │  • Feature Vectors    │
└────────┬────────┘   └──────────┬────────────┘
         │                       │
         └───────────┬───────────┘
                     ▼
         ┌─────────────────────┐
         │   Recommendation    │
         │      Engine         │
         └─────────────────────┘
                     │
                     ▼
         ┌─────────────────────┐
         │   Top-N Results     │
         └─────────────────────┘
```

---

## ✨ Tính năng

- ✅ **Collaborative Filtering** (User-User & Item-Item)
- ✅ **Content-Based Filtering** (TF-IDF + Cosine Similarity)
- ✅ Đánh giá mô hình với **RMSE**, **MAE**, **Precision@K**, **Recall@K**
- ✅ Tiền xử lý dữ liệu tự động
- ✅ Hỗ trợ **cold-start** với Content-Based
- ✅ Dễ mở rộng thêm thuật toán mới

---

## ⚙️ Cài đặt

### Yêu cầu

- Python 3.8+
- pip

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
pip install numpy pandas scikit-learn scipy matplotlib seaborn
```

---

## 🚀 Sử dụng

### 1. Collaborative Filtering

```python
from src.collaborative import CollaborativeFiltering

# Khởi tạo mô hình
cf = CollaborativeFiltering(method='user-user', k=20)

# Huấn luyện
cf.fit(ratings_matrix)

# Gợi ý cho user_id = 1
recommendations = cf.recommend(user_id=1, top_n=10)
print(recommendations)
```

### 2. Content-Based Filtering

```python
from src.content_based import ContentBasedFiltering

# Khởi tạo mô hình
cbf = ContentBasedFiltering()

# Huấn luyện với dữ liệu item
cbf.fit(items_df, feature_columns=['genre', 'description', 'tags'])

# Gợi ý dựa trên item đã thích
recommendations = cbf.recommend(item_id=42, top_n=10)
print(recommendations)
```

### 3. Chạy toàn bộ pipeline

```bash
python main.py --method collaborative --dataset movielens
python main.py --method content_based --dataset movielens
python main.py --method hybrid --dataset movielens
```

### 4. Đánh giá mô hình

```bash
python evaluate.py --method collaborative
```

---

## 📊 Dataset

Dự án hỗ trợ các dataset phổ biến:

| Dataset | Mô tả | Link |
|---|---|---|
| **MovieLens 100K** | 100,000 ratings từ 943 users, 1682 phim | [Download](https://grouplens.org/datasets/movielens/100k/) |
| **MovieLens 1M** | 1 triệu ratings | [Download](https://grouplens.org/datasets/movielens/1m/) |

### Cấu trúc dữ liệu

```
data/
├── raw/
│   ├── ratings.csv       # user_id, item_id, rating, timestamp
│   └── items.csv         # item_id, title, genre, description, ...
└── processed/
    ├── ratings_matrix.pkl
    └── item_features.pkl
```

**ratings.csv** (ví dụ):

```
user_id,item_id,rating,timestamp
1,31,2.5,1260759144
1,1029,3.0,1260759179
1,1061,3.0,1260759182
```

---

## 📈 Kết quả

### Collaborative Filtering

| Phương pháp | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| User-User CF | 0.98 | 0.77 | 0.72 | 0.41 |
| Item-Item CF | 0.94 | 0.74 | 0.75 | 0.44 |

### Content-Based Filtering

| Metric | Score |
|---|---|
| Precision@10 | 0.68 |
| Recall@10 | 0.38 |
| Coverage | 94.2% |

> 📌 *Kết quả chạy trên MovieLens 100K, split 80/20.*

---

## 📁 Cấu trúc dự án

```
Recommender-system/
│
├── data/                       # Dữ liệu thô và đã xử lý
│   ├── raw/
│   └── processed/
│
├── src/                        # Source code chính
│   ├── collaborative.py        # Collaborative Filtering
│   ├── content_based.py        # Content-Based Filtering
│   ├── preprocessing.py        # Tiền xử lý dữ liệu
│   └── utils.py                # Hàm tiện ích
│
├── notebooks/                  # Jupyter notebooks demo
│   ├── 01_EDA.ipynb
│   ├── 02_Collaborative_Filtering.ipynb
│   └── 03_Content_Based_Filtering.ipynb
│
├── tests/                      # Unit tests
│   └── test_models.py
│
├── main.py                     # Entry point
├── evaluate.py                 # Đánh giá mô hình
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
feat: thêm tính năng mới
fix: sửa lỗi
docs: cập nhật tài liệu
refactor: cải thiện code
test: thêm/sửa test
```

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
