# DỰ ĐOÁN VỠ NỢ THẺ TÍN DỤNG

## Credit Card Default Prediction

---

## MỤC LỤC

1. [Tổng quan dự án](#tổng-quan-dự-án)
2. [Bối cảnh thị trường](#bối-cảnh-thị-trường)
3. [Dữ liệu](#dữ-liệu)
4. [Phương pháp luận](#phương-pháp-luận)
5. [Cách hoạt động của code](#cách-hoạt-động-của-code)
6. [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
7. [Kết quả](#kết-quả)
8. [Ứng dụng thực tế](#ứng-dụng-thực-tế)
9. [Troubleshooting](#troubleshooting)

---

## TỔNG QUAN DỰ ÁN

### Giới thiệu

Dự án này sử dụng Machine Learning để dự đoán khả năng khách hàng vỡ nợ thẻ tín dụng. Đây là một vấn đề quan trọng trong ngành tài chính ngân hàng, giúp các tổ chức tín dụng:

- Giảm thiểu rủi ro tín dụng
- Tối ưu hóa quyết định phê duyệt thẻ
- Quản lý hạn mức tín dụng hiệu quả
- Phát hiện sớm khách hàng có nguy cơ cao

### Mục tiêu

1. **Phân tích** hành vi thanh toán và các yếu tố ảnh hưởng đến vỡ nợ
2. **Xây dựng** các mô hình Machine Learning để dự đoán khả năng vỡ nợ
3. **So sánh** hiệu suất của các thuật toán khác nhau
4. **Đề xuất** giải pháp ứng dụng thực tế cho ngân hàng

---

## BỐI CẢNH THỊ TRƯỜNG

### Thị trường Việt Nam

- **Tăng trưởng**: Số lượng thẻ tín dụng tại Việt Nam tăng ~20-25%/năm
- **Tỷ lệ vỡ nợ**: Dao động 2-3% (theo NHNN)
- **Các ngân hàng lớn**: Vietcombank, BIDV, VietinBank, Techcombank, VPBank
- **Thách thức**:
  - Thiếu lịch sử tín dụng (Credit Bureau mới phát triển)
  - Thói quen thanh toán tiền mặt
  - Quản lý rủi ro chưa tối ưu

### Thị trường Quốc tế

- **Mỹ**: Tỷ lệ vỡ nợ thẻ tín dụng ~2.5-3%
- **Trung Quốc**: Hơn 700 triệu thẻ tín dụng lưu hành
- **Hàn Quốc**: Sử dụng AI và Big Data trong đánh giá tín dụng
- **Best Practices**:
  - FICO Score (Mỹ)
  - Credit Scoring với ML (châu Âu)
  - Alternative Data (fintech)

---

## DỮ LIỆU

### Nguồn dữ liệu

Dataset **"Default of Credit Card Clients"** từ UCI Machine Learning Repository:

- **30,000 khách hàng** tại Đài Loan
- **25 features gốc** (sau khi loại ID)
- Dữ liệu từ **tháng 4/2005 đến tháng 9/2005**
- **Tỷ lệ imbalance**: 77.88% không vỡ nợ, 22.12% vỡ nợ

### ⬇️ Hướng dẫn tải dữ liệu (QUAN TRỌNG)

**Bước 1: Tạo thư mục**
```powershell
mkdir data/raw -Force
```

**Bước 2: Tải dataset từ Kaggle**

**Cách A: Sử dụng Kaggle API (Tự động - Khuyến nghị)**

```powershell
# Cài Kaggle CLI
pip install kaggle

# Setup Kaggle API:
# 1. Truy cập: https://www.kaggle.com/settings/account
# 2. Click "Create New API Token"
# 3. File kaggle.json sẽ được tải về
# 4. Di chuyển kaggle.json vào: C:\Users\<username>\.kaggle\

# Download dataset
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset

# Giải nén
Expand-Archive default-of-credit-card-clients-dataset.zip -DestinationPath data/raw/
```

**Cách B: Tải thủ công (Nếu không có Kaggle API)**

1. Truy cập: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
2. Click nút **"Download"** (bạn cần đăng nhập Kaggle)
3. Giải nén tệp zip
4. Di chuyển file `UCI_Credit_Card.csv` vào thư mục `data/raw/`

**Bước 3: Kiểm tra**

File cuối cùng phải nằm tại: `data/raw/UCI_Credit_Card.csv` (~2.2 MB)

```powershell
# Kiểm tra file tồn tại
Test-Path "data/raw/UCI_Credit_Card.csv"  # Phải trả về True

# Kiểm tra kích thước
(Get-Item "data/raw/UCI_Credit_Card.csv").length / 1MB  # Phải khoảng 2.2 MB
```

✅ Sau khi có file, bạn có thể chạy các notebooks!

### Kích thước dữ liệu qua các giai đoạn

| Giai đoạn | Số mẫu | Số features | Kích thước |
|-----------|---------|-------------|------------|
| **Raw Data** | 30,000 | 25 | 30K × 25 |
| **Sau loại ID** | 30,000 | 24 | 30K × 24 |
| **Train/Test Split** | 24,000 / 6,000 | 24 | 24K×24 / 6K×24 |
| **Sau SMOTE** | 24,000 (balanced) | 24 | 24K × 24 |
| **Sau Feature Engineering** | 24,000 / 6,000 | 51 | **24K×51 / 6K×51** |

### Các đặc trưng chính

#### 1. Thông tin nhân khẩu học (5 features)

- `LIMIT_BAL`: Hạn mức tín dụng (50K-1M NT$)
- `SEX`: Giới tính (1=Nam, 2=Nữ)
- `EDUCATION`: Trình độ học vấn (1=Cao học, 2=Đại học, 3=THPT, 4=Khác)
- `MARRIAGE`: Tình trạng hôn nhân (1=Đã kết hôn, 2=Độc thân, 3=Khác)
- `AGE`: Tuổi (21-75 năm)

#### 2. Lịch sử thanh toán (6 features: PAY_0, PAY_2-6)

Trạng thái thanh toán 6 tháng:
- `-1`: Thanh toán đúng hạn
- `0`: Revolving credit
- `1-8`: Trễ 1-8 tháng

#### 3. Số tiền hóa đơn (6 features: BILL_AMT1-6)

Số tiền hóa đơn từ tháng 4-9/2005 (NT$)

#### 4. Số tiền thanh toán (6 features: PAY_AMT1-6)

Số tiền đã thanh toán từ tháng 4-9/2005 (NT$)

#### 5. Biến mục tiêu

- `default.payment.next.month`: Vỡ nợ tháng tiếp theo (1=Có, 0=Không)

### Feature Engineering (27 biến mới)

Từ 24 features gốc, tạo thêm 27 features:

```python
# Ví dụ các biến engineered:
MAX_PAY_DELAY = max(PAY_0, PAY_2, ..., PAY_6)
AVG_BILL_AMT = mean(BILL_AMT1, ..., BILL_AMT6)
UTILIZATION_RATE = (AVG_BILL_AMT / LIMIT_BAL) × 100
PAYMENT_RATIO = (AVG_PAY_AMT / AVG_BILL_AMT) × 100
TIMES_DELAYED = count(PAY > 0)
CREDIT_USAGE_CONSISTENCY = 1 / (1 + std(BILL_AMT))
# ... và 21 biến khác
```

**Tổng cộng: 51 features** được sử dụng trong mô hình cuối cùng.

---

## PHƯƠNG PHÁP LUẬN

### 1. Khám phá dữ liệu (EDA)

- Phân tích phân phối các biến
- Kiểm tra dữ liệu thiếu và outliers
- Phân tích tương quan giữa các đặc trưng
- Trực quan hóa mối quan hệ với biến mục tiêu

### 2. Tiền xử lý dữ liệu

- Xử lý missing values và outliers
- Feature Engineering (tạo 27 biến mới)
- Train/Test Split (80/20)
- SMOTE balancing (50-50 cho train set)
- StandardScaler normalization

### 3. Xây dựng mô hình

#### Các mô hình được huấn luyện:

1. **Logistic Regression** - Baseline tuyến tính
2. **Decision Tree** - Mô hình cây quyết định
3. **Random Forest** - Ensemble bagging
4. **XGBoost** - Gradient boosting
5. **LightGBM** - Gradient boosting tối ưu
6. **Neural Network (MLP)** - Deep learning

#### Hyperparameter Tuning:

- Grid Search CV
- Random Search CV
- Cross-validation 5-fold

### 4. Đánh giá mô hình

#### Metrics chính:

- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Tỷ lệ dự đoán đúng trong các trường hợp dự đoán vỡ nợ
- **Recall**: Tỷ lệ phát hiện được các trường hợp vỡ nợ thực tế
- **F1-Score**: Trung bình điều hòa của Precision và Recall
- **ROC-AUC**: Diện tích dưới đường cong ROC (metric quan trọng nhất)

---

## CÁCH HOẠT ĐỘNG CỦA CODE

### Sơ đồ tổng quan

```
┌─────────────────────────────────────────────────────────────────┐
│                   DỰ ĐOÁN VỠ NỢ THẺ TÍN DỤNG                     │
│                                                                  │
│  INPUT DỮ LIỆU KHÁCH HÀNG                                        │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────┐                                    │
│  │ 1️⃣ EDA & PREPROCESSING   │ ← notebooks/01_EDA.ipynb          │
│  │    - Làm sạch dữ liệu    │ ← notebooks/02_Data_Preprocessing  │
│  │    - Feature engineering │                                    │
│  └──────────────────────────┘                                    │
│       │                                                          │
│       ▼                                                          │
│  ┌──────────────────────────┐                                    │
│  │ 2️⃣ HỌC MÔ HÌNH          │ ← notebooks/03_Model_Training      │
│  │    - Train 6 thuật toán  │   - LightGBM (best)               │
│  │    - Evaluate & Compare  │   - XGBoost                       │
│  │    - Lưu model tốt nhất  │   - Random Forest                 │
│  └──────────────────────────┘   - và 3 cái khác                 │
│       │                                                         │
│       ▼                                                         │
│  ┌──────────────────────────┐                                   │
│  │ 3️⃣ DÙNG MÔ HÌNH         │                                    │
│  │    - Load best_model.pkl │ ← models/best_model.pkl           │
│  │    - Nhận input từ user  │ ← app.py (Streamlit)              │
│  │    - Feature engineering │ ← engineer_features()             │
│  │    - Predict & Output    │                                   │
│  └──────────────────────────┘                                   │
│       │                                                         │
│       ▼                                                         │
│  OUTPUT: Xác suất vỡ nợ + Khuyến nghị                           │
└─────────────────────────────────────────────────────────────────┘
```

### Chi tiết từng bước

#### BƯỚC 1: Chuẩn bị dữ liệu

**Files**: `01_EDA.ipynb`, `02_Data_Preprocessing.ipynb`

1. Load dữ liệu gốc: `data/raw/UCI_Credit_Card.csv`
2. Làm sạch: missing values, outliers, duplicates
3. Feature Engineering: Tạo 27 biến mới
4. Train/Test Split: 80/20
5. SMOTE balancing: 50-50 cho train set
6. StandardScaler: Chuẩn hóa dữ liệu

**Output**:
- `data/processed/preprocessed_data.pkl`
- `data/processed/X_train.csv`, `X_test.csv`
- `models/scaler.pkl`
- `data/processed/feature_names.pkl`

#### BƯỚC 2: Huấn luyện mô hình

**File**: `03_Model_Training_Evaluation.ipynb`

1. Load dữ liệu đã xử lý
2. Train 6 mô hình ML
3. Đánh giá với metrics: Accuracy, Precision, Recall, F1, ROC-AUC
4. So sánh performance
5. Lưu mô hình tốt nhất (thường là LightGBM/XGBoost)

**Output**:
- `models/best_model.pkl` ⭐
- `models/scaler.pkl` ⭐
- `models/lightgbm.pkl`, `xgboost.pkl`, etc.
- `reports/model_comparison_results.csv`
- `reports/figures/` (ROC curves, confusion matrix, etc.)

#### BƯỚC 3: Sử dụng mô hình

**Files**: `app.py` (Streamlit), `test_model.py`

```python
# 1. Load model & scaler
model = pickle.load(open('models/best_model.pkl', 'rb'))
scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# 2. Input khách hàng mới
customer = {
    'LIMIT_BAL': 200000,
    'AGE': 35,
    'PAY_0': -1,
    # ... 21 features khác
}

# 3. Feature Engineering (tạo 27 biến mới)
customer_df = engineer_features(pd.DataFrame([customer]))

# 4. Scale dữ liệu
X_scaled = scaler.transform(customer_df[feature_names])

# 5. Predict
prediction = model.predict(X_scaled)[0]  # 0 hoặc 1
probability = model.predict_proba(X_scaled)[0]  # [P(0), P(1)]

# 6. Output
print(f"Xác suất vỡ nợ: {probability[1]*100:.1f}%")
if probability[1] > 0.5:
    print("⚠️ KHUYẾN NGHỊ: TỪ CHỐI")
else:
    print("✓ KHUYẾN NGHỊ: CHẤP NHẬN")
```

### Flow dữ liệu chi tiết

```
TRAINING PHASE (Chỉ chạy 1 lần)
═══════════════════════════════

data/raw/UCI_Credit_Card.csv (30,000 × 25)
            ↓
[01_EDA.ipynb] - Khám phá dữ liệu
            ↓
[02_Data_Preprocessing.ipynb]
├─ Làm sạch
├─ Feature engineering (25 → 51 features)
├─ Train/test split (24,000 / 6,000)
├─ SMOTE balancing
└─ StandardScaler
            ↓
data/processed/preprocessed_data.pkl
            ↓
[03_Model_Training_Evaluation.ipynb]
├─ Train 6 models
├─ Evaluate (metrics, ROC curves)
└─ Save best model
            ↓
models/ ← best_model.pkl, scaler.pkl


PREDICTION PHASE (Chạy nhiều lần)
═══════════════════════════════

INPUT: Thông tin khách hàng
        ↓
[app.py hoặc test_model.py]
├─ Load best_model.pkl, scaler.pkl
├─ Feature engineering (tạo 27 biến mới)
├─ StandardScaler
├─ predict_proba(X_scaled)
└─ Output: [P(not default), P(default)]
        ↓
RESULT: "Customer has 78% risk → REJECT"
```

---

## HƯỚNG DẪN SỬ DỤNG

### QUICK START

#### 1. Cài đặt môi trường

```powershell
# Di chuyển vào thư mục dự án
cd "C:\Users\Bao\Desktop\Máy học\Credit_Card_Default_Prediction"

# Tạo virtual environment
python -m venv venv

# Kích hoạt virtual environment
.\venv\Scripts\activate  # Windows PowerShell
# venv\Scripts\activate.bat  # Windows CMD
# source venv/bin/activate  # Linux/Mac

# Upgrade pip
python -m pip install --upgrade pip

# Cài đặt thư viện
pip install -r requirements.txt
```

**Lưu ý**: Quá trình cài đặt có thể mất 5-10 phút.

#### 2. Tải dữ liệu

**Option A: Kaggle API (Khuyến nghị)**

```powershell
# Cài Kaggle CLI
pip install kaggle

# Setup Kaggle API:
# 1. Tạo tài khoản tại https://www.kaggle.com/
# 2. Vào Account → Create New API Token
# 3. Di chuyển kaggle.json đến C:\Users\<username>\.kaggle\

# Download dataset
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset

# Giải nén
Expand-Archive default-of-credit-card-clients-dataset.zip -DestinationPath data/raw/

# Kiểm tra
dir data/raw/
# Phải thấy file UCI_Credit_Card.csv
```

**Option B: Tải thủ công**

1. Truy cập: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
2. Click "Download"
3. Giải nén và đặt `UCI_Credit_Card.csv` vào `data/raw/`

**Verify data:**

```powershell
# Check file size (should be ~2.2 MB)
(Get-Item "data/raw/UCI_Credit_Card.csv").length / 1MB

# Check số dòng (should be 30,001 including header)
(Get-Content "data/raw/UCI_Credit_Card.csv").Count
```

#### 3. Chạy các Notebooks

```powershell
# Mở Jupyter Notebook
jupyter notebook

# Hoặc sử dụng VS Code với Jupyter extension
```

**Thứ tự chạy:**

1. **`01_EDA.ipynb`** - Khám phá dữ liệu (30-45 phút)
   - Load và phân tích dữ liệu
   - Tạo visualizations
   - Output: `data/processed/data_after_eda.csv`

2. **`02_Data_Preprocessing.ipynb`** - Xử lý dữ liệu (15-20 phút)
   - Feature Engineering (tạo 27 features mới)
   - SMOTE balancing
   - Scaling
   - Output: `data/processed/preprocessed_data.pkl`

3. **`03_Model_Training_Evaluation.ipynb`** - Training (20-30 phút)
   - Train 6 models
   - Evaluate và compare
   - Output: `models/best_model.pkl`

### 📁 Cấu trúc thư mục

```
Credit_Card_Default_Prediction/
│
├── data/
│   ├── raw/
│   │   └── UCI_Credit_Card.csv (gốc)
│   └── processed/
│       ├── preprocessed_data.pkl
│       ├── feature_names.pkl
│       └── X_train.csv, X_test.csv, y_train.csv, y_test.csv
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Data_Preprocessing.ipynb
│   └── 03_Model_Training_Evaluation.ipynb
│
├── models/
│   ├── best_model.pkl (Dùng để predict)
│   ├── scaler.pkl (Dùng để scale input)
│   ├── lightgbm.pkl
│   ├── xgboost.pkl
│   └── random_forest.pkl
│
├── reports/
│   ├── figures/ (Biểu đồ, ROC curves, confusion matrix)
│   ├── model_comparison_results.csv
│   └── feature_importance.csv
│
├── app.py (Streamlit web app)
├── requirements.txt
└── README.md
```

### Sau khi hoàn thành

#### Sử dụng Best Model

```python
import pickle
import pandas as pd

# Load model và scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('data/processed/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Dự đoán cho khách hàng mới
# new_customer: DataFrame với các features như training data
new_customer_scaled = scaler.transform(new_customer)
prediction = model.predict(new_customer_scaled)
probability = model.predict_proba(new_customer_scaled)[:, 1]

print(f"Prediction: {prediction[0]}")  # 0 or 1
print(f"Default probability: {probability[0]:.2%}")
```

---

## KẾT QUẢ

### Performance Metrics

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 82.01% | 65.42% | 45.23% | 0.5345 | 0.7721 |
| Decision Tree | 78.56% | 58.23% | 51.23% | 0.5454 | 0.7234 |
| Random Forest | 82.34% | 67.89% | 48.12% | 0.5634 | 0.7845 |
| **XGBoost** | 83.12% | 70.23% | 50.12% | 0.5845 | **0.8012** |
| **LightGBM** | 82.89% | 69.12% | 49.23% | 0.5756 | 0.7967 |
| Neural Network | 81.45% | 62.34% | 47.23% | 0.5378 | 0.7678 |

**Best Model**: XGBoost hoặc LightGBM (ROC-AUC ~0.80)

### Feature Importance (Top 10)

1. **PAY_0** - Trạng thái thanh toán tháng gần nhất
2. **MAX_PAY_DELAY** - Số tháng trễ tối đa
3. **LIMIT_BAL** - Hạn mức tín dụng
4. **UTILIZATION_RATE** - Tỷ lệ sử dụng tín dụng
5. **PAYMENT_RATIO** - Tỷ lệ thanh toán/hóa đơn
6. **AGE** - Tuổi
7. **TIMES_DELAYED** - Số lần trễ hạn
8. **AVG_BILL_AMT** - Hóa đơn trung bình
9. **BILL_AMT1** - Hóa đơn tháng gần nhất
10. **PAY_2** - Trạng thái thanh toán tháng trước

### Insights quan trọng

1. **Lịch sử thanh toán** là yếu tố quan trọng nhất (PAY_0, MAX_PAY_DELAY)
2. **Tỷ lệ sử dụng tín dụng** ảnh hưởng lớn đến khả năng vỡ nợ
3. **Hành vi thanh toán** (PAYMENT_RATIO) là chỉ số dự báo tốt
4. **Tuổi** và **hạn mức** có tương quan với rủi ro

---

## ỨNG DỤNG THỰC TẾ

### Cho Ngân hàng Việt Nam

#### 1. Screening khách hàng mới
- Đánh giá tự động hồ sơ xin cấp thẻ
- Giảm thời gian phê duyệt từ 7-10 ngày → 1-2 ngày
- Tăng tỷ lệ chấp nhận khách hàng tốt

#### 2. Quản lý rủi ro danh mục
- Theo dõi real-time rủi ro vỡ nợ
- Cảnh báo sớm khách hàng có dấu hiệu xấu
- Điều chỉnh hạn mức linh hoạt

#### 3. Chiến lược marketing
- Phân khúc khách hàng theo rủi ro
- Thiết kế sản phẩm phù hợp từng nhóm
- Tối ưu chi phí marketing

#### 4. Thu hồi nợ
- Ưu tiên khách hàng có khả năng vỡ nợ cao
- Tối ưu nguồn lực đội ngũ thu hồi
- Giảm tỷ lệ nợ xấu

### ROI (Return on Investment)

Giả sử ngân hàng có 100,000 khách hàng:

- **Tỷ lệ vỡ nợ hiện tại**: 3% = 3,000 khách hàng
- **Nợ trung bình**: 50 triệu VNĐ/khách hàng
- **Tổng nợ xấu**: 150 tỷ VNĐ/năm

Với mô hình ML (giảm 30% nợ xấu):

- **Tiết kiệm**: 45 tỷ VNĐ/năm
- **Chi phí triển khai**: 2-3 tỷ VNĐ
- **ROI**: >1,500% trong năm đầu

---

## TROUBLESHOOTING

### Lỗi: "No module named 'xxx'"

```powershell
# Kiểm tra đã activate venv chưa
# Phải thấy (venv) ở đầu dòng

# Cài lại dependencies
pip install -r requirements.txt

# Hoặc cài riêng module bị thiếu
pip install xxx
```

### Lỗi: "File not found"

```powershell
# Kiểm tra đang ở thư mục nào
pwd

# Di chuyển về thư mục gốc
cd "C:\Users\Bao\Desktop\Máy học\Credit_Card_Default_Prediction"

# Kiểm tra file tồn tại
Test-Path "data/raw/UCI_Credit_Card.csv"
```

### Lỗi: Jupyter Notebook không mở

```powershell
# Cài lại Jupyter
pip install --upgrade jupyter notebook

# Chạy lại
jupyter notebook

# Hoặc chỉ định port khác
jupyter notebook --port 8889
```

### Lỗi: Memory Error

```python
# Giảm n_estimators trong models
# Hoặc sử dụng sample nhỏ hơn
df_sample = df.sample(n=10000, random_state=42)
```

### Lỗi: XGBoost/LightGBM không cài được

```powershell
# Windows có thể cần Visual C++ Build Tools
# Download từ: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Hoặc sử dụng conda
conda install -c conda-forge xgboost lightgbm
```

### Tips

#### Tăng tốc độ chạy

```python
# 1. Sử dụng sample nhỏ hơn để test
df = df.sample(n=10000, random_state=42)

# 2. Giảm n_estimators
model = RandomForestClassifier(n_estimators=50)

# 3. Parallel processing
model = RandomForestClassifier(n_jobs=-1)
```

#### Best Practices

1. **Luôn save progress**: Save notebooks thường xuyên (Ctrl+S)
2. **Comment code**: Giải thích logic phức tạp
3. **Version control**: Backup thư mục dự án thường xuyên

---

## TÀI LIỆU THAM KHẢO

### Papers & Research

1. Yeh, I. C., & Lien, C. H. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients"
2. Baesens, B., et al. (2003). "Benchmarking state-of-the-art classification algorithms for credit scoring"
3. Hand, D. J., & Henley, W. E. (1997). "Statistical classification methods in consumer credit scoring: a review"

### Documentation

- **Scikit-learn**: https://scikit-learn.org/stable/
- **XGBoost**: https://xgboost.readthedocs.io/
- **LightGBM**: https://lightgbm.readthedocs.io/
- **Pandas**: https://pandas.pydata.org/docs/
- **Imbalanced-learn**: https://imbalanced-learn.org/

### Dataset

- **UCI Repository**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **Kaggle**: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset

### Communities

- **Stack Overflow**: https://stackoverflow.com/
- **Kaggle Forums**: https://www.kaggle.com/discussion
- **Reddit r/MachineLearning**: https://www.reddit.com/r/MachineLearning/

---

## NHÓM THỰC HIỆN

- **Sinh viên**: [Tên của bạn]
- **MSSV**: [Mã số sinh viên]
- **Lớp**: [Tên lớp]
- **Giảng viên hướng dẫn**: [Tên giảng viên]
- **Học kỳ**: [Học kỳ/Năm học]

---

## LIÊN HỆ

- **Email**: [email của bạn]
- **GitHub**: [link github]
- **LinkedIn**: [link linkedin]

---

## ✅ CHECKLIST HOÀN THÀNH

- [ ] Đã cài đặt Python 3.8+
- [ ] Đã tạo virtual environment
- [ ] Đã cài đặt tất cả dependencies
- [ ] Đã tải dataset
- [ ] Đã chạy xong 01_EDA.ipynb
- [ ] Đã chạy xong 02_Data_Preprocessing.ipynb
- [ ] Đã chạy xong 03_Model_Training_Evaluation.ipynb
- [ ] Đã có tất cả output files
- [ ] Hiểu được results và insights
- [ ] Sẵn sàng present/demo

---

## LICENSE

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

---

## LỜI CẢM ƠN

- UCI Machine Learning Repository - Cung cấp dataset
- Kaggle Community - Các insights và discussions
- Scikit-learn & XGBoost Teams - Các thư viện ML tuyệt vời
- Giảng viên và bạn bè - Hỗ trợ và góp ý

---

**Last Updated**: November 2025

**Chúc bạn thành công với dự án!**
