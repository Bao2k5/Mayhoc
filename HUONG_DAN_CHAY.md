# HƯỚNG DẪN CHẠY DỰ ÁN

# Credit Card Default Prediction

---

## ⚡ QUICK START

### 1. Cài đặt môi trường

```powershell
# Di chuyển vào thư mục dự án
cd "C:\Users\Bao\Desktop\Máy học\Credit_Card_Default_Prediction"

# Tạo virtual environment (khuyến nghị)
python -m venv venv

# Kích hoạt virtual environment
.\venv\Scripts\activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt
```

### 2. Tải dữ liệu

**Option 1: Tải từ Kaggle (Dễ nhất)**

```powershell
# Cài đặt Kaggle CLI
pip install kaggle

# Download dataset
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset

# Giải nén
Expand-Archive default-of-credit-card-clients-dataset.zip -DestinationPath data/raw/
```

**Option 2: Tải thủ công**

1. Truy cập: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
2. Click "Download"
3. Giải nén và đặt file `UCI_Credit_Card.csv` vào `data/raw/`

### 3. Chạy các Notebooks theo thứ tự

```powershell
# Mở Jupyter Notebook
jupyter notebook

# Hoặc sử dụng VS Code với Jupyter extension
```

**Thứ tự chạy:**

1. `notebooks/01_EDA.ipynb` - Khám phá dữ liệu (30-45 phút)
2. `notebooks/02_Data_Preprocessing.ipynb` - Xử lý dữ liệu (15-20 phút)
3. `notebooks/03_Model_Training_Evaluation.ipynb` - Training models (20-30 phút)

---

## 📋 CHI TIẾT TỪNG BƯỚC

### BƯỚC 1: Chuẩn bị môi trường

#### Kiểm tra Python version

```powershell
python --version
# Cần Python 3.8 trở lên
```

#### Nếu chưa có Python

- Tải từ: https://www.python.org/downloads/
- Chọn "Add Python to PATH" khi cài đặt

#### Tạo virtual environment

```powershell
# Tại thư mục gốc của dự án
python -m venv venv

# Kích hoạt (PowerShell)
.\venv\Scripts\activate

# Kích hoạt (CMD)
venv\Scripts\activate.bat

# Kiểm tra đã activate chưa
# Sẽ thấy (venv) ở đầu dòng command
```

#### Cài đặt thư viện

```powershell
# Upgrade pip
python -m pip install --upgrade pip

# Cài đặt tất cả dependencies
pip install -r requirements.txt

# Kiểm tra cài đặt
pip list
```

**Lưu ý:** Quá trình cài đặt có thể mất 5-10 phút tùy tốc độ mạng.

---

### BƯỚC 2: Tải và chuẩn bị dữ liệu

#### Option A: Sử dụng Kaggle API (Khuyến nghị)

**Setup Kaggle API:**

1. Tạo tài khoản Kaggle: https://www.kaggle.com/
2. Vào Account → Create New API Token
3. File `kaggle.json` sẽ được download
4. Di chuyển `kaggle.json` đến:
   - Windows: `C:\Users\<username>\.kaggle\`
   - Linux/Mac: `~/.kaggle/`

**Download dataset:**

```powershell
# Cài Kaggle
pip install kaggle

# Download
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset

# Giải nén
Expand-Archive default-of-credit-card-clients-dataset.zip -DestinationPath data/raw/

# Kiểm tra
dir data/raw/
# Phải thấy file UCI_Credit_Card.csv
```

#### Option B: Tải thủ công

1. Truy cập: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
2. Click nút "Download" (cần đăng nhập)
3. Giải nén file zip
4. Copy `UCI_Credit_Card.csv` vào thư mục `data/raw/`

**Verify data:**

```powershell
# Check file size (should be ~2.2 MB)
(Get-Item "data/raw/UCI_Credit_Card.csv").length / 1MB

# Check số dòng (should be 30,001 including header)
(Get-Content "data/raw/UCI_Credit_Card.csv").Count
```

---

### BƯỚC 3: Chạy Exploratory Data Analysis

```powershell
# Mở Jupyter Notebook
jupyter notebook

# Browser sẽ tự động mở
# Navigate to notebooks/01_EDA.ipynb
```

**Hoặc sử dụng VS Code:**

1. Mở VS Code
2. Mở file `01_EDA.ipynb`
3. Select Kernel: Python 3 (venv)
4. Run All Cells (Ctrl+Shift+Enter)

**Trong notebook này:**

- Load và khám phá dữ liệu
- Phân tích từng features
- Tạo visualizations
- Lưu processed data vào `data/processed/`

**Expected outputs:**

- `data/processed/data_after_eda.csv`
- Figures trong `reports/figures/`:
  - `target_distribution.png`
  - `gender_analysis.png`
  - `education_analysis.png`
  - `age_analysis.png`
  - `credit_limit_analysis.png`
  - `payment_history_analysis.png`
  - `bill_payment_analysis.png`
  - `correlation_analysis.png`

**Thời gian:** ~30-45 phút

---

### BƯỚC 4: Data Preprocessing & Feature Engineering

```powershell
# Chạy notebook 02
# notebooks/02_Data_Preprocessing.ipynb
```

**Trong notebook này:**

- Xử lý dữ liệu bất thường
- Feature Engineering (tạo 27 features mới)
- Feature Scaling
- Handle Imbalanced Data (SMOTE)
- Train-Test Split
- Lưu preprocessed data

**Expected outputs:**

- `data/processed/preprocessed_data.pkl` - Tất cả versions của data
- `data/processed/X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `models/scaler.pkl` - StandardScaler object
- `data/processed/feature_names.pkl`
- Figures:
  - `feature_scaling.png`
  - `imbalanced_data_handling.png`

**Thời gian:** ~15-20 phút

---

### BƯỚC 5: Model Training & Evaluation

```powershell
# Chạy notebook 03
# notebooks/03_Model_Training_Evaluation.ipynb
```

**Trong notebook này:**

- Train 6 ML models:
  1. Logistic Regression
  2. Decision Tree
  3. Random Forest
  4. XGBoost
  5. LightGBM
  6. Neural Network
- Đánh giá và so sánh
- Feature Importance Analysis
- Cost-Benefit Analysis
- Lưu models

**Expected outputs:**

- `models/*.pkl` - Tất cả trained models
- `models/best_model.pkl` - Best model (thường là XGBoost)
- `reports/model_comparison_results.csv`
- `reports/feature_importance.csv`
- Figures:
  - `model_comparison.png`
  - `roc_curves_all.png`
  - `confusion_matrix_best.png`
  - `feature_importance.png`
  - `cost_benefit_analysis.png`

**Thời gian:** ~20-30 phút

---

## 🎯 SAU KHI HOÀN THÀNH

### Kiểm tra kết quả

```powershell
# Check tất cả files đã tạo
dir data/processed/
dir models/
dir reports/
dir reports/figures/
```

### Đọc báo cáo

Mở file `reports/Bao_Cao_Tieu_Luan.md` để đọc báo cáo đầy đủ.

### Sử dụng Best Model

```python
import pickle
import pandas as pd
import numpy as np

# Load model và scaler
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('models/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load feature names
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

## 🐛 TROUBLESHOOTING

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

```powershell
# Giảm n_estimators trong models
# Hoặc tăng RAM
# Hoặc sử dụng sample nhỏ hơn của data

# Trong notebook:
df_sample = df.sample(n=10000, random_state=42)
```

### Lỗi: Kernel died / crashed

```powershell
# Restart kernel trong Jupyter
# Hoặc
pip install --upgrade ipykernel
python -m ipykernel install --user --name venv
```

### Lỗi: XGBoost/LightGBM không cài được

```powershell
# Windows có thể cần Visual C++ Build Tools
# Download từ: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Hoặc sử dụng conda
conda install -c conda-forge xgboost lightgbm
```

---

## 💡 TIPS

### Tăng tốc độ chạy

1. **Sử dụng sample nhỏ hơn để test:**

```python
df = df.sample(n=10000, random_state=42)
```

2. **Giảm n_estimators:**

```python
model = RandomForestClassifier(n_estimators=50)  # thay vì 100
```

3. **Sử dụng n_jobs=-1 để parallel processing:**

```python
model = RandomForestClassifier(n_jobs=-1)
```

### Debug dễ dàng hơn

1. **Thêm print statements:**

```python
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
```

2. **Sử dụng try-except:**

```python
try:
    model.fit(X_train, y_train)
except Exception as e:
    print(f"Error: {e}")
```

3. **Check intermediate results:**

```python
# Sau mỗi bước quan trọng
df.head()
df.info()
df.describe()
```

### Best Practices

1. **Luôn save progress:**

   - Save notebooks thường xuyên (Ctrl+S)
   - Save intermediate results to CSV/pickle

2. **Comment code của bạn:**

   - Giải thích logic phức tạp
   - Note down parameters quan trọng

3. **Version control:**
   - Nếu biết Git: commit sau mỗi milestone
   - Backup thư mục dự án thường xuyên

---

## 📞 HỖ TRỢ

### Resources

- **Documentation:**

  - Scikit-learn: https://scikit-learn.org/stable/
  - XGBoost: https://xgboost.readthedocs.io/
  - Pandas: https://pandas.pydata.org/docs/

- **Communities:**
  - Stack Overflow: https://stackoverflow.com/
  - Kaggle Forums: https://www.kaggle.com/discussion
  - Reddit r/MachineLearning: https://www.reddit.com/r/MachineLearning/

### Contact

- **Sinh viên:** [Tên của bạn]
- **Email:** [Email của bạn]
- **GitHub:** [Link GitHub repository nếu có]

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
- [ ] Đã đọc báo cáo tiểu luận
- [ ] Hiểu được results và insights
- [ ] Sẵn sàng present/demo

---

**Chúc bạn thành công với dự án! 🎉**

Nếu có vấn đề, đừng ngần ngại hỏi thầy cô hoặc bạn bè. Machine Learning là journey, không phải destination!
