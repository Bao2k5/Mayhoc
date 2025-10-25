# Hướng dẫn tải Dataset

## Dataset: Default of Credit Card Clients

### Nguồn dữ liệu chính:

#### 1. UCI Machine Learning Repository (Khuyến nghị)

- **Link**: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **File**: default of credit card clients.xls
- **Kích thước**: ~2.5MB
- **Định dạng**: Excel (.xls)

**Cách tải:**

1. Truy cập link trên
2. Click vào "Data Folder"
3. Tải file "default of credit card clients.xls"
4. Đổi tên thành `UCI_Credit_Card.csv` và lưu vào thư mục này

#### 2. Kaggle (Dễ dàng hơn)

- **Link**: https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset
- **File**: UCI_Credit_Card.csv
- **Kích thước**: ~2.2MB
- **Định dạng**: CSV

**Cách tải từ Kaggle:**

Cách 1: Download trực tiếp

```
1. Truy cập link Kaggle trên
2. Click "Download" button
3. Giải nén và lưu file UCI_Credit_Card.csv vào thư mục này
```

Cách 2: Sử dụng Kaggle API (Khuyến nghị)

```bash
# Cài đặt Kaggle API
pip install kaggle

# Setup Kaggle credentials (cần tạo API key trên Kaggle)
# Đặt file kaggle.json vào ~/.kaggle/ (Linux/Mac) hoặc C:\Users\<username>\.kaggle\ (Windows)

# Tải dataset
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset

# Giải nén
unzip default-of-credit-card-clients-dataset.zip -d raw/

# Hoặc trên Windows PowerShell:
Expand-Archive default-of-credit-card-clients-dataset.zip -DestinationPath raw/
```

### Thông tin Dataset

**Số lượng mẫu**: 30,000 khách hàng
**Số lượng features**: 24 (23 features + 1 target)
**Thời gian**: Tháng 4/2005 đến Tháng 9/2005
**Nguồn**: Taiwan Credit Card Database

### Mô tả các cột:

#### Demographic Information

- `ID`: ID của khách hàng
- `LIMIT_BAL`: Hạn mức tín dụng (NT$)
- `SEX`: Giới tính (1=nam, 2=nữ)
- `EDUCATION`: Học vấn (1=cao học, 2=đại học, 3=THPT, 4=khác)
- `MARRIAGE`: Hôn nhân (1=đã kết hôn, 2=độc thân, 3=khác)
- `AGE`: Tuổi

#### Payment Status (Sep 2005 - Apr 2005)

- `PAY_0`: Trạng thái thanh toán tháng 9/2005
- `PAY_2`: Trạng thái thanh toán tháng 8/2005
- `PAY_3`: Trạng thái thanh toán tháng 7/2005
- `PAY_4`: Trạng thái thanh toán tháng 6/2005
- `PAY_5`: Trạng thái thanh toán tháng 5/2005
- `PAY_6`: Trạng thái thanh toán tháng 4/2005

**Giá trị trạng thái thanh toán:**

- -1: Thanh toán đúng hạn
- 1: Trễ 1 tháng
- 2: Trễ 2 tháng
- ...
- 8: Trễ 8 tháng trở lên

#### Bill Statement Amount

- `BILL_AMT1`: Số tiền hóa đơn tháng 9/2005 (NT$)
- `BILL_AMT2`: Số tiền hóa đơn tháng 8/2005 (NT$)
- `BILL_AMT3`: Số tiền hóa đơn tháng 7/2005 (NT$)
- `BILL_AMT4`: Số tiền hóa đơn tháng 6/2005 (NT$)
- `BILL_AMT5`: Số tiền hóa đơn tháng 5/2005 (NT$)
- `BILL_AMT6`: Số tiền hóa đơn tháng 4/2005 (NT$)

#### Previous Payment Amount

- `PAY_AMT1`: Số tiền thanh toán tháng 9/2005 (NT$)
- `PAY_AMT2`: Số tiền thanh toán tháng 8/2005 (NT$)
- `PAY_AMT3`: Số tiền thanh toán tháng 7/2005 (NT$)
- `PAY_AMT4`: Số tiền thanh toán tháng 6/2005 (NT$)
- `PAY_AMT5`: Số tiền thanh toán tháng 5/2005 (NT$)
- `PAY_AMT6`: Số tiền thanh toán tháng 4/2005 (NT$)

#### Target Variable

- `default.payment.next.month`: Vỡ nợ tháng tiếp theo (1=có, 0=không)

### Cấu trúc thư mục sau khi tải:

```
data/
├── raw/
│   └── UCI_Credit_Card.csv          # File gốc tải về
├── processed/
│   ├── data_after_eda.csv           # Sau EDA
│   ├── preprocessed_data.pkl        # Sau preprocessing
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
└── README.md                         # File này
```

### Tham khảo:

**Paper gốc:**
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. Expert Systems with Applications, 36(2), 2473-2480.

**DOI**: 10.1016/j.eswa.2007.12.020

### Lưu ý:

- File Excel từ UCI có thể cần convert sang CSV
- Kaggle version đã ở định dạng CSV, dễ sử dụng hơn
- Đảm bảo encoding là UTF-8 khi đọc CSV

### Troubleshooting:

**Lỗi khi đọc file Excel:**

```python
# Dùng pandas để convert
import pandas as pd
df = pd.read_excel('default of credit card clients.xls', header=1)
df.to_csv('UCI_Credit_Card.csv', index=False)
```

**Lỗi encoding:**

```python
# Thử encoding khác
df = pd.read_csv('UCI_Credit_Card.csv', encoding='latin-1')
# hoặc
df = pd.read_csv('UCI_Credit_Card.csv', encoding='cp1252')
```
