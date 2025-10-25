# DỰ ĐOÁN VỠ NỢ THẺ TÍN DỤNG

## Credit Card Default Prediction

---

## 📋 TỔNG QUAN DỰ ÁN

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

## 🌍 BỐI CẢNH THỊ TRƯỜNG

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

## 📊 DỮ LIỆU

### Nguồn dữ liệu

Dự án sử dụng dataset **"Default of Credit Card Clients"** từ UCI Machine Learning Repository, bao gồm:

- **30,000 khách hàng** tại Đài Loan
- **24 đặc trúng** (features)
- Dữ liệu từ **tháng 4/2005 đến tháng 9/2005**

### Các đặc trưng chính

#### 1. Thông tin nhân khẩu học

- `LIMIT_BAL`: Hạn mức tín dụng (NT$)
- `SEX`: Giới tính (1=Nam, 2=Nữ)
- `EDUCATION`: Trình độ học vấn (1=Cao học, 2=Đại học, 3=THPT, 4=Khác)
- `MARRIAGE`: Tình trạng hôn nhân (1=Đã kết hôn, 2=Độc thân, 3=Khác)
- `AGE`: Tuổi

#### 2. Lịch sử thanh toán (PAY_0 đến PAY_6)

Trạng thái thanh toán từ tháng 4-9/2005:

- `-1`: Thanh toán đúng hạn
- `1`: Trễ 1 tháng
- `2`: Trễ 2 tháng
- ...
- `8`: Trễ 8 tháng trở lên

#### 3. Số tiền hóa đơn (BILL_AMT1 đến BILL_AMT6)

Số tiền hóa đơn từ tháng 4-9/2005 (NT$)

#### 4. Số tiền thanh toán (PAY_AMT1 đến PAY_AMT6)

Số tiền đã thanh toán từ tháng 4-9/2005 (NT$)

#### 5. Biến mục tiêu

- `default.payment.next.month`: Vỡ nợ tháng tiếp theo (1=Có, 0=Không)

---

## 🛠️ PHƯƠNG PHÁP LUẬN

### 1. Khám phá dữ liệu (EDA)

- Phân tích phân phối các biến
- Kiểm tra dữ liệu thiếu và outliers
- Phân tích tương quan giữa các đặc trưng
- Trực quan hóa mối quan hệ với biến mục tiêu

### 2. Tiền xử lý dữ liệu

- Xử lý missing values
- Xử lý outliers
- Feature Engineering:
  - Tạo biến về tỷ lệ sử dụng tín dụng
  - Tạo biến về xu hướng thanh toán
  - Tạo biến về mức độ trễ hạn trung bình
- Chuẩn hóa/Standardization
- Xử lý class imbalance (SMOTE, undersampling)

### 3. Xây dựng mô hình

#### 3.1. Baseline Models

- **Logistic Regression**: Mô hình tuyến tính cơ bản
- **Decision Tree**: Mô hình cây quyết định

#### 3.2. Advanced Models

- **Random Forest**: Ensemble learning với bagging
- **Gradient Boosting (XGBoost/LightGBM)**: Ensemble learning với boosting
- **Support Vector Machine (SVM)**: Phân loại với kernel
- **Neural Network**: Deep learning approach

#### 3.3. Hyperparameter Tuning

- Grid Search CV
- Random Search CV
- Bayesian Optimization

### 4. Đánh giá mô hình

#### Metrics chính

- **Accuracy**: Độ chính xác tổng thể
- **Precision**: Tỷ lệ dự đoán đúng trong các trường hợp dự đoán vỡ nợ
- **Recall**: Tỷ lệ phát hiện được các trường hợp vỡ nợ thực tế
- **F1-Score**: Trung bình điều hòa của Precision và Recall
- **ROC-AUC**: Diện tích dưới đường cong ROC
- **Confusion Matrix**: Ma trận nhầm lẫn
- **Cost-Benefit Analysis**: Phân tích chi phí-lợi ích

#### Business Metrics

- **False Positive Cost**: Chi phí khi từ chối khách hàng tốt
- **False Negative Cost**: Chi phí khi chấp nhận khách hàng xấu
- **Total Cost**: Tổng chi phí dự kiến

---

## 📁 CẤU TRÚC THỨ MỤC

```
Credit_Card_Default_Prediction/
│
├── data/                              # Dữ liệu
│   ├── raw/                           # Dữ liệu gốc
│   ├── processed/                     # Dữ liệu đã xử lý
│   └── README.md                      # Hướng dẫn tải dữ liệu
│
├── notebooks/                         # Jupyter Notebooks
│   ├── 01_EDA.ipynb                  # Khám phá dữ liệu
│   ├── 02_Data_Preprocessing.ipynb   # Tiền xử lý
│   ├── 03_Model_Training.ipynb       # Huấn luyện mô hình
│   └── 04_Model_Evaluation.ipynb     # Đánh giá và so sánh
│
├── src/                               # Source code
│   ├── data_processing.py            # Module xử lý dữ liệu
│   ├── feature_engineering.py        # Module tạo đặc trưng
│   ├── model_training.py             # Module huấn luyện
│   └── model_evaluation.py           # Module đánh giá
│
├── models/                            # Mô hình đã lưu
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── best_model.pkl
│
├── reports/                           # Báo cáo
│   ├── figures/                      # Hình ảnh, biểu đồ
│   ├── Tieu_Luan.pdf                # Tiểu luận hoàn chỉnh
│   └── Presentation.pptx             # Slide thuyết trình
│
├── requirements.txt                   # Thư viện cần thiết
└── README.md                         # File này
```

---

## 🚀 HƯỚNG DẪN SỬ DỤNG

### 1. Cài đặt môi trường

```bash
# Clone hoặc tải dự án
cd Credit_Card_Default_Prediction

# Tạo virtual environment (khuyến nghị)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Cài đặt thư viện
pip install -r requirements.txt
```

### 2. Tải dữ liệu

Tải dataset từ UCI Repository:

- Link: https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- Đặt file vào thư mục `data/raw/`

Hoặc sử dụng Kaggle:

```bash
kaggle datasets download -d uciml/default-of-credit-card-clients-dataset
```

### 3. Chạy các notebook

Theo thứ tự:

1. `01_EDA.ipynb` - Khám phá và hiểu dữ liệu
2. `02_Data_Preprocessing.ipynb` - Tiền xử lý và feature engineering
3. `03_Model_Training.ipynb` - Huấn luyện các mô hình
4. `04_Model_Evaluation.ipynb` - Đánh giá và chọn mô hình tốt nhất

---

## 📈 KẾT QUẢ DỰ KIẾN

### Performance Metrics

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | ~0.82    | ~0.65     | ~0.45  | ~0.53    | ~0.77   |
| Random Forest       | ~0.82    | ~0.68     | ~0.48  | ~0.56    | ~0.78   |
| XGBoost             | ~0.83    | ~0.70     | ~0.50  | ~0.58    | ~0.80   |
| Neural Network      | ~0.82    | ~0.67     | ~0.47  | ~0.55    | ~0.79   |

### Insights quan trọng

1. **Lịch sử thanh toán** là yếu tố quan trọng nhất
2. **Hạn mức tín dụng** và tỷ lệ sử dụng ảnh hưởng lớn
3. **Tuổi** và **trình độ học vấn** có tương quan với khả năng vỡ nợ
4. **Xu hướng thanh toán** trong 6 tháng gần nhất rất có giá trị

---

## 💡 ỨNG DỤNG THỰC TẾ

### Cho Ngân hàng Việt Nam

#### 1. Screening khách hàng mới

- Đánh giá tự động hồ sơ xin cấp thẻ
- Giảm thời gian phê duyệt từ 7-10 ngày xuống 1-2 ngày
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

## 📚 TÀI LIỆU THAM KHẢO

### Papers & Research

1. Yeh, I. C., & Lien, C. H. (2009). "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients"
2. Baesens, B., et al. (2003). "Benchmarking state-of-the-art classification algorithms for credit scoring"
3. Hand, D. J., & Henley, W. E. (1997). "Statistical classification methods in consumer credit scoring: a review"

### Ngân hàng Việt Nam

- NHNN: Báo cáo phát triển thị trường thẻ Việt Nam
- Vietcombank: Credit Risk Management Framework
- Techcombank: Digital Banking & AI Applications
- VPBank: Fintech Innovation in Vietnam

### International Best Practices

- Federal Reserve: Credit Card Market Research
- FICO Score Methodology
- Basel III: Credit Risk Management Guidelines
- PSD2: Open Banking Standards (EU)

### Tools & Libraries

- Scikit-learn Documentation
- XGBoost Documentation
- Imbalanced-learn for handling class imbalance
- SHAP for model interpretability

---

## 👥 NHÓM THỰC HIỆN

- **Sinh viên**: [Tên của bạn]
- **MSSV**: [Mã số sinh viên]
- **Lớp**: [Tên lớp]
- **Giảng viên hướng dẫn**: [Tên giảng viên]
- **Học kỳ**: [Học kỳ/Năm học]

---

## 📞 LIÊN HỆ

- **Email**: [email của bạn]
- **GitHub**: [link github]
- **LinkedIn**: [link linkedin]

---

## 📝 LICENSE

Dự án này được phát triển cho mục đích học tập và nghiên cứu.

---

## 🙏 LỜI CẢM ƠN

- UCI Machine Learning Repository - Cung cấp dataset
- Kaggle Community - Các insights và discussions
- Scikit-learn & XGBoost Teams - Các thư viện ML tuyệt vời
- Giảng viên và bạn bè - Hỗ trợ và góp ý

---

**Last Updated**: October 2025
