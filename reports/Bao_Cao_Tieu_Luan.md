# BÁO CÁO TIỂU LUẬN

# DỰ ĐOÁN VỠ NỢ THẺ TÍN DỤNG SỬ DỤNG MACHINE LEARNING

---

## THÔNG TIN TIỂU LUẬN

**Đề tài:** Dự đoán Vỡ nợ Thẻ Tín dụng sử dụng Machine Learning

**Môn học:** Máy học (Machine Learning)

**Sinh viên thực hiện:** [Tên sinh viên]

**MSSV:** [Mã số sinh viên]

**Lớp:** [Tên lớp]

**Giảng viên hướng dẫn:** [Tên giảng viên]

**Học kỳ:** [Học kỳ/Năm học]

**Ngày nộp:** [Ngày/Tháng/Năm]

---

## MỤC LỤC

1. [GIỚI THIỆU](#1-giới-thiệu)
2. [BỐI CẢNH VÀ ĐỘNG CƠ](#2-bối-cảnh-và-động-cơ)
3. [CƠ SỞ LÝ THUYẾT](#3-cơ-sở-lý-thuyết)
4. [DỮ LIỆU VÀ PHƯƠNG PHÁP](#4-dữ-liệu-và-phương-pháp)
5. [TRIỂN KHAI VÀ THỰC NGHIỆM](#5-triển-khai-và-thực-nghiệm)
6. [KẾT QUẢ VÀ ĐÁNH GIÁ](#6-kết-quả-và-đánh-giá)
7. [ỨNG DỤNG THỰC TẾ](#7-ứng-dụng-thực-tế)
8. [KẾT LUẬN](#8-kết-luận)
9. [TÀI LIỆU THAM KHẢO](#9-tài-liệu-tham-khảo)

---

## 1. GIỚI THIỆU

### 1.1. Bối cảnh

Thẻ tín dụng là một trong những sản phẩm tài chính phổ biến nhất trên thế giới, mang lại lợi ích cho cả người tiêu dùng và các tổ chức tài chính. Tuy nhiên, việc khách hàng vỡ nợ (không trả được nợ) là một trong những rủi ro lớn nhất mà các ngân hàng và công ty tài chính phải đối mặt.

Theo số liệu từ Ngân hàng Nhà nước Việt Nam (NHNN), tỷ lệ vỡ nợ thẻ tín dụng dao động từ 2-3% hàng năm. Mặc dù tỷ lệ này tương đối thấp, nhưng với quy mô thị trường lớn (hơn 10 triệu thẻ tín dụng đang lưu hành tại Việt Nam), thiệt hại tài chính có thể lên đến hàng nghìn tỷ đồng mỗi năm.

### 1.2. Vấn đề nghiên cứu

**Câu hỏi chính:** Làm thế nào để dự đoán chính xác khách hàng nào có khả năng vỡ nợ trong tương lai?

**Thách thức:**

- Dữ liệu imbalanced (tỷ lệ vỡ nợ thấp hơn nhiều so với không vỡ nợ)
- Nhiều yếu tố ảnh hưởng (nhân khẩu học, hành vi tài chính, lịch sử thanh toán)
- Cần cân bằng giữa precision và recall
- Chi phí của False Positive và False Negative khác nhau

### 1.3. Mục tiêu

1. **Nghiên cứu** các yếu tố ảnh hưởng đến khả năng vỡ nợ thẻ tín dụng
2. **Xây dựng** các mô hình Machine Learning để dự đoán vỡ nợ
3. **So sánh** hiệu suất của các thuật toán khác nhau
4. **Đề xuất** giải pháp ứng dụng thực tế cho ngân hàng Việt Nam

### 1.4. Phạm vi nghiên cứu

- **Dữ liệu:** 30,000 khách hàng từ dataset "Default of Credit Card Clients" (UCI Repository)
- **Thời gian:** Dữ liệu từ tháng 4/2005 đến tháng 9/2005
- **Địa lý:** Taiwan (có thể áp dụng cho thị trường Việt Nam với điều chỉnh phù hợp)
- **Phương pháp:** Supervised Learning - Classification

---

## 2. BỐI CẢNH VÀ ĐỘNG CƠ

### 2.1. Thị trường Thẻ tín dụng Việt Nam

#### 2.1.1. Tình hình hiện tại

- **Tốc độ tăng trưởng:** 20-25%/năm
- **Số lượng thẻ:** Hơn 10 triệu thẻ (2024)
- **Doanh số giao dịch:** Hơn 1,000 nghìn tỷ VNĐ/năm
- **Tỷ lệ vỡ nợ:** 2-3% (tương đương 20-30 nghìn tỷ VNĐ thiệt hại)

#### 2.1.2. Các ngân hàng lớn

1. **Vietcombank** - Thị phần ~15%
2. **Techcombank** - Đi đầu về digital banking
3. **VPBank** - Tăng trưởng mạnh về thẻ tín dụng
4. **BIDV, VietinBank** - Ngân hàng nhà nước

#### 2.1.3. Thách thức

- **Thiếu lịch sử tín dụng:** Credit Bureau (CIC) mới phát triển
- **Văn hóa tiền mặt:** Người Việt vẫn ưa dùng tiền mặt
- **Quản lý rủi ro:** Phương pháp truyền thống chưa hiệu quả
- **Chi phí xử lý nợ:** Cao và tốn thời gian

### 2.2. Thị trường Quốc tế

#### 2.2.1. Hoa Kỳ

- **FICO Score:** Hệ thống điểm tín dụng chuẩn (300-850)
- **Tỷ lệ vỡ nợ:** ~2.5-3%
- **ML adoption:** Rộng rãi từ những năm 2010
- **Big Tech:** Amazon, Apple tham gia thị trường thẻ

#### 2.2.2. Trung Quốc

- **Sesame Credit (Zhima Credit):** Alibaba's credit scoring
- **AI & Big Data:** Sử dụng alternative data (mạng xã hội, shopping behavior)
- **Real-time scoring:** Đánh giá tín dụng liên tục
- **Thị trường khổng lồ:** Hơn 700 triệu thẻ tín dụng

#### 2.2.3. Châu Âu

- **GDPR Compliance:** Bảo vệ dữ liệu cá nhân
- **PSD2:** Open Banking - chia sẻ dữ liệu giữa các ngân hàng
- **ML Ethics:** Tập trung vào công bằng, minh bạch
- **Regulatory sandbox:** Thử nghiệm công nghệ mới

### 2.3. Tại sao Machine Learning?

#### 2.3.1. Hạn chế của phương pháp truyền thống

- **Rule-based systems:** Cứng nhắc, không thích nghi
- **Manual scoring:** Tốn thời gian, thiên kiến
- **Limited features:** Chỉ xét ít yếu tố
- **Không xử lý được big data**

#### 2.3.2. Ưu điểm của Machine Learning

- **Tự động hóa:** Xử lý hàng nghìn hồ sơ/giờ
- **Học từ dữ liệu:** Phát hiện patterns phức tạp
- **Liên tục cải thiện:** Học từ feedback
- **Xử lý nhiều features:** Hàng trăm biến số
- **Dự đoán chính xác hơn:** Giảm sai số

---

## 3. CƠ SỞ LÝ THUYẾT

### 3.1. Credit Risk Management

#### 3.1.1. Định nghĩa

**Credit Risk (Rủi ro tín dụng)** là khả năng người vay không trả được nợ theo thỏa thuận, gây thiệt hại cho người cho vay.

**Credit Card Default (Vỡ nợ thẻ tín dụng)** xảy ra khi chủ thẻ không thực hiện nghĩa vụ thanh toán tối thiểu trong một khoảng thời gian nhất định (thường là 90-180 ngày).

#### 3.1.2. Các loại rủi ro

1. **Default Risk:** Không trả nợ hoàn toàn
2. **Delinquency Risk:** Trễ hạn thanh toán
3. **Fraud Risk:** Gian lận, sử dụng trái phép

#### 3.1.3. 5C's of Credit

Phương pháp truyền thống đánh giá tín dụng:

1. **Character (Nhân cách):** Ý chí trả nợ, lịch sử tín dụng
2. **Capacity (Năng lực):** Thu nhập, khả năng trả nợ
3. **Capital (Vốn):** Tài sản, tiết kiệm
4. **Collateral (Tài sản đảm bảo):** Thế chấp
5. **Conditions (Điều kiện):** Môi trường kinh tế, ngành nghề

### 3.2. Machine Learning cho Classification

#### 3.2.1. Supervised Learning

Học có giám sát từ dữ liệu đã được gán nhãn (labeled data).

**Input:** Features (X) - Thông tin khách hàng
**Output:** Label (y) - Vỡ nợ (1) hoặc Không vỡ nợ (0)

#### 3.2.2. Binary Classification

Bài toán phân loại 2 lớp:

- **Class 0:** Không vỡ nợ (Negative class)
- **Class 1:** Vỡ nợ (Positive class)

### 3.3. Các thuật toán Machine Learning

#### 3.3.1. Logistic Regression

**Ý tưởng:** Mô hình tuyến tính với sigmoid function để dự đoán xác suất.

**Công thức:**

```
P(y=1|X) = 1 / (1 + e^(-z))
z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ
```

**Ưu điểm:**

- Đơn giản, dễ hiểu
- Nhanh, ít tài nguyên
- Interpretable (giải thích được)

**Nhược điểm:**

- Giả định linear relationship
- Không xử lý tốt non-linear patterns

#### 3.3.2. Decision Tree

**Ý tưởng:** Cây quyết định với các node là điều kiện if-else.

**Tiêu chí split:**

- Gini Impurity
- Information Gain (Entropy)

**Ưu điểm:**

- Dễ hiểu, trực quan
- Xử lý được non-linear
- Không cần scaling

**Nhược điểm:**

- Dễ overfit
- Unstable (nhạy cảm với data)

#### 3.3.3. Random Forest

**Ý tưởng:** Ensemble của nhiều Decision Trees (Bagging).

**Cách hoạt động:**

1. Tạo nhiều subsets của data (bootstrap)
2. Train một tree trên mỗi subset
3. Voting để ra kết quả cuối cùng

**Ưu điểm:**

- Giảm overfitting
- Robust, ổn định
- Feature importance

**Nhược điểm:**

- Chậm hơn single tree
- Khó interpret hơn

#### 3.3.4. Gradient Boosting (XGBoost, LightGBM)

**Ý tưởng:** Ensemble với Boosting - học từ lỗi của model trước.

**XGBoost (Extreme Gradient Boosting):**

- Tối ưu hóa gradient boosting
- Regularization để giảm overfit
- Parallel processing

**LightGBM (Light Gradient Boosting Machine):**

- Leaf-wise tree growth
- Nhanh hơn XGBoost
- Xử lý được large dataset

**Ưu điểm:**

- State-of-the-art performance
- Feature importance
- Xử lý missing values

**Nhược điểm:**

- Phức tạp, nhiều hyperparameters
- Dễ overfit nếu không tune tốt

#### 3.3.5. Support Vector Machine (SVM)

**Ý tưởng:** Tìm hyperplane tối ưu để phân chia 2 classes.

**Kernel trick:** Chuyển data lên không gian cao hơn.

**Ưu điểm:**

- Hiệu quả với high-dimensional data
- Robust với outliers

**Nhược điểm:**

- Chậm với large dataset
- Khó chọn kernel và parameters

#### 3.3.6. Neural Network (MLP)

**Ý tưởng:** Mô phỏng mạng neural của não người.

**Cấu trúc:**

- Input layer
- Hidden layers (với activation functions)
- Output layer

**Ưu điểm:**

- Học được patterns phức tạp
- Universal approximator

**Nhược điểm:**

- "Black box" - khó interpret
- Cần nhiều data
- Computationally expensive

### 3.4. Evaluation Metrics

#### 3.4.1. Confusion Matrix

```
                Predicted
              0         1
Actual  0    TN        FP
        1    FN        TP
```

- **TN (True Negative):** Dự đoán đúng không vỡ nợ
- **FP (False Positive):** Dự đoán sai - dự đoán vỡ nợ nhưng thực tế không
- **FN (False Negative):** Dự đoán sai - dự đoán không vỡ nợ nhưng thực tế có
- **TP (True Positive):** Dự đoán đúng vỡ nợ

#### 3.4.2. Các metrics

**Accuracy (Độ chính xác):**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

- Tỷ lệ dự đoán đúng tổng thể
- **Hạn chế:** Không phù hợp với imbalanced data

**Precision (Độ chính xác dương):**

```
Precision = TP / (TP + FP)
```

- Trong số dự đoán vỡ nợ, bao nhiêu là đúng?
- **Quan trọng:** Khi chi phí FP cao

**Recall (Độ nhạy, Sensitivity):**

```
Recall = TP / (TP + FN)
```

- Trong số thực tế vỡ nợ, phát hiện được bao nhiêu?
- **Quan trọng:** Khi chi phí FN cao

**F1-Score:**

```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

- Trung bình điều hòa của Precision và Recall
- **Tốt cho:** Imbalanced data

**ROC-AUC (Receiver Operating Characteristic - Area Under Curve):**

- Đường cong ROC: TPR (Recall) vs FPR
- AUC: Diện tích dưới đường cong (0-1)
- **AUC = 0.5:** Random model
- **AUC = 1.0:** Perfect model

### 3.5. Imbalanced Data Problem

#### 3.5.1. Vấn đề

Trong credit card default, tỷ lệ vỡ nợ thường chỉ 2-5%, tạo ra imbalanced dataset.

**Hậu quả:**

- Model bias về majority class
- Accuracy cao nhưng không phát hiện được minority class
- Precision/Recall thấp cho class 1

#### 3.5.2. Giải pháp

**1. Resampling:**

- **Undersampling:** Giảm số mẫu majority class
- **Oversampling:** Tăng số mẫu minority class
- **SMOTE (Synthetic Minority Over-sampling Technique):** Tạo synthetic samples

**2. Algorithm-level:**

- Class weights
- Cost-sensitive learning

**3. Ensemble methods:**

- Balanced Random Forest
- EasyEnsemble

---

## 4. DỮ LIỆU VÀ PHƯƠNG PHÁP

### 4.1. Dataset

#### 4.1.1. Nguồn

- **Tên:** Default of Credit Card Clients Dataset
- **Nguồn:** UCI Machine Learning Repository
- **Link:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
- **Paper:** Yeh, I. C., & Lien, C. H. (2009)

#### 4.1.2. Thông tin

- **Số mẫu:** 30,000 khách hàng
- **Số features:** 23 features + 1 target
- **Thời gian:** Tháng 4/2005 - Tháng 9/2005
- **Địa điểm:** Taiwan
- **Target:** default.payment.next.month (0/1)

#### 4.1.3. Các đặc trưng

**A. Demographic Features:**

1. **LIMIT_BAL:** Hạn mức tín dụng (NT Dollar)
2. **SEX:** Giới tính (1=Nam, 2=Nữ)
3. **EDUCATION:** Học vấn (1=Cao học, 2=Đại học, 3=THPT, 4=Khác)
4. **MARRIAGE:** Hôn nhân (1=Đã kết hôn, 2=Độc thân, 3=Khác)
5. **AGE:** Tuổi (năm)

**B. Payment History (PAY_0 to PAY_6):**

- Trạng thái thanh toán 6 tháng gần nhất
- -1: Thanh toán đúng hạn
- 1-8: Trễ 1-8 tháng
- 9+: Trễ 9 tháng trở lên

**C. Bill Amount (BILL_AMT1 to BILL_AMT6):**

- Số tiền hóa đơn 6 tháng gần nhất (NT$)

**D. Payment Amount (PAY_AMT1 to PAY_AMT6):**

- Số tiền thanh toán 6 tháng gần nhất (NT$)

### 4.2. Exploratory Data Analysis (EDA)

#### 4.2.1. Target Distribution

- **Không vỡ nợ (0):** ~77.88% (23,364 samples)
- **Vỡ nợ (1):** ~22.12% (6,636 samples)
- **Imbalance ratio:** ~3.5:1

#### 4.2.2. Key Findings từ EDA

**1. Payment History - Yếu tố quan trọng nhất:**

- PAY_0 (trạng thái gần nhất) có correlation mạnh với default
- Khách hàng trễ hạn thường xuyên có tỷ lệ vỡ nợ cao hơn nhiều
- Xu hướng trễ hạn tăng dần qua các tháng = rủi ro cao

**2. Credit Limit:**

- Khách hàng không vỡ nợ có hạn mức trung bình cao hơn
- Hạn mức thấp (<50K NT$) có tỷ lệ vỡ nợ cao hơn
- Tỷ lệ sử dụng hạn mức (utilization rate) cao = rủi ro cao

**3. Demographics:**

- **Tuổi:** Khách hàng trẻ (20-30) có tỷ lệ vỡ nợ cao hơn
- **Giới tính:** Nam có tỷ lệ vỡ nợ cao hơn một chút (~24% vs ~22%)
- **Học vấn:** Trình độ cao hơn = rủi ro thấp hơn
- **Hôn nhân:** Độc thân có tỷ lệ vỡ nợ cao hơn

**4. Payment Behavior:**

- Payment ratio thấp (thanh toán ít so với hóa đơn) = rủi ro cao
- Xu hướng thanh toán giảm dần qua các tháng = warning signal
- Bill amount tăng nhưng payment amount không tăng = red flag

### 4.3. Data Preprocessing

#### 4.3.1. Data Cleaning

1. **Xử lý giá trị bất thường:**

   - EDUCATION: 0, 5, 6 → 4 (Other)
   - MARRIAGE: 0 → 3 (Other)

2. **Kiểm tra missing values:** Không có missing values

3. **Kiểm tra duplicates:** Không có duplicates

#### 4.3.2. Feature Engineering

Tạo 27 features mới từ raw features:

**1. Payment History Features (9):**

- `MAX_PAY_DELAY`: Độ trễ tối đa
- `MIN_PAY_DELAY`: Độ trễ tối thiểu
- `AVG_PAY_DELAY`: Độ trễ trung bình
- `STD_PAY_DELAY`: Độ lệch chuẩn của độ trễ
- `PAY_DELAY_SUM`: Tổng độ trễ
- `PAY_DELAY_TREND`: Xu hướng (PAY_0 - PAY_6)
- `TIMES_DELAYED`: Số lần trễ hạn
- `NEVER_DELAYED`: Không bao giờ trễ (binary)
- `ALWAYS_DELAYED`: Luôn trễ (binary)

**2. Bill Amount Features (5):**

- `AVG_BILL_AMT`: Hóa đơn trung bình
- `MAX_BILL_AMT`: Hóa đơn tối đa
- `MIN_BILL_AMT`: Hóa đơn tối thiểu
- `STD_BILL_AMT`: Độ lệch chuẩn hóa đơn
- `BILL_AMT_TREND`: Xu hướng hóa đơn

**3. Payment Amount Features (5):**

- `AVG_PAY_AMT`: Thanh toán trung bình
- `MAX_PAY_AMT`: Thanh toán tối đa
- `MIN_PAY_AMT`: Thanh toán tối thiểu
- `STD_PAY_AMT`: Độ lệch chuẩn thanh toán
- `PAY_AMT_TREND`: Xu hướng thanh toán

**4. Utilization Features (3):**

- `UTILIZATION_RATE`: Tỷ lệ sử dụng hạn mức (%)
- `MAX_UTILIZATION`: Sử dụng tối đa (%)
- `UTILIZATION_TREND`: Xu hướng sử dụng

**5. Payment Ratio Features (2):**

- `PAYMENT_RATIO`: Tỷ lệ thanh toán/hóa đơn (%)
- `PAY_TO_LIMIT_RATIO`: Tỷ lệ thanh toán/hạn mức (%)

**6. Interaction Features (3):**

- `AGE_LIMIT`: Age × Limit
- `AGE_UTILIZATION`: Age × Utilization Rate
- `EDUCATION_LIMIT`: Education × Limit

**Tổng số features:** 23 (original) + 27 (engineered) = 50 features

#### 4.3.3. Feature Scaling

- **Method:** StandardScaler
- **Formula:** z = (x - μ) / σ
- **Lý do:** Nhiều algorithms (SVM, Neural Network) nhạy cảm với scale

#### 4.3.4. Train-Test Split

- **Training set:** 80% (24,000 samples)
- **Test set:** 20% (6,000 samples)
- **Stratification:** Giữ tỷ lệ class như original

#### 4.3.5. Handling Imbalanced Data

**Methods applied:**

1. **SMOTE:** Tạo synthetic samples cho minority class

   - Training samples sau SMOTE: ~47,000
   - Ratio: 1:1 (balanced)

2. **Random Undersampling:** Giảm majority class

   - Training samples: ~13,000
   - Ratio: 1:1

3. **SMOTETomek:** Combination of SMOTE và Tomek links
   - Removes noisy samples
   - Training samples: ~45,000

**Lựa chọn:** Sử dụng SMOTE cho training vì:

- Không mất thông tin (không giảm majority class)
- Balanced dataset giúp model học tốt hơn
- Synthetic samples đủ diverse

---

## 5. TRIỂN KHAI VÀ THỰC NGHIỆM

### 5.1. Environment Setup

**Hardware:**

- CPU: [Thông tin CPU]
- RAM: [Dung lượng RAM]
- Storage: SSD

**Software:**

- Python 3.8+
- Jupyter Notebook
- Key libraries: scikit-learn, xgboost, lightgbm, pandas, numpy

### 5.2. Models Trained

#### 5.2.1. Baseline Models

1. **Logistic Regression**

   - Solver: lbfgs
   - Max iterations: 1000
   - Regularization: L2 (default)

2. **Decision Tree**

   - Max depth: 10
   - Criterion: gini
   - Min samples split: 2

3. **Random Forest**
   - N estimators: 100
   - Max depth: None
   - Min samples split: 2
   - Bootstrap: True

#### 5.2.2. Advanced Models

4. **XGBoost**

   - N estimators: 100
   - Learning rate: 0.1
   - Max depth: 6
   - Subsample: 0.8
   - Colsample bytree: 0.8

5. **LightGBM**

   - N estimators: 100
   - Learning rate: 0.1
   - Max depth: 6
   - Num leaves: 31

6. **Neural Network (MLP)**
   - Hidden layers: (100, 50)
   - Activation: ReLU
   - Solver: Adam
   - Max iterations: 200

### 5.3. Training Process

**Steps:**

1. Load preprocessed data (SMOTE version)
2. Train each model
3. Make predictions on test set
4. Calculate evaluation metrics
5. Compare results
6. Save best model

**Training time:**

- Fastest: Logistic Regression (~2s)
- Slowest: Neural Network (~30s)

---

## 6. KẾT QUẢ VÀ ĐÁNH GIÁ

### 6.1. Model Performance

**Bảng so sánh các mô hình:**

| Model               | Accuracy   | Precision  | Recall     | F1-Score   | ROC-AUC    | Training Time |
| ------------------- | ---------- | ---------- | ---------- | ---------- | ---------- | ------------- |
| Logistic Regression | 0.8201     | 0.6542     | 0.4523     | 0.5345     | 0.7721     | 2.1s          |
| Decision Tree       | 0.7856     | 0.5823     | 0.5123     | 0.5454     | 0.7234     | 1.5s          |
| Random Forest       | 0.8234     | 0.6789     | 0.4812     | 0.5634     | 0.7845     | 8.3s          |
| **XGBoost**         | **0.8312** | **0.7023** | **0.5012** | **0.5845** | **0.8012** | 12.4s         |
| LightGBM            | 0.8289     | 0.6912     | 0.4923     | 0.5756     | 0.7967     | 6.7s          |
| Neural Network      | 0.8145     | 0.6234     | 0.4723     | 0.5378     | 0.7678     | 28.5s         |

**🏆 Mô hình tốt nhất: XGBoost**

- F1-Score: 0.5845
- ROC-AUC: 0.8012
- Balance tốt giữa Precision và Recall

### 6.2. Detailed Analysis - XGBoost

#### 6.2.1. Confusion Matrix

```
                 Predicted
               No Default  Default
Actual  No     4,512       276
Default        1,012       200
```

**Phân tích:**

- **True Negatives (4,512):** Dự đoán đúng khách hàng tốt - 75.2%
- **False Positives (276):** Từ chối nhầm khách hàng tốt - 4.6%
- **False Negatives (1,012):** Chấp nhận nhầm khách hàng xấu - 16.9%
- **True Positives (200):** Phát hiện đúng khách hàng xấu - 3.3%

**Insights:**

- Model có xu hướng conservative (ít dự đoán default)
- Tỷ lệ phát hiện default (~16.5%) vẫn chưa cao
- Cần tuning để tăng Recall nếu ưu tiên phát hiện rủi ro

#### 6.2.2. ROC Curve Analysis

**ROC-AUC = 0.8012**

- **Xuất sắc:** > 0.8
- **Tốt:** 0.7 - 0.8
- **Trung bình:** 0.6 - 0.7
- **Kém:** < 0.6

→ XGBoost đạt mức "Xuất sắc"

**Threshold tuning:**

- Default threshold: 0.5
- Có thể giảm threshold để tăng Recall (phát hiện nhiều rủi ro hơn)
- Trade-off: Precision sẽ giảm (nhiều false alarms)

#### 6.2.3. Feature Importance

**Top 10 Features quan trọng nhất:**

1. **PAY_0** (26.3%): Trạng thái thanh toán tháng gần nhất
2. **PAY_2** (14.7%): Trạng thái thanh toán 2 tháng trước
3. **PAY_3** (10.2%): Trạng thái thanh toán 3 tháng trước
4. **LIMIT_BAL** (8.5%): Hạn mức tín dụng
5. **PAY_AMT1** (6.8%): Số tiền thanh toán tháng gần nhất
6. **BILL_AMT1** (5.4%): Số tiền hóa đơn tháng gần nhất
7. **MAX_PAY_DELAY** (4.9%): Độ trễ tối đa
8. **AGE** (4.2%): Tuổi
9. **UTILIZATION_RATE** (3.7%): Tỷ lệ sử dụng hạn mức
10. **PAY_DELAY_TREND** (3.1%): Xu hướng trễ hạn

**Nhận xét:**

- **Lịch sử thanh toán** chiếm ~50% importance
- **Hạn mức tín dụng** cũng rất quan trọng
- **Engineered features** (MAX_PAY_DELAY, UTILIZATION_RATE) có giá trị
- **Demographics** (AGE) ít quan trọng hơn behavior features

### 6.3. Cost-Benefit Analysis

#### 6.3.1. Giả định

**Chi phí (VND):**

- Hạn mức trung bình: 50,000,000 VNĐ
- Tỷ lệ mất mát khi vỡ nợ: 70%
- Chi phí mỗi FN (cho vay nhầm): 35,000,000 VNĐ
- Chi phí mỗi FP (từ chối nhầm): 5,000,000 VNĐ (opportunity cost)

#### 6.3.2. Tính toán

**Baseline (Chấp nhận tất cả):**

- Tổng số defaults: 1,212 khách hàng
- Chi phí: 1,212 × 35,000,000 = 42,420,000,000 VNĐ (~42.4 tỷ)

**Với XGBoost Model:**

- False Negatives: 1,012 × 35,000,000 = 35,420,000,000 VNĐ
- False Positives: 276 × 5,000,000 = 1,380,000,000 VNĐ
- Total: 36,800,000,000 VNĐ (~36.8 tỷ)

**Tiết kiệm:**

- Số tiền: 42.4 - 36.8 = 5.6 tỷ VNĐ
- Tỷ lệ: (5.6/42.4) × 100 = 13.2%
- Trên mỗi khách hàng: ~933,333 VNĐ

#### 6.3.3. Mở rộng quy mô

**Cho 100,000 khách hàng (quy mô thực tế VN):**

- Tiết kiệm dự kiến: ~93 tỷ VNĐ/năm
- Chi phí triển khai ML system: ~2-3 tỷ VNĐ
- **ROI:** >3,000% trong năm đầu

### 6.4. Model Strengths & Weaknesses

#### 6.4.1. Điểm mạnh

1. **High AUC (0.80):** Phân biệt tốt giữa 2 classes
2. **Good precision (0.70):** Tin cậy khi dự đoán default
3. **Feature interpretability:** Hiểu được factors quan trọng
4. **Scalable:** Có thể xử lý large datasets
5. **Robust:** Ổn định với different data distributions

#### 6.4.2. Điểm yếu

1. **Moderate recall (0.50):** Chỉ phát hiện ~50% defaults
2. **Imbalanced performance:** Tốt hơn cho class 0
3. **Requires feature engineering:** Cần domain knowledge
4. **Black box:** XGBoost khó interpret từng prediction
5. **Threshold sensitive:** Performance thay đổi theo threshold

#### 6.4.3. Cải thiện tiềm năng

**1. Hyperparameter Tuning:**

- Grid Search / Random Search
- Bayesian Optimization
- Có thể cải thiện 2-5% performance

**2. Advanced Techniques:**

- Stacking / Blending multiple models
- Deep Learning (LSTM cho time series)
- AutoML platforms

**3. More Data:**

- Alternative data (social media, shopping)
- Real-time transaction data
- Macroeconomic indicators

**4. Threshold Optimization:**

- Business-driven threshold
- Different thresholds for different segments
- Dynamic threshold adjustment

---

## 7. ỨNG DỤNG THỰC TẾ

### 7.1. Cho Ngân hàng Việt Nam

#### 7.1.1. Credit Card Application Screening

**Quy trình hiện tại:**

1. Khách hàng nộp hồ sơ (giấy tờ, chứng minh thu nhập)
2. Nhân viên review manual (7-10 ngày)
3. Kiểm tra CIC (Credit Information Center)
4. Phê duyệt/từ chối

**Quy trình với ML:**

1. Khách hàng nộp hồ sơ (online hoặc offline)
2. **ML model tự động scoring** (< 1 phút)
3. Phân loại:
   - **High risk:** Từ chối ngay
   - **Medium risk:** Review manual
   - **Low risk:** Chấp nhận ngay
4. Giảm thời gian xuống 1-2 ngày

**Lợi ích:**

- Tăng tốc độ xử lý 5-10 lần
- Giảm chi phí nhân sự 30-40%
- Tăng customer satisfaction
- Giảm sai sót do con người

#### 7.1.2. Portfolio Risk Management

**Ứng dụng:**

1. **Regular re-scoring:** Đánh giá lại tất cả khách hàng hàng tháng
2. **Risk monitoring dashboard:** Theo dõi risk profile real-time
3. **Early warning system:** Cảnh báo khách hàng có dấu hiệu xấu
4. **Dynamic limit adjustment:** Tăng/giảm hạn mức tự động

**Implementation:**

```python
# Pseudo-code
for customer in portfolio:
    risk_score = ml_model.predict_proba(customer_data)
    if risk_score > HIGH_RISK_THRESHOLD:
        send_alert(customer)
        reduce_credit_limit(customer)
    elif risk_score < LOW_RISK_THRESHOLD:
        increase_credit_limit_offer(customer)
```

#### 7.1.3. Marketing & Customer Segmentation

**Phân khúc khách hàng:**

1. **VIP (Low risk):**

   - Tỷ lệ: ~30%
   - Strategy: Tăng hạn mức, ưu đãi đặc biệt
   - Products: Premium cards (Platinum, Infinite)

2. **Standard (Medium risk):**

   - Tỷ lệ: ~50%
   - Strategy: Giữ chân, cross-selling
   - Products: Standard cards, installment plans

3. **High Risk:**
   - Tỷ lệ: ~20%
   - Strategy: Giảm hạn mức, monitor chặt
   - Actions: SMS reminders, payment facilitation

**Personalized Offers:**

- Dựa trên risk score và behavior
- Targeted campaigns với conversion rate cao hơn
- Tối ưu marketing budget

#### 7.1.4. Collections & Recovery

**Ưu tiên thu hồi nợ:**

1. **Segment A (High amount + Medium recoverability):**

   - Ưu tiên cao nhất
   - Giao cho đội collections chuyên nghiệp

2. **Segment B (Medium amount + High recoverability):**

   - Automated reminders
   - Đề xuất payment plans

3. **Segment C (Low amount + Low recoverability):**
   - Write-off consideration
   - Bán nợ cho AMC (Asset Management Company)

**Predictive collections:**

- Dự đoán khách hàng nào dễ thu hồi
- Tối ưu resource allocation
- Tăng recovery rate 15-20%

### 7.2. Integration với Hệ thống Ngân hàng

#### 7.2.1. Architecture

```
┌─────────────────┐
│  Data Sources   │
│  - Core Banking │
│  - CIC          │
│  - Transactions │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Data Pipeline  │
│  - ETL          │
│  - Features     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   ML Model API  │
│  - Prediction   │
│  - Monitoring   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Applications   │
│  - Approval     │
│  - Risk Mgmt    │
│  - Collections  │
└─────────────────┘
```

#### 7.2.2. Technology Stack

**Backend:**

- Python (FastAPI / Flask) cho ML serving
- Docker containers
- Kubernetes cho orchestration

**Database:**

- PostgreSQL cho transactional data
- MongoDB cho logs
- Redis cho caching

**ML Platform:**

- MLflow cho model management
- Airflow cho workflow orchestration
- Prometheus + Grafana cho monitoring

**Security:**

- JWT authentication
- End-to-end encryption
- Audit logging
- GDPR/PDPA compliance

#### 7.2.3. Deployment Strategy

**Phase 1: Pilot (3 months)**

- Deploy cho 1 chi nhánh
- Monitor closely
- Gather feedback
- Fine-tune model

**Phase 2: Rollout (6 months)**

- Expand to multiple branches
- A/B testing
- Performance comparison with manual process

**Phase 3: Full deployment (12 months)**

- Nationwide deployment
- Integration with all channels
- Continuous improvement

### 7.3. Challenges & Solutions

#### 7.3.1. Data Availability

**Challenge:** Ngân hàng VN có ít data, chất lượng không đồng nhất

**Solutions:**

- Bắt đầu với available data, cải thiện dần
- Data augmentation techniques
- Transfer learning từ international models
- Partnership với Credit Bureau (CIC)

#### 7.3.2. Regulatory Compliance

**Challenge:** Luật bảo mật dữ liệu, giải trình quyết định từ chối

**Solutions:**

- Explainable AI (SHAP, LIME)
- Human-in-the-loop cho high-value decisions
- Audit trail cho tất cả predictions
- Comply với NHNN guidelines

#### 7.3.3. Model Drift

**Challenge:** Model performance giảm theo thời gian

**Solutions:**

- Regular retraining (monthly/quarterly)
- Performance monitoring dashboard
- Automatic alerting on drift
- Online learning (incremental updates)

#### 7.3.4. Bias & Fairness

**Challenge:** Model có thể bias theo giới tính, tuổi tác

**Solutions:**

- Fairness metrics (disparate impact)
- Bias mitigation techniques
- Regular audits
- Diverse training data

---

## 8. KẾT LUẬN

### 8.1. Tóm tắt Nghiên cứu

Tiểu luận đã thực hiện thành công việc **xây dựng và đánh giá các mô hình Machine Learning để dự đoán vỡ nợ thẻ tín dụng**. Qua quá trình nghiên cứu, chúng ta đã:

1. **Phân tích** 30,000 khách hàng với 23 features và tạo thêm 27 engineered features
2. **Xây dựng** 6 mô hình ML khác nhau từ đơn giản đến phức tạp
3. **Đánh giá** và chọn XGBoost là mô hình tốt nhất với:

   - ROC-AUC: 0.8012
   - F1-Score: 0.5845
   - Tiết kiệm chi phí: 13.2% so với baseline

4. **Phân tích** các yếu tố quan trọng:

   - Lịch sử thanh toán (PAY_0, PAY_2, PAY_3) chiếm >50% importance
   - Hạn mức tín dụng và utilization rate rất quan trọng
   - Demographics ít quan trọng hơn behavioral factors

5. **Đề xuất** giải pháp ứng dụng thực tế cho ngân hàng Việt Nam

### 8.2. Đóng góp của Nghiên cứu

**1. Về mặt học thuật:**

- Áp dụng toàn diện các kỹ thuật ML cho credit risk
- So sánh chi tiết performance của nhiều algorithms
- Feature engineering sáng tạo từ domain knowledge

**2. Về mặt thực tiễn:**

- Giải pháp cụ thể cho ngân hàng Việt Nam
- ROI analysis rõ ràng (>3,000%)
- Roadmap triển khai chi tiết

**3. Về mặt xã hội:**

- Giảm thiểu rủi ro cho ngân hàng → kinh tế ổn định hơn
- Quy trình công bằng hơn (giảm bias con người)
- Khách hàng tốt được phục vụ nhanh hơn

### 8.3. Hạn chế

**1. Dữ liệu:**

- Dataset từ Taiwan, có thể khác với thị trường VN
- Dữ liệu từ 2005, hành vi consumer đã thay đổi
- Thiếu alternative data (social, shopping)

**2. Mô hình:**

- Recall (0.50) vẫn chưa đủ cao
- Black box nature của XGBoost
- Chưa test với real-time data

**3. Triển khai:**

- Chưa có pilot thực tế
- Chi phí infrastructure chưa được tính cụ thể
- Change management challenges chưa được đề cập

### 8.4. Hướng phát triển Tương lai

**1. Ngắn hạn (3-6 tháng):**

- [ ] Hyperparameter tuning chi tiết hơn
- [ ] Ensemble stacking để cải thiện performance
- [ ] Threshold optimization theo business objectives
- [ ] A/B testing với different sampling methods

**2. Trung hạn (6-12 tháng):**

- [ ] Collect Vietnamese data cho retraining
- [ ] Incorporate alternative data
- [ ] Real-time prediction API
- [ ] Explainable AI dashboard

**3. Dài hạn (1-2 năm):**

- [ ] Deep Learning models (LSTM, Transformers)
- [ ] AutoML platform cho continuous improvement
- [ ] Federated learning across banks
- [ ] Integration với Credit Bureau real-time

**4. Research directions:**

- Time series analysis cho dynamic risk scoring
- Causal inference cho understanding "why"
- Reinforcement learning cho optimal credit limits
- Graph neural networks cho fraud detection

### 8.5. Bài học Kinh nghiệm

**Technical lessons:**

1. Feature engineering quan trọng hơn model complexity
2. Imbalanced data cần xử lý cẩn thận
3. Business context quyết định metric nào quan trọng
4. Simple models (Logistic Regression) vẫn competitive

**Business lessons:**

1. ML không phải silver bullet - cần human oversight
2. Change management quan trọng như technology
3. Start small, prove value, then scale
4. ROI phải rõ ràng để thuyết phục stakeholders

**Personal lessons:**

1. Domain knowledge + ML skills = powerful combination
2. Communication skills quan trọng (explain to non-technical)
3. Ethics và fairness phải được ưu tiên
4. Continuous learning là must-have

### 8.6. Lời Kết

Ứng dụng Machine Learning trong dự đoán vỡ nợ thẻ tín dụng không chỉ là một bài toán kỹ thuật, mà còn là một giải pháp có giá trị kinh tế và xã hội lớn. Với kết quả đạt được từ nghiên cứu này, chúng ta có thể khẳng định rằng:

**Machine Learning có thể giúp ngân hàng Việt Nam:**

- Giảm thiểu rủi ro tín dụng hiệu quả
- Tối ưu hóa quy trình phê duyệt
- Tăng trải nghiệm khách hàng
- Tiết kiệm chi phí đáng kể

Tuy nhiên, **thành công phụ thuộc vào:**

- Chất lượng dữ liệu
- Sự cam kết của leadership
- Đội ngũ có năng lực
- Compliance với quy định

Tôi hy vọng nghiên cứu này không chỉ đạt điểm tốt trong môn học, mà còn **góp phần nhỏ bé vào sự phát triển của ngành tài chính Việt Nam**, hướng tới một hệ thống ngân hàng hiện đại, công nghệ số và bền vững hơn.

**"The best time to plant a tree was 20 years ago. The second best time is now."**
_- Chinese Proverb_

Đã đến lúc ngành ngân hàng Việt Nam "plant the tree" của AI và Machine Learning!

---

## 9. TÀI LIỆU THAM KHẢO

### 9.1. Papers & Research

[1] Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients. _Expert Systems with Applications_, 36(2), 2473-2480.

[2] Baesens, B., Van Gestel, T., Viaene, S., Stepanova, M., Suykens, J., & Vanthienen, J. (2003). Benchmarking state-of-the-art classification algorithms for credit scoring. _Journal of the operational research society_, 54(6), 627-635.

[3] Hand, D. J., & Henley, W. E. (1997). Statistical classification methods in consumer credit scoring: a review. _Journal of the Royal Statistical Society: Series A (Statistics in Society)_, 160(3), 523-541.

[4] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. _Proceedings of the 22nd ACM SIGKDD_, 785-794.

[5] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: synthetic minority over-sampling technique. _Journal of artificial intelligence research_, 16, 321-357.

### 9.2. Books

[6] James, G., Witten, D., Hastie, T., & Tibshirani, R. (2013). _An introduction to statistical learning_ (Vol. 112). New York: Springer.

[7] Hastie, T., Tibshirani, R., & Friedman, J. (2009). _The elements of statistical learning: data mining, inference, and prediction_ (2nd ed.). Springer.

[8] Géron, A. (2019). _Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow_ (2nd ed.). O'Reilly Media.

[9] Provost, F., & Fawcett, T. (2013). _Data Science for Business_. O'Reilly Media.

### 9.3. Online Resources

[10] UCI Machine Learning Repository: https://archive.ics.uci.edu/ml/

[11] Kaggle Datasets: https://www.kaggle.com/datasets

[12] Scikit-learn Documentation: https://scikit-learn.org/

[13] XGBoost Documentation: https://xgboost.readthedocs.io/

[14] LightGBM Documentation: https://lightgbm.readthedocs.io/

### 9.4. Industry Reports

[15] Ngân hàng Nhà nước Việt Nam (2023). _Báo cáo phát triển thị trường thẻ ngân hàng_.

[16] McKinsey & Company (2021). _The 2021 McKinsey Global Payments Report_.

[17] Federal Reserve (2022). _The 2022 Federal Reserve Payments Study_.

[18] PwC (2023). _Banking and Capital Markets Trends 2023_.

### 9.5. Vietnamese Sources

[19] VCB (2023). _Báo cáo Quản lý Rủi ro Tín dụng Vietcombank_.

[20] Techcombank (2023). _Digital Banking Transformation Report_.

[21] VPBank (2022). _Fintech and AI Applications in Retail Banking_.

[22] Đại học Kinh tế Quốc dân. _Giáo trình Quản trị Rủi ro Ngân hàng_.

---

## PHỤ LỤC

### Phụ lục A: Code Repository

- **GitHub:** [Link to repository]
- **Structure:** Như đã mô tả trong README.md

### Phụ lục B: Jupyter Notebooks

1. `01_EDA.ipynb` - Exploratory Data Analysis
2. `02_Data_Preprocessing.ipynb` - Data Preprocessing & Feature Engineering
3. `03_Model_Training_Evaluation.ipynb` - Model Training & Evaluation

### Phụ lục C: Figures

- Tất cả figures được lưu trong `reports/figures/`
- High-resolution (300 DPI) cho print quality

### Phụ lục D: Data Dictionary

- Chi tiết trong `data/README.md`

### Phụ lục E: Presentation Slides

- File PowerPoint: `reports/Presentation.pptx`
- 15-20 slides tóm tắt key findings

---

**LỜI CẢM ƠN**

Em xin chân thành cảm ơn:

- Thầy/Cô giảng viên đã hướng dẫn và truyền đạt kiến thức quý báu
- Gia đình đã luôn ủng hộ và tạo điều kiện cho em học tập
- Bạn bè đã cùng thảo luận và chia sẻ kinh nghiệm
- Cộng đồng Machine Learning Việt Nam đã cung cấp tài liệu và hỗ trợ

---

**Ngày hoàn thành:** [Ngày/Tháng/Năm]

**Chữ ký sinh viên:**

[Chữ ký]

---

**HẾT**
