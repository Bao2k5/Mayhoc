Mục tiêu: Phân tích nhanh hệ thống "Credit Card Default Prediction" để làm đồ án — bao gồm kiến trúc, dữ liệu, pipeline, mô hình, cách chạy, đề xuất thí nghiệm, và cấu trúc báo cáo.

1) Tổng quan hệ thống
- Ứng dụng: Streamlit demo (file `app.py`) cho dự đoán vỡ nợ thẻ tín dụng.
- Dữ liệu: nằm trong `data/` (raw và processed). File processed chứa `preprocessed_data.pkl`, `feature_names.pkl`, X_train/X_test, y_train/y_test.
- Mô hình: nhiều mô hình đã train lưu trong `models/` (best_model.pkl, lightgbm.pkl, xgboost.pkl, random_forest.pkl, logistic_regression.pkl, neural_network.pkl) và `scaler.pkl`.
- Notebook: `notebooks/` gồm EDA, preprocessing, training/evaluation.

2) Thành phần chính và luồng dữ liệu
- Ingestion: `data/raw/UCI_Credit_Card.csv` → xử lý trong notebooks 02_Data_Preprocessing.ipynb → tạo dữ liệu đã tiền xử lý trong `data/processed/`.
- Feature engineering: hàm `engineer_features()` trong `app.py` tái tạo các biến engineered giống training (ví dụ MAX_PAY_DELAY, AVG_BILL_AMT, UTILIZATION_RATE,...).
- Modeling: model đã train (LightGBM được dùng làm best_model). Khi dự đoán: client input → engineer_features → sắp xếp feature theo `feature_names.pkl` → scale bằng `scaler.pkl` → model.predict/_proba.
- UI: streamlit app có sidebar (thông tin model), tabs: Assessment (input + predict), Demo (pre-built scenarios), Guide.

3) Cách chạy (local)
- Thiết lập môi trường (pip):
  - Sử dụng `requirements_min.txt` hoặc `requirements.txt`.
  - Lệnh nhanh (Windows cmd):

    python -m venv venv
    venv\Scripts\activate
    pip install -r requirements_min.txt

- Chạy app:

    cd "C:\Users\Bao\Desktop\Máy học\Credit_Card_Default_Prediction"
    streamlit run app.py --server.port 8501

- Kiểm thử model CLI (demo script):

    python test_model.py

4) Những test tôi đã chạy (tôi đã thực hiện để kiểm tra tính ổn định)
- Kiểm tra syntax `py_compile` cho `app.py` — OK.
- Chạy `test_model.py` từ thư mục project: load model & scaler OK; in demo cho 2 case (good/risky). Có cảnh báo sklearn về feature names nhưng không gây lỗi.
- Khởi và restart Streamlit server để xác nhận app đang lắng nghe port 8501.

5) Những điểm cần lưu ý / rủi ro
- Đồng bộ feature names: app phải sử dụng cùng `feature_names.pkl` như khi training; nếu engineer_features thay đổi, mapping có thể sai.
- Kiểm soát kiểu dữ liệu nhập (ví dụ PAY_* là integer trong khoảng -2..9). Thay number_input bằng selectbox sẽ giảm lỗi người dùng.
- Plotly: một số thuộc tính cần giá trị màu hợp lệ (đã fix `'transparent'` → `'rgba(0,0,0,0)'`).
- Streamlit styling: CSS tuỳ chỉnh có thể không áp dụng đều cho tất cả component — cần tinh chỉnh thêm cho dark mode.

6) Đề xuất cho đồ án (các phần và thí nghiệm)
A. Tiền xử lý & EDA (bắt buộc)
  - Mô tả dữ liệu, phân bố target, missing values, phân tích theo nhóm (gender, education, age).
  - Thực hiện feature selection và trình bày trực quan (correlation, feature importance).

B. Xây dựng baseline & model comparison (bắt buộc)
  - Baseline: Logistic Regression (bài toán nhị phân) với StandardScaler.
  - So sánh: RandomForest, XGBoost, LightGBM, simple NN.
  - Metrics: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, Confusion matrix, và cost-based metric nếu có chi phí.

C. Xử lý dữ liệu mất cân bằng
  - Thử oversampling (SMOTE), undersampling, class weights; so sánh kết quả trên ROC/PR.

D. Feature engineering (quan trọng để cải thiện)
  - Các biến trend/consistency (đã có vài biến); bổ sung interactions, polynomial, lag features.

E. Giải thích model & fairness
  - SHAP hoặc permutation importance để giải thích feature contributions.
  - Kiểm tra fairness theo giới tính/tuổi/education (audit bias).

F. Triển khai & UX
  - Hoàn thiện Streamlit app: input validation, selectbox cho PAY_*, dark-mode fixes, responsive layout.
  - Tối ưu hiệu năng model (pickle, caching), thêm endpoint API nếu cần.

7) Kế hoạch thực hiện đồ án (gợi ý timeline 4-6 tuần)
- Tuần 1: EDA, data cleaning, báo cáo khám phá dữ liệu.
- Tuần 2: Baseline models + preprocessing pipeline (scaler, feature list).
- Tuần 3: Feature engineering + model tuning (Grid/Random/Optuna), cross-validation.
- Tuần 4: Model explanation (SHAP), fairness checks, select final model.
- Tuần 5: Xây dựng Streamlit/UX, tích hợp model, viết báo cáo.
- Tuần 6: Hoàn thiện, slide demo, wrap-up.

8) Deliverables gợi ý cho đồ án
- Notebook EDA và Notebook Training (có seed và notebook readable).
- Script reproducible: `train.py` (train + save model, scaler, feature_names).
- `requirements.txt` và hướng dẫn chạy (README.md).
- Streamlit app (`app.py`) để demo.
- Báo cáo PDF/markdown + slides.

9) Các bước tôi có thể làm tiếp ngay (hãy chọn):
- (1) Chuyển `PAY_*` inputs sang selectbox (loại bỏ lỗi nhập tay). — tăng tính an toàn UX.
- (2) Tinh chỉnh CSS để dark-mode nhất quán (đặc biệt input/select styles). 
- (3) Thêm một `train.py` nhỏ để tái tạo model từ `data/processed` (nếu muốn tạo pipeline reproducible).
- (4) Tạo `README_PROJECT.md` với hướng dẫn đồ án ngắn gọn.

---
Nếu bạn muốn, tôi sẽ thực hiện một trong các mục (1)-(4) ngay bây giờ. Nếu bạn cần phân tích sâu hơn cho phần cụ thể (ví dụ: phân tích feature importance hoặc code để reproduce training), nói cho tôi biết mục ưu tiên của đồ án — tôi sẽ triển khai tiếp và test kỹ.

