"""
DEMO: Test Model Dự đoán Vỡ nợ Thẻ tín dụng
============================================
Script này cho phép bạn test model với dữ liệu mẫu hoặc dữ liệu tự nhập
"""

import pickle
import pandas as pd
import numpy as np
from pathlib import Path

# Load model và scaler
print("="*70)
print("🚀 ĐANG TẢI MODEL...")
print("="*70)

model_path = Path('models/best_model.pkl')
scaler_path = Path('models/scaler.pkl')

with open(model_path, 'rb') as f:
    model = pickle.load(f)
    
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

print("✅ Đã load model thành công!")
print(f"📊 Model: LightGBM (Best performing model)")
print(f"🎯 ROC-AUC Score: 0.7692")
print()

# Load feature names
with open('data/processed/feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

print(f"📋 Số features: {len(feature_names)}")
print()

def create_sample_customer(customer_type='good'):
    """
    Tạo dữ liệu khách hàng mẫu
    customer_type: 'good' (khách hàng tốt) hoặc 'risky' (khách hàng rủi ro)
    """
    if customer_type == 'good':
        # Khách hàng tốt - ít rủi ro vỡ nợ
        data = {
            'LIMIT_BAL': 200000,  # Hạn mức cao
            'SEX': 1,  # Nam
            'EDUCATION': 1,  # Cao học
            'MARRIAGE': 1,  # Đã kết hôn
            'AGE': 35,  # Độ tuổi trung niên
            'PAY_0': -1,  # Thanh toán đúng hạn
            'PAY_2': -1,
            'PAY_3': -1,
            'PAY_4': -1,
            'PAY_5': -1,
            'PAY_6': -1,
            'BILL_AMT1': 50000,
            'BILL_AMT2': 48000,
            'BILL_AMT3': 47000,
            'BILL_AMT4': 45000,
            'BILL_AMT5': 43000,
            'BILL_AMT6': 42000,
            'PAY_AMT1': 50000,  # Thanh toán đầy đủ
            'PAY_AMT2': 48000,
            'PAY_AMT3': 47000,
            'PAY_AMT4': 45000,
            'PAY_AMT5': 43000,
            'PAY_AMT6': 42000,
        }
        description = "✅ KHÁCH HÀNG TỐT"
    else:
        # Khách hàng rủi ro - cao rủi ro vỡ nợ
        data = {
            'LIMIT_BAL': 50000,  # Hạn mức thấp
            'SEX': 2,  # Nữ
            'EDUCATION': 3,  # THPT
            'MARRIAGE': 2,  # Độc thân
            'AGE': 25,  # Trẻ
            'PAY_0': 2,  # Trễ hạn 2 tháng
            'PAY_2': 2,
            'PAY_3': 1,
            'PAY_4': 3,
            'PAY_5': 2,
            'PAY_6': 1,
            'BILL_AMT1': 48000,  # Sử dụng gần hết hạn mức
            'BILL_AMT2': 47000,
            'BILL_AMT3': 49000,
            'BILL_AMT4': 48500,
            'BILL_AMT5': 47500,
            'BILL_AMT6': 48000,
            'PAY_AMT1': 1000,  # Thanh toán rất ít
            'PAY_AMT2': 1500,
            'PAY_AMT3': 1200,
            'PAY_AMT4': 1000,
            'PAY_AMT5': 1100,
            'PAY_AMT6': 1000,
        }
        description = "⚠️ KHÁCH HÀNG RỦI RO"
    
    return data, description

def engineer_features(df):
    """Tạo các engineered features giống như trong preprocessing"""
    
    # Payment delay features
    pay_cols = [f'PAY_{i}' for i in [0, 2, 3, 4, 5, 6]]
    df['MAX_PAY_DELAY'] = df[pay_cols].max(axis=1)
    df['AVG_PAY_DELAY'] = df[pay_cols].mean(axis=1)
    df['PAY_DELAY_SUM'] = df[pay_cols].sum(axis=1)
    
    # Bill amount features
    bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
    df['AVG_BILL_AMT'] = df[bill_cols].mean(axis=1)
    df['MAX_BILL_AMT'] = df[bill_cols].max(axis=1)
    df['MIN_BILL_AMT'] = df[bill_cols].min(axis=1)
    df['STD_BILL_AMT'] = df[bill_cols].std(axis=1)
    df['BILL_AMT_TREND'] = df['BILL_AMT1'] - df['BILL_AMT6']
    
    # Payment amount features
    pay_amt_cols = [f'PAY_AMT{i}' for i in range(1, 7)]
    df['AVG_PAY_AMT'] = df[pay_amt_cols].mean(axis=1)
    df['MAX_PAY_AMT'] = df[pay_amt_cols].max(axis=1)
    df['MIN_PAY_AMT'] = df[pay_amt_cols].min(axis=1)
    df['STD_PAY_AMT'] = df[pay_amt_cols].std(axis=1)
    df['PAY_AMT_TREND'] = df['PAY_AMT1'] - df['PAY_AMT6']
    
    # Utilization and payment ratio
    df['UTILIZATION_RATE'] = (df['AVG_BILL_AMT'] / df['LIMIT_BAL'] * 100).clip(upper=100)
    df['MAX_UTILIZATION'] = (df['MAX_BILL_AMT'] / df['LIMIT_BAL'] * 100).clip(upper=100)
    df['PAYMENT_RATIO'] = np.where(df['AVG_BILL_AMT'] > 0,
                                    (df['AVG_PAY_AMT'] / df['AVG_BILL_AMT'] * 100).clip(upper=200),
                                    0)
    
    # Advanced features
    df['TIMES_DELAYED'] = (df[pay_cols] > 0).sum(axis=1)
    df['CREDIT_USAGE_CONSISTENCY'] = 1 / (1 + df['STD_BILL_AMT'])
    df['PAYMENT_CONSISTENCY'] = 1 / (1 + df['STD_PAY_AMT'])
    
    # Interaction features
    df['AGE_LIMIT'] = df['AGE'] * df['LIMIT_BAL'] / 1000000
    df['EDUCATION_LIMIT'] = df['EDUCATION'] * df['LIMIT_BAL'] / 100000
    df['PAY_TO_LIMIT_RATIO'] = df['AVG_PAY_AMT'] / df['LIMIT_BAL']
    
    # Recent payment behavior
    df['RECENT_PAY_TREND'] = df['PAY_0'] - df['PAY_2']
    df['RECENT_BILL_TREND'] = df['BILL_AMT1'] - df['BILL_AMT2']
    df['RECENT_PAYMENT_TREND'] = df['PAY_AMT1'] - df['PAY_AMT2']
    
    return df

def predict_default(customer_data):
    """Dự đoán khả năng vỡ nợ"""
    
    # Tạo DataFrame
    df = pd.DataFrame([customer_data])
    
    # Engineer features
    df = engineer_features(df)
    
    # Đảm bảo có đủ tất cả features
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Sắp xếp theo đúng thứ tự features
    df = df[feature_names]
    
    # Scale features
    X_scaled = scaler.transform(df)
    
    # Dự đoán
    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0]
    
    return prediction, probability

def display_prediction(customer_data, description, prediction, probability):
    """Hiển thị kết quả dự đoán"""
    
    print("="*70)
    print(f"📊 {description}")
    print("="*70)
    
    # Thông tin khách hàng
    print("\n📋 THÔNG TIN KHÁCH HÀNG:")
    print(f"  • Hạn mức tín dụng: {customer_data['LIMIT_BAL']:,} NT$")
    print(f"  • Giới tính: {'Nam' if customer_data['SEX'] == 1 else 'Nữ'}")
    
    edu_map = {1: 'Cao học', 2: 'Đại học', 3: 'THPT', 4: 'Khác'}
    print(f"  • Học vấn: {edu_map.get(customer_data['EDUCATION'], 'Không rõ')}")
    
    marriage_map = {1: 'Đã kết hôn', 2: 'Độc thân', 3: 'Khác'}
    print(f"  • Tình trạng: {marriage_map.get(customer_data['MARRIAGE'], 'Không rõ')}")
    print(f"  • Tuổi: {customer_data['AGE']}")
    
    print(f"\n💳 LỊCH SỬ THANH TOÁN:")
    print(f"  • Trạng thái tháng gần nhất: {customer_data['PAY_0']}")
    print(f"  • Hóa đơn tháng gần nhất: {customer_data['BILL_AMT1']:,} NT$")
    print(f"  • Thanh toán tháng gần nhất: {customer_data['PAY_AMT1']:,} NT$")
    
    # Tính toán metrics
    utilization = (customer_data['BILL_AMT1'] / customer_data['LIMIT_BAL'] * 100)
    payment_ratio = (customer_data['PAY_AMT1'] / customer_data['BILL_AMT1'] * 100) if customer_data['BILL_AMT1'] > 0 else 0
    
    print(f"\n📈 CHỈ SỐ TÀI CHÍNH:")
    print(f"  • Tỷ lệ sử dụng hạn mức: {utilization:.1f}%")
    print(f"  • Tỷ lệ thanh toán: {payment_ratio:.1f}%")
    
    # Kết quả dự đoán
    print("\n" + "="*70)
    print("🎯 KẾT QUẢ DỰ ĐOÁN:")
    print("="*70)
    
    prob_no_default = probability[0] * 100
    prob_default = probability[1] * 100
    
    print(f"\n  Xác suất KHÔNG vỡ nợ: {prob_no_default:.2f}%")
    print(f"  Xác suất VỠ NỢ: {prob_default:.2f}%")
    
    print("\n" + "-"*70)
    
    if prediction == 0:
        print("  ✅ KẾT LUẬN: Khách hàng CÓ KHẢ NĂNG thanh toán")
        print("  💚 Khuyến nghị: CHẤP NHẬN đơn xin tín dụng")
    else:
        print("  ⚠️ KẾT LUẬN: Khách hàng CÓ RủI RO vỡ nợ cao")
        print("  ❌ Khuyến nghị: XEM XÉT KỸ hoặc TỪ CHỐI đơn xin tín dụng")
    
    print("-"*70)
    
    # Risk assessment
    if prob_default < 20:
        risk_level = "🟢 THẤP"
    elif prob_default < 40:
        risk_level = "🟡 TRUNG BÌNH"
    elif prob_default < 60:
        risk_level = "🟠 CAO"
    else:
        risk_level = "🔴 RẤT CAO"
    
    print(f"\n  📊 Mức độ rủi ro: {risk_level}")
    print("\n" + "="*70 + "\n")

def main():
    """Chạy demo"""
    
    print("\n" + "="*70)
    print("🎯 DEMO: HỆ THỐNG DỰ ĐOÁN VỠ NỢ THẺ TÍN DỤNG")
    print("="*70)
    print("\nĐây là demo model Machine Learning dự đoán khả năng")
    print("khách hàng vỡ nợ dựa trên lịch sử tài chính và demographics.\n")
    
    # Test với khách hàng tốt
    print("\n" + "▶" *35)
    print("TEST 1: KHÁCH HÀNG LÀNH MẠNH")
    print("▶" *35 + "\n")
    
    good_customer, good_desc = create_sample_customer('good')
    prediction, probability = predict_default(good_customer)
    display_prediction(good_customer, good_desc, prediction, probability)
    
    # Test với khách hàng rủi ro
    print("\n" + "▶" *35)
    print("TEST 2: KHÁCH HÀNG RỦI RO")
    print("▶" *35 + "\n")
    
    risky_customer, risky_desc = create_sample_customer('risky')
    prediction, probability = predict_default(risky_customer)
    display_prediction(risky_customer, risky_desc, prediction, probability)
    
    # Thông tin thêm
    print("\n" + "="*70)
    print("📚 THÔNG TIN MODEL")
    print("="*70)
    print("\n  • Algorithm: LightGBM (Gradient Boosting)")
    print("  • Features: 50 features (23 gốc + 27 engineered)")
    print("  • Training data: 30,000 khách hàng")
    print("  • Performance:")
    print("    - Accuracy: 80.43%")
    print("    - ROC-AUC: 0.7692")
    print("    - F1-Score: 0.5092")
    print("\n  • Business Impact:")
    print("    - Tiết kiệm: 19.0 tỷ VND (41.0%)")
    print("    - Tiết kiệm TB/khách hàng: 3,172,500 VND")
    print("\n" + "="*70)
    print("\n✅ Demo hoàn thành! Bạn có thể chỉnh sửa script để test")
    print("với dữ liệu khác hoặc tích hợp vào ứng dụng của bạn.\n")

if __name__ == "__main__":
    main()
