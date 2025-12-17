# train_lightgbm.py
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import warnings

warnings.filterwarnings('ignore')

# 1. 加载数据
train_path = r'F:\Project\Breast\train_data.csv'
val_path = r'F:\Project\Breast\validation_data.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

print("=" * 60)
print("数据加载完成")
print("=" * 60)

# 2. 选择指定的6个特征
selected_features = [
    'radius_worst',
    'concave points_mean',
    'radius_se',
    'concavity_worst',
    'area_worst',
    'compactness_mean'
]

print(f"使用的6个特征: {selected_features}")

# 3. 准备数据
X_train = train_df[selected_features]
y_train = train_df['diagnosis']
X_val = val_df[selected_features]
y_val = val_df['diagnosis']

# 4. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 5. 训练LightGBM模型
print("\n正在训练LightGBM模型...")

# 设置LightGBM参数
lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 0,
    'random_state': 42,
    'n_jobs': -1
}

# 创建LightGBM数据集
train_data = lgb.Dataset(X_train_scaled, label=y_train)
val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)

# 训练模型
model = lgb.train(
    lgb_params,
    train_data,
    valid_sets=[val_data],
    num_boost_round=100,
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(10)]
)

# 6. 模型评估
y_pred_proba = model.predict(X_val_scaled)
y_pred = (y_pred_proba > 0.5).astype(int)

accuracy = accuracy_score(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_pred_proba)

print("\n" + "=" * 60)
print("模型评估结果:")
print("=" * 60)
print(f"准确率: {accuracy:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print("\n分类报告:")
print(classification_report(y_val, y_pred, target_names=['良性(B)', '恶性(M)']))

# 7. 获取特征重要性
feature_importance = model.feature_importance(importance_type='gain')
feature_names = selected_features

importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n" + "=" * 60)
print("特征重要性:")
print("=" * 60)
for i, (_, row) in enumerate(importance_df.iterrows(), 1):
    print(f"{i}. {row['feature']:20s} 重要性: {row['importance']:.4f}")

# 8. 保存模型和预处理对象
output_dir = r'F:\Project\Breast\deployment'
import os

os.makedirs(output_dir, exist_ok=True)

# 保存模型
model_path = os.path.join(output_dir, 'lightgbm_model.pkl')
joblib.dump(model, model_path)

# 保存标准化器
scaler_path = os.path.join(output_dir, 'scaler.pkl')
joblib.dump(scaler, scaler_path)

# 保存特征信息
feature_info = {
    'selected_features': selected_features,
    'feature_importance': importance_df.to_dict('records')
}

import json

with open(os.path.join(output_dir, 'feature_info.json'), 'w', encoding='utf-8') as f:
    json.dump(feature_info, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 60)
print("模型保存完成:")
print("=" * 60)
print(f"模型文件: {model_path}")
print(f"标准化器: {scaler_path}")
print(f"特征信息: {output_dir}\\feature_info.json")


# 9. 创建推理测试函数
def predict_single_sample(features_dict):
    """预测单个样本"""
    # 将字典转换为DataFrame
    input_df = pd.DataFrame([features_dict])

    # 确保特征顺序正确
    input_df = input_df[selected_features]

    # 标准化
    input_scaled = scaler.transform(input_df)

    # 预测
    probability = model.predict(input_scaled)[0]
    prediction = 1 if probability > 0.5 else 0
    prediction_label = '恶性(M)' if prediction == 1 else '良性(B)'

    return {
        'prediction': prediction,
        'prediction_label': prediction_label,
        'probability': probability,
        'probability_percent': f"{probability * 100:.2f}%"
    }


# 测试推理函数
print("\n" + "=" * 60)
print("推理测试:")
print("=" * 60)

# 创建一个测试样本（使用验证集的平均值）
test_sample = {
    'radius_worst': X_val['radius_worst'].mean(),
    'concave points_mean': X_val['concave points_mean'].mean(),
    'radius_se': X_val['radius_se'].mean(),
    'concavity_worst': X_val['concavity_worst'].mean(),
    'area_worst': X_val['area_worst'].mean(),
    'compactness_mean': X_val['compactness_mean'].mean()
}

result = predict_single_sample(test_sample)
print(f"测试样本预测结果:")
print(f"  预测标签: {result['prediction_label']}")
print(f"  恶性概率: {result['probability_percent']}")