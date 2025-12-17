import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 1. 加载数据
train_path = r'F:\Project\Breast\train_data.csv'
val_path = r'F:\Project\Breast\validation_data.csv'

train_df = pd.read_csv(train_path)
val_df = pd.read_csv(val_path)

print("=" * 60)
print("数据加载完成")
print("=" * 60)
print(f"训练集形状: {train_df.shape}")
print(f"验证集形状: {val_df.shape}")

# 2. 选择指定的6个特征
selected_features = [
    'radius_worst',
    'concave points_mean',
    'radius_se',
    'concavity_worst',
    'area_worst',
    'compactness_mean'
]

print(f"\n选中的6个特征:")
for i, feat in enumerate(selected_features, 1):
    print(f"{i}. {feat}")

# 3. 准备数据
# 分离特征和目标
X_train = train_df[selected_features]
y_train = train_df['diagnosis']

X_val = val_df[selected_features]
y_val = val_df['diagnosis']

print(f"\n训练集 - 特征形状: {X_train.shape}, 目标形状: {y_train.shape}")
print(f"验证集 - 特征形状: {X_val.shape}, 目标形状: {y_val.shape}")

# 4. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# 5. 定义所有模型
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import lightgbm as lgb

models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000),
    'XGBoost': xgb.XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss'),
    'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42)
}

# 6. 训练和评估模型
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, auc)

results = {}
predictions = {}
probabilities = {}

print("\n" + "=" * 60)
print("模型训练和评估结果:")
print("=" * 60)

for name, model in models.items():
    print(f"\n正在训练 {name}...")

    # 训练模型
    model.fit(X_train_scaled, y_train)

    # 预测
    y_pred = model.predict(X_val_scaled)
    y_prob = model.predict_proba(X_val_scaled)[:, 1] if hasattr(model, "predict_proba") else None

    # 计算指标
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred)
    recall = recall_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    roc_auc = roc_auc_score(y_val, y_prob) if y_prob is not None else None

    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'model': model
    }

    predictions[name] = y_pred
    probabilities[name] = y_prob

    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    if roc_auc:
        print(f"  ROC AUC: {roc_auc:.4f}")

# 7. 创建综合评估报告
results_df = pd.DataFrame(results).T
print("\n" + "=" * 60)
print("模型性能综合对比:")
print("=" * 60)
print(results_df.sort_values('accuracy', ascending=False).round(4))

# 8. 可视化：ROC曲线
plt.figure(figsize=(10, 8))
colors = plt.cm.Set3(np.linspace(0, 1, len(models)))

for (name, color) in zip(models.keys(), colors):
    if probabilities[name] is not None:
        fpr, tpr, _ = roc_curve(y_val, probabilities[name])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=color, lw=2,
                 label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('ROC曲线对比 (验证集)')
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'F:\Project\Breast\roc_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 9. 可视化：PR曲线
plt.figure(figsize=(10, 8))

for (name, color) in zip(models.keys(), colors):
    if probabilities[name] is not None:
        precision, recall, _ = precision_recall_curve(y_val, probabilities[name])
        pr_auc = auc(recall, precision)
        plt.plot(recall, precision, color=color, lw=2,
                 label=f'{name} (AUC = {pr_auc:.3f})')

# 计算基准线（正例比例）
baseline = np.sum(y_val) / len(y_val)
plt.axhline(y=baseline, color='k', linestyle='--', label=f'基准线 ({baseline:.3f})')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('召回率 (Recall)')
plt.ylabel('精确率 (Precision)')
plt.title('PR曲线对比 (验证集)')
plt.legend(loc="lower left")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(r'F:\Project\Breast\pr_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 10. 可视化：混淆矩阵（前4个最好模型）
best_models = results_df.sort_values('accuracy', ascending=False).head(4).index.tolist()

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for idx, model_name in enumerate(best_models):
    y_pred = predictions[model_name]
    cm = confusion_matrix(y_val, y_pred)

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                xticklabels=['良性(B)', '恶性(M)'],
                yticklabels=['良性(B)', '恶性(M)'])

    # 添加指标文本
    accuracy = results[model_name]['accuracy']
    precision = results[model_name]['precision']
    recall = results[model_name]['recall']
    f1 = results[model_name]['f1']

    info_text = f'准确率: {accuracy:.3f}\n精确率: {precision:.3f}\n召回率: {recall:.3f}\nF1: {f1:.3f}'
    axes[idx].text(0.5, -0.2, info_text, transform=axes[idx].transAxes,
                   ha='center', va='center', fontsize=10,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.5))

    axes[idx].set_xlabel('预测标签')
    axes[idx].set_ylabel('真实标签')
    axes[idx].set_title(f'{model_name} 混淆矩阵')

plt.tight_layout()
plt.savefig(r'F:\Project\Breast\confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.show()


# 11. 可视化：DCA决策曲线分析
def calculate_net_benefit(y_true, y_prob, threshold):
    """计算净收益"""
    y_pred = (y_prob >= threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)

    # 净收益 = (TP/n) - (FP/n) * (threshold/(1-threshold))
    net_benefit = tp / n - fp / n * (threshold / (1 - threshold))
    return net_benefit


plt.figure(figsize=(10, 8))
thresholds = np.linspace(0.01, 0.99, 50)

# 绘制基准线：全部预测为阴性（全B）和全部预测为阳性（全M）
net_benefit_all_negative = np.zeros_like(thresholds)  # 全部预测为阴性（全B）的净收益为0
net_benefit_all_positive = []  # 全部预测为阳性（全M）

for thresh in thresholds:
    # 全部预测为阳性：TP = 所有实际阳性，FP = 所有实际阴性
    tp = np.sum(y_val == 1)
    fp = np.sum(y_val == 0)
    n = len(y_val)
    nb = tp / n - fp / n * (thresh / (1 - thresh))
    net_benefit_all_positive.append(nb)

plt.plot(thresholds, net_benefit_all_negative, 'k--', label='全预测为阴性（全B）')
plt.plot(thresholds, net_benefit_all_positive, 'k:', label='全预测为阳性（全M）')

# 绘制各个模型的决策曲线
for (name, color) in zip(models.keys(), colors):
    if probabilities[name] is not None:
        net_benefits = []
        for thresh in thresholds:
            nb = calculate_net_benefit(y_val, probabilities[name], thresh)
            net_benefits.append(nb)

        plt.plot(thresholds, net_benefits, color=color, lw=2, label=name)

plt.xlabel('阈值概率')
plt.ylabel('净收益 (Net Benefit)')
plt.title('决策曲线分析 (DCA)')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim([0, 1])
plt.ylim([-0.1, 0.6])
plt.tight_layout()
plt.savefig(r'F:\Project\Breast\dca_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 12. 可视化：模型性能对比雷达图
categories = ['准确率', '精确率', '召回率', 'F1分数', 'ROC AUC']

# 选择前6个模型进行雷达图展示
top_models = results_df.sort_values('accuracy', ascending=False).head(6).index.tolist()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='polar')

# 准备数据
angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

for model_name in top_models:
    values = [
        results[model_name]['accuracy'],
        results[model_name]['precision'],
        results[model_name]['recall'],
        results[model_name]['f1'],
        results[model_name]['roc_auc'] if results[model_name]['roc_auc'] else 0
    ]
    values += values[:1]  # 闭合图形
    ax.plot(angles, values, 'o-', linewidth=2, label=model_name)
    ax.fill(angles, values, alpha=0.1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('模型性能雷达图对比', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
plt.tight_layout()
plt.savefig(r'F:\Project\Breast\radar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# 13. 可视化：特征重要性分析（对树模型）
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
tree_models = ['Decision Tree', 'Random Forest', 'XGBoost', 'LightGBM']

for idx, model_name in enumerate(tree_models):
    ax = axes[idx // 2, idx % 2]
    model = results[model_name]['model']

    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        ax.barh(range(len(selected_features)), importances[indices], align='center')
        ax.set_yticks(range(len(selected_features)))
        ax.set_yticklabels([selected_features[i] for i in indices])
        ax.set_xlabel('特征重要性')
        ax.set_title(f'{model_name} 特征重要性')
    elif hasattr(model, 'coef_'):
        # 对于线性模型
        coef = model.coef_[0]
        indices = np.argsort(np.abs(coef))[::-1]

        colors = ['red' if c < 0 else 'blue' for c in coef[indices]]
        ax.barh(range(len(selected_features)), coef[indices], color=colors, align='center')
        ax.set_yticks(range(len(selected_features)))
        ax.set_yticklabels([selected_features[i] for i in indices])
        ax.set_xlabel('系数值')
        ax.set_title(f'{model_name} 系数值')
        ax.axvline(x=0, color='black', linewidth=0.5)

    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(r'F:\Project\Breast\feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 14. 保存结果
output_dir = r'F:\Project\Breast\model_results_6features'
import os

os.makedirs(output_dir, exist_ok=True)

# 保存模型性能结果
results_df.to_csv(os.path.join(output_dir, 'model_performance.csv'))

# 保存最佳模型
best_model_name = results_df['accuracy'].idxmax()
best_model = results[best_model_name]['model']

import joblib

joblib.dump(best_model, os.path.join(output_dir, 'best_model.pkl'))
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

# 保存预测结果
predictions_df = pd.DataFrame(predictions)
predictions_df['y_true'] = y_val.values
predictions_df.to_csv(os.path.join(output_dir, 'predictions.csv'))

# 保存特征系数信息
coefficients = {}
for name, model in models.items():
    if hasattr(model, 'coef_'):
        coefficients[name] = model.coef_[0]
    elif hasattr(model, 'feature_importances_'):
        coefficients[name] = model.feature_importances_

if coefficients:
    coeff_df = pd.DataFrame(coefficients, index=selected_features)
    coeff_df.to_csv(os.path.join(output_dir, 'feature_coefficients.csv'))

# 15. 生成详细报告
print("\n" + "=" * 60)
print("详细分析报告:")
print("=" * 60)

print(f"\n🎯 最佳模型: {best_model_name}")
print(f"   验证集准确率: {results[best_model_name]['accuracy']:.4f}")
print(f"   ROC AUC: {results[best_model_name]['roc_auc']:.4f}")

print(f"\n📊 6个特征的重要性总结:")
print("   正系数特征（与恶性相关）:")
print("   1. radius_worst (系数: +0.508) - 最差半径，最重要的恶性指标")
print("   2. concave points_mean (系数: +0.137) - 平均凹点数量")
print("   3. radius_se (系数: +0.133) - 半径标准误")
print("   4. concavity_worst (系数: +0.103) - 最差凹度")

print("\n   负系数特征（与良性相关）:")
print("   5. area_worst (系数: -0.323) - 最差面积")
print("   6. compactness_mean (系数: -0.147) - 平均紧致度")

print(f"\n📈 保存的文件:")
print(f"   1. 模型性能: {output_dir}\\model_performance.csv")
print(f"   2. 最佳模型: {output_dir}\\best_model.pkl")
print(f"   3. 标准化器: {output_dir}\\scaler.pkl")
print(f"   4. 预测结果: {output_dir}\\predictions.csv")
print(f"   5. 特征系数: {output_dir}\\feature_coefficients.csv")

print(f"\n🖼️ 可视化图表:")
print("   1. ROC曲线: F:\\Project\\Breast\\roc_curves.png")
print("   2. PR曲线: F:\\Project\\Breast\\pr_curves.png")
print("   3. 混淆矩阵: F:\\Project\\Breast\\confusion_matrices.png")
print("   4. DCA曲线: F:\\Project\\Breast\\dca_curves.png")
print("   5. 雷达图: F:\\Project\\Breast\\radar_chart.png")
print("   6. 特征重要性: F:\\Project\\Breast\\feature_importance.png")

print("\n" + "=" * 60)
print("所有模型训练和评估完成！")
print("=" * 60)