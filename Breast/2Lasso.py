import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
import joblib
import os

# 设置样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# 1. 加载训练数据
train_path = r'F:\Project\Breast\train_data.csv'
train_df = pd.read_csv(train_path)

print("=" * 60)
print("训练数据基本信息:")
print("=" * 60)
print(f"训练数据形状: {train_df.shape}")

# 2. 分离特征和目标变量
X_train = train_df.iloc[:, :-2]  # 特征（前30列）
y_train = train_df['diagnosis']  # 编码后的标签（0=良性, 1=恶性）

print(f"特征矩阵形状: {X_train.shape}")
print(f"目标变量形状: {y_train.shape}")

# 3. 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
feature_names = X_train.columns.tolist()

# 4. 使用LassoCV选择最佳alpha
print("\n" + "=" * 60)
print("使用LassoCV选择最佳正则化参数alpha...")
print("=" * 60)

alphas = np.logspace(-4, 0, 50)
lasso_cv = LassoCV(alphas=alphas, cv=5, max_iter=10000, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)

print(f"最佳alpha值: {lasso_cv.alpha_:.6f}")

# 5. 使用最佳alpha训练Lasso模型
lasso_best = Lasso(alpha=lasso_cv.alpha_, max_iter=10000, random_state=42)
lasso_best.fit(X_train_scaled, y_train)

# 获取系数
coef = lasso_best.coef_

# 6. 筛选系数绝对值大于0.1的特征
print("\n" + "=" * 60)
print("筛选系数绝对值 > 0.1 的特征:")
print("=" * 60)

# 创建系数DataFrame
coef_df = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef,
    'abs_coefficient': np.abs(coef)
})

# 筛选系数绝对值大于0.1的特征
selected_features_df = coef_df[coef_df['abs_coefficient'] > 0.1].copy()
selected_features_df = selected_features_df.sort_values('abs_coefficient', ascending=False)

# 按系数值排序（正负分开显示）
selected_features_pos = selected_features_df[selected_features_df['coefficient'] > 0].sort_values('coefficient', ascending=False)
selected_features_neg = selected_features_df[selected_features_df['coefficient'] < 0].sort_values('coefficient')

print(f"\n找到 {len(selected_features_df)} 个系数绝对值大于0.1的特征:")
print(f"正系数特征 ({len(selected_features_pos)}个): 与恶性正相关")
print(f"负系数特征 ({len(selected_features_neg)}个): 与恶性负相关（与良性正相关）")

print("\n" + "-" * 60)
print("正系数特征（值越大越可能是恶性）:")
print("-" * 60)
for i, (idx, row) in enumerate(selected_features_pos.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:30s} 系数: {row['coefficient']:+.6f}")

print("\n" + "-" * 60)
print("负系数特征（值越大越可能是良性）:")
print("-" * 60)
for i, (idx, row) in enumerate(selected_features_neg.iterrows(), 1):
    print(f"{i:2d}. {row['feature']:30s} 系数: {row['coefficient']:+.6f}")

# 7. 获取选中的特征名称
selected_features = selected_features_df['feature'].tolist()

print(f"\n" + "=" * 60)
print("特征选择统计:")
print("=" * 60)
print(f"原始特征总数: {len(feature_names)}")
print(f"选中特征数量: {len(selected_features)}")
print(f"特征减少比例: {(1 - len(selected_features)/len(feature_names)):.1%}")
print(f"保留的特征比例: {len(selected_features)/len(feature_names):.1%}")

# 8. 创建选中特征的数据集
X_train_selected = X_train[selected_features]
X_train_selected_scaled = X_train_scaled[:, [feature_names.index(f) for f in selected_features]]

# 9. 验证选中特征的效果
print("\n" + "=" * 60)
print("特征选择效果验证:")
print("=" * 60)

# 使用逻辑回归进行验证
logreg_full = LogisticRegression(max_iter=1000, random_state=42)
scores_full = cross_val_score(logreg_full, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"全部特征 ({len(feature_names)}个) 的交叉验证准确率:")
print(f"  平均: {scores_full.mean():.4f}")
print(f"  标准差: {scores_full.std():.4f}")

logreg_selected = LogisticRegression(max_iter=1000, random_state=42)
scores_selected = cross_val_score(logreg_selected, X_train_selected_scaled, y_train, cv=5, scoring='accuracy')
print(f"\n选中特征 ({len(selected_features)}个) 的交叉验证准确率:")
print(f"  平均: {scores_selected.mean():.4f}")
print(f"  标准差: {scores_selected.std():.4f}")

# 计算性能变化
accuracy_change = (scores_selected.mean() - scores_full.mean()) / scores_full.mean() * 100
print(f"\n准确率变化: {accuracy_change:+.2f}%")

if accuracy_change > 0:
    print("✓ 特征选择提升了模型性能！")
elif abs(accuracy_change) < 2:
    print("○ 特征选择对性能影响不大，但简化了模型。")
else:
    print("⚠ 特征选择降低了模型性能，可能需要调整阈值。")

# 10. 可视化特征系数
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：选中特征的系数（按值排序）
ax1 = axes[0, 0]
all_selected_features = pd.concat([selected_features_pos, selected_features_neg])
colors = ['red' if c < 0 else 'blue' for c in all_selected_features['coefficient']]
bars = ax1.barh(range(len(all_selected_features)), all_selected_features['coefficient'], color=colors)
ax1.set_yticks(range(len(all_selected_features)))
ax1.set_yticklabels(all_selected_features['feature'])
ax1.set_xlabel('Lasso系数值')
ax1.set_title(f'选中特征系数 (绝对值 > 0.1, 共{len(all_selected_features)}个)')
ax1.axvline(x=0, color='black', linewidth=0.8)

# 添加系数值标签
for i, (bar, coeff) in enumerate(zip(bars, all_selected_features['coefficient'])):
    ax1.text(coeff + (0.01 if coeff >= 0 else -0.03), bar.get_y() + bar.get_height()/2,
             f'{coeff:.3f}', ha='left' if coeff >= 0 else 'right', va='center', fontsize=9)

# 子图2：系数绝对值分布
ax2 = axes[0, 1]
sorted_by_abs = selected_features_df.sort_values('abs_coefficient', ascending=True)
ax2.barh(range(len(sorted_by_abs)), sorted_by_abs['abs_coefficient'], color='green')
ax2.set_yticks(range(len(sorted_by_abs)))
ax2.set_yticklabels(sorted_by_abs['feature'])
ax2.set_xlabel('系数绝对值')
ax2.set_title('特征重要性排序（按系数绝对值）')
ax2.axvline(x=0.1, color='red', linestyle='--', linewidth=1.5, label='阈值=0.1')

# 添加阈值线说明
ax2.text(0.1 + 0.01, len(sorted_by_abs)/2, f'阈值线\n(>0.1保留)',
         verticalalignment='center', color='red', fontweight='bold')

# 子图3：特征数量对比
ax3 = axes[1, 0]
categories = ['原始特征', '选中特征']
counts = [len(feature_names), len(selected_features)]
colors_comp = ['lightblue', 'lightgreen']
bars3 = ax3.bar(categories, counts, color=colors_comp)
ax3.set_ylabel('特征数量')
ax3.set_title('特征选择前后数量对比')
ax3.grid(True, alpha=0.3, axis='y')

# 在柱状图上显示数量
for bar, count in zip(bars3, counts):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2, height + 0.5,
             str(count), ha='center', va='bottom', fontweight='bold')

# 子图4：模型性能对比
ax4 = axes[1, 1]
x_pos = [0, 1]
full_mean = scores_full.mean()
full_std = scores_full.std()
selected_mean = scores_selected.mean()
selected_std = scores_selected.std()

bars4 = ax4.bar(x_pos, [full_mean, selected_mean], yerr=[full_std, selected_std],
               capsize=10, color=['lightcoral', 'lightgreen'], alpha=0.7)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(['全部特征', '选中特征'])
ax4.set_ylabel('交叉验证准确率')
ax4.set_title('模型性能对比')
ax4.set_ylim([0.9, 1.0])
ax4.grid(True, alpha=0.3, axis='y')

# 在柱状图上显示准确率
for bar, mean, std in zip(bars4, [full_mean, selected_mean], [full_std, selected_std]):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2, height + 0.002,
             f'{mean:.4f}\n±{std:.4f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig(r'F:\Project\Breast\lasso_features_gt_0.1.png', dpi=300, bbox_inches='tight')
plt.show()

# 11. 保存结果
output_dir = r'F:\Project\Breast\feature_selection_lasso_0.1'
os.makedirs(output_dir, exist_ok=True)

# 保存选中特征列表
selected_features_df.to_csv(os.path.join(output_dir, 'selected_features_gt_0.1.csv'), index=False)

# 保存所有特征的系数（标记是否选中）
coef_df['selected'] = coef_df['abs_coefficient'] > 0.1
coef_df_sorted = coef_df.sort_values('abs_coefficient', ascending=False)
coef_df_sorted.to_csv(os.path.join(output_dir, 'all_features_with_selection.csv'), index=False)

# 保存选中特征的数据集
train_selected_df = pd.DataFrame(X_train_selected)
train_selected_df['diagnosis'] = y_train.values
train_selected_df['diagnosis_original'] = train_df['diagnosis_original'].values
train_selected_df.to_csv(os.path.join(output_dir, 'train_data_selected.csv'), index=False)

# 保存标准化器
joblib.dump(scaler, os.path.join(output_dir, 'scaler.pkl'))

# 保存特征索引映射
feature_mapping = {
    'feature_names': feature_names,
    'selected_features': selected_features,
    'selected_indices': [feature_names.index(f) for f in selected_features]
}
import json
with open(os.path.join(output_dir, 'feature_mapping.json'), 'w') as f:
    json.dump(feature_mapping, f, indent=2)

print(f"\n" + "=" * 60)
print("结果保存:")
print("=" * 60)
print(f"1. 选中特征列表: {output_dir}\\selected_features_gt_0.1.csv")
print(f"2. 所有特征系数: {output_dir}\\all_features_with_selection.csv")
print(f"3. 选中特征数据集: {output_dir}\\train_data_selected.csv")
print(f"4. 标准化器: {output_dir}\\scaler.pkl")
print(f"5. 特征映射文件: {output_dir}\\feature_mapping.json")
print(f"6. 可视化图表: F:\\Project\\Breast\\lasso_features_gt_0.1.png")

# 12. 生成特征选择报告
print("\n" + "=" * 60)
print("特征选择报告:")
print("=" * 60)

print("\n🔍 最重要的5个特征（按绝对值）:")
top5 = selected_features_df.head(5)
for i, (idx, row) in enumerate(top5.iterrows(), 1):
    direction = "正相关" if row['coefficient'] > 0 else "负相关"
    print(f"{i}. {row['feature']}")
    print(f"   系数: {row['coefficient']:.4f} ({direction})")
    print(f"   绝对值: {row['abs_coefficient']:.4f}")

print(f"\n📊 被排除的特征 ({len(feature_names) - len(selected_features)}个):")
excluded_features = coef_df[~coef_df['selected']]['feature'].tolist()
if excluded_features:
    # 分组显示
    for i in range(0, len(excluded_features), 5):
        print("   " + ", ".join(excluded_features[i:i+5]))

print("\n💡 特征含义解释:")
print("   - 正系数特征: 值越大，越可能为恶性(M)")
print("   - 负系数特征: 值越大，越可能为良性(B)")

print("\n" + "=" * 60)
print("Lasso特征选择完成！系数绝对值>0.1的特征已筛选。")
print("=" * 60)

# 13. 使用建议
print("\n📋 后续使用建议:")
print("1. 对验证集应用相同的特征选择:")
print("   ```python")
print("   # 加载验证集")
print("   val_df = pd.read_csv(r'F:\\Project\\Breast\\splitted_data\\validation_data.csv')")
print("   X_val = val_df.iloc[:, :-2]")
print("   y_val = val_df['diagnosis']")
print("   ")
print("   # 使用相同的标准化器")
print("   scaler = joblib.load(r'F:\\Project\\Breast\\feature_selection_lasso_0.1\\scaler.pkl')")
print("   X_val_scaled = scaler.transform(X_val)")
print("   ")
print("   # 只选择相同的特征")
print("   selected_features = pd.read_csv(r'F:\\Project\\Breast\\feature_selection_lasso_0.1\\selected_features_gt_0.1.csv')")
print("   selected_features_list = selected_features['feature'].tolist()")
print("   X_val_selected = X_val[selected_features_list]")
print("   X_val_selected_scaled = X_val_scaled[:, [feature_names.index(f) for f in selected_features_list]]")
print("   ```")