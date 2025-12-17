import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# 1. 加载数据
file_path = r'F:\Project\Breast\breast-cancer.csv'
df = pd.read_csv(file_path)

print("=" * 60)
print("数据基本信息:")
print("=" * 60)
print(f"数据形状: {df.shape}")
print(f"数据列名:\n{df.columns.tolist()}")
print(f"\n目标变量分布:")
print(df['diagnosis'].value_counts())

# 2. 分离特征和目标变量
X = df.iloc[:, 1:]  # 所有特征列（从第二列开始）
y = df.iloc[:, 0]   # 第一列：diagnosis

print(f"\n特征矩阵形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 3. 将目标变量编码为数值（B=0, M=1）
le = LabelEncoder()
y_encoded = le.fit_transform(y)  # B->0, M->1
print(f"\n目标变量编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 4. 按8:2划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded,
    test_size=0.2,              # 20%作为验证集
    random_state=42,            # 随机种子
    stratify=y_encoded          # 保持类别比例一致
)

# 5. 同时保存原始标签和编码标签
train_df = pd.DataFrame(X_train, columns=X.columns)
train_df['diagnosis'] = y_train
train_df['diagnosis_original'] = le.inverse_transform(y_train)  # 原始标签

val_df = pd.DataFrame(X_val, columns=X.columns)
val_df['diagnosis'] = y_val
val_df['diagnosis_original'] = le.inverse_transform(y_val)  # 原始标签

# 6. 创建保存目录
save_dir = r'F:\Project\Breast\\'
os.makedirs(save_dir, exist_ok=True)

# 7. 保存为CSV文件
train_path = os.path.join(save_dir, 'train_data.csv')
val_path = os.path.join(save_dir, 'validation_data.csv')

train_df.to_csv(train_path, index=False)
val_df.to_csv(val_path, index=False)

print("\n" + "=" * 60)
print("数据集划分结果:")
print("=" * 60)
print(f"训练集: {X_train.shape[0]} 个样本 (80%)")
print(f"验证集: {X_val.shape[0]} 个样本 (20%)")

print(f"\n训练集类别分布:")
print(f"良性(B): {np.sum(y_train==0)} 个 ({np.sum(y_train==0)/len(y_train):.1%})")
print(f"恶性(M): {np.sum(y_train==1)} 个 ({np.sum(y_train==1)/len(y_train):.1%})")

print(f"\n验证集类别分布:")
print(f"良性(B): {np.sum(y_val==0)} 个 ({np.sum(y_val==0)/len(y_val):.1%})")
print(f"恶性(M): {np.sum(y_val==1)} 个 ({np.sum(y_val==1)/len(y_val):.1%})")

print(f"\n文件保存位置:")
print(f"训练集: {train_path}")
print(f"验证集: {val_path}")