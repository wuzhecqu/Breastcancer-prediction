import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt


# ------------------ 页面配置 ------------------
st.set_page_config(
    page_title="乳腺癌诊断预测与解释系统",
    page_icon="🩺",
    layout="wide"
)


# ------------------ 加载模型和解释器 (缓存) ------------------
@st.cache_resource
def load_artifacts():
    """加载模型、标准化器、SHAP解释器和特征信息"""
    try:
        model = joblib.load('lightgbm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        with open('feature_info.json', 'r', encoding='utf-8') as f:
            import json
            feature_info = json.load(f)

        # 创建SHAP解释器（针对LightGBM树模型）
        explainer = shap.TreeExplainer(model)

        # 计算基础期望值（即模型在训练数据上的平均输出）
        # 注意：为了演示，这里用训练集的一部分来计算期望值。实际部署应预计算好。
        # 这里简化处理，可以从explainer直接获取（如果模型是树模型且提供了背景数据）
        expected_value = explainer.expected_value

        return model, scaler, explainer, expected_value, feature_info
    except Exception as e:
        st.error(f"加载模型组件失败: {e}")
        return None, None, None, None, None


# 加载
model, scaler, explainer, expected_value, feature_info = load_artifacts()

# ------------------ 侧边栏：用户输入 ------------------
# 在侧边栏输入部分，确保所有数值类型一致
st.sidebar.header("🔬 输入患者特征值")

selected_features = [
    'radius_worst', 'concave points_mean', 'radius_se',
    'concavity_worst', 'area_worst', 'compactness_mean'
]

feature_inputs = {}
for feat in selected_features:
    # 确保step是浮点数
    if feat == 'radius_worst':
        min_val, max_val, default_val, step_val = 10.0, 30.0, 15.0, 0.1
    elif feat == 'concave points_mean':
        min_val, max_val, default_val, step_val = 0.0, 0.2, 0.05, 0.001
    elif feat == 'radius_se':
        min_val, max_val, default_val, step_val = 0.2, 2.0, 0.5, 0.01
    elif feat == 'concavity_worst':
        min_val, max_val, default_val, step_val = 0.0, 0.5, 0.1, 0.01
    elif feat == 'area_worst':
        min_val, max_val, default_val, step_val = 500.0, 2000.0, 800.0, 10.0  # 注意：10.0不是10
    elif feat == 'compactness_mean':
        min_val, max_val, default_val, step_val = 0.05, 0.3, 0.15, 0.001
    else:
        min_val, max_val, default_val, step_val = 0.0, 1.0, 0.5, 0.01
    
    # 显式转换为float，确保类型一致
    value = st.slider(
        label=feat,
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        step=float(step_val),  # 关键修复点
        format="%.3f" if step_val < 0.01 else "%.1f"
    )
    feature_inputs[feat] = value

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 进行诊断预测", type="primary", use_container_width=True)

# ------------------ 主页面 ------------------
st.title("🩺 乳腺癌诊断预测与可解释性分析")
st.markdown("本系统基于LightGBM模型，使用6个关键细胞核特征预测肿瘤的良恶性，并提供模型决策依据的可视化解释[citation:2]。")

# 当点击预测按钮时
if predict_button and model is not None:
    with st.spinner('正在分析特征并生成预测...'):

        # 1. 准备输入数据
        input_df = pd.DataFrame([feature_inputs])
        input_df = input_df[selected_features]  # 确保列顺序
        input_scaled = scaler.transform(input_df)

        # 2. 进行预测
        probability = model.predict(input_scaled)[0]  # 预测为恶性(M)的概率
        prediction = 1 if probability > 0.5 else 0
        prediction_label = "恶性 (M)" if prediction == 1 else "良性 (B)"

        # 3. 计算SHAP值（局部解释）
        shap_values = explainer.shap_values(input_scaled)
        # 对于二分类，通常取输出为类别1（恶性）的SHAP值
        shap_val_for_instance = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]

        # ------------------ 显示预测结果 ------------------
        st.header("📊 预测结果")
        col1, col2, col3 = st.columns(3)
        with col1:
            if prediction == 1:
                st.error(f"**预测分类: {prediction_label}**")
            else:
                st.success(f"**预测分类: {prediction_label}**")
        with col2:
            st.metric(label="**恶性概率**", value=f"{probability:.2%}")
        with col3:
            # 风险等级
            if probability < 0.2:
                risk = "低风险"
                color = "green"
            elif probability < 0.6:
                risk = "中风险"
                color = "orange"
            else:
                risk = "高风险"
                color = "red"
            st.markdown(f"**风险等级**: :{color}[{risk}]")

        # 预测概率进度条
        st.progress(float(probability), text=f"恶性概率: {probability:.2%}")

        # ------------------ 显示SHAP解释 ------------------
        st.header("🧠 模型决策解释 (SHAP)")
        st.markdown("""
        **SHAP (SHapley Additive exPlanations)** 值解释了每个特征如何影响**本次特定预测**。
        - **红色箭头**：将该样本的预测值**推高**（增加恶性可能）的特征。
        - **蓝色箭头**：将预测值**拉低**（降低恶性可能）的特征。
        - **基础值** (`base value`): 模型在训练集所有样本上预测的平均输出。
        - **输出值** (`f(x)`): 模型对当前输入样本的原始预测输出（经过Sigmoid函数转换后即得到上述恶性概率）。
        """)

        # 创建两个选项卡：力力图和特征影响
        tab1, tab2 = st.tabs(["📈 SHAP 力力图 (Force Plot)", "📊 特征影响分解"])

        with tab1:
            st.subheader("局部解释力力图")
            st.markdown(f"基础值 (所有患者的平均预测): **{expected_value[1]:.4f}**")

            # 使用SHAP生成力力图（matplotlib版本，更适合Streamlit）
            plt.figure(figsize=(10, 4))
            shap.force_plot(
                base_value=expected_value[1],  # 类别1的基础期望值
                shap_values=shap_val_for_instance,
                features=input_df.iloc[0],
                feature_names=selected_features,
                matplotlib=True,
                show=False,
                text_rotation=15
            )
            plt.tight_layout()
            st.pyplot(plt)
            plt.clf()  # 清除图形

            st.caption("""
            **解读提示**：力的长度代表特征影响的大小，方向（红/蓝）代表影响的方向。所有特征影响力的总和将预测值从“基础值”推到了最终的“输出值”。
            """)

        with tab2:
            st.subheader("特征影响值明细表")
            # 创建影响值DataFrame
            impact_df = pd.DataFrame({
                '特征': selected_features,
                'SHAP值 (影响力)': shap_val_for_instance,
                '特征值': input_df.iloc[0].values,
                '影响方向': ['推高风险' if v > 0 else '降低风险' for v in shap_val_for_instance]
            })
            impact_df = impact_df.sort_values('SHAP值 (影响力)', key=abs, ascending=False)

            st.dataframe(
                impact_df.style.format({'SHAP值 (影响力)': '{:.4f}', '特征值': '{:.4f}'}),
                use_container_width=True
            )

            # 可选：绘制条形图
            fig, ax = plt.subplots(figsize=(9, 5))
            colors = ['tomato' if x > 0 else 'dodgerblue' for x in impact_df['SHAP值 (影响力)']]
            y_pos = np.arange(len(impact_df))
            ax.barh(y_pos, impact_df['SHAP值 (影响力)'], color=colors)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(impact_df['特征'])
            ax.set_xlabel('SHAP值 (对恶性概率的影响)')
            ax.set_title('各特征对本次预测的贡献')
            ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            st.pyplot(fig)

        # ------------------ 临床建议 ------------------
        st.header("💡 临床解读与建议")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("关键风险驱动因素")
            # 找出影响力最大的两个特征（正负各一个）
            top_positive = impact_df[impact_df['SHAP值 (影响力)'] > 0].nlargest(1, 'SHAP值 (影响力)')
            top_negative = impact_df[impact_df['SHAP值 (影响力)'] < 0].nsmallest(1, 'SHAP值 (影响力)')

            if not top_positive.empty:
                feat = top_positive.iloc[0]['特征']
                val = top_positive.iloc[0]['特征值']
                st.markdown(f"✅ **主要风险因素**: `{feat}` = {val:.3f}")
                st.markdown(f"   - 该值高于典型良性样本，显著增加了恶性风险。")

            if not top_negative.empty:
                feat = top_negative.iloc[0]['特征']
                val = top_negative.iloc[0]['特征值']
                st.markdown(f"✅ **主要良性指标**: `{feat}` = {val:.3f}")
                st.markdown(f"   - 该值在良性范围内，有助于降低恶性评分。")

        with col_b:
            st.subheader("后续步骤建议")
            if prediction == 1 or probability > 0.3:
                st.warning("""
                **建议进行进一步临床评估：**
                - 建议进行穿刺活检以明确病理诊断。
                - 结合影像学报告（如乳腺X线摄影、超声）进行综合判断。
                - 咨询肿瘤科或乳腺外科专家。
                """)
            else:
                st.info("""
                **建议定期随访监测：**
                - 建议根据年龄和风险因素进行常规乳腺癌筛查。
                - 保持健康生活方式，注意乳房自查。
                - 如有任何新发症状，及时就医。
                """)

        # ------------------ 特征值对比 ------------------
        st.header("📋 输入特征值汇总")
        st.dataframe(input_df.T.rename(columns={0: '输入值'}), use_container_width=True)

# 初始状态或无模型时
elif not predict_button:
    st.info("👈 请在左侧侧边栏输入特征值，然后点击 **'进行诊断预测'** 按钮。")

    # 显示特征说明表
    if feature_info and 'feature_importance' in feature_info:
        st.subheader("模型使用的6个关键特征及其重要性")
        importance_df = pd.DataFrame(feature_info['feature_importance']).sort_values('importance', ascending=False)
        st.dataframe(importance_df, use_container_width=True)

        # 特征含义解释
        with st.expander("📚 点击查看特征临床意义"):
            st.markdown("""
            | 特征 | 临床意义 |
            |------|----------|
            | `radius_worst` | **最差半径**：肿块最大截面的半径，是最重要的恶性指标之一。值越大，恶性可能性通常越高。 |
            | `area_worst` | **最差面积**：与半径相关，但此处系数为负，可能指示某些特定形态。 |
            | `concave points_mean` | **平均凹点数量**：细胞核轮廓中凹点的平均数量。凹点越多、越深，越可能是恶性。 |
            | `compactness_mean` | **平均紧致度**：细胞核形状接近圆形的程度（周长² / 面积）。值越高越不规则，常与恶性相关。 |
            | `radius_se` | **半径标准误**：细胞核半径的变异程度。恶性细胞通常大小更不一致。 |
            | `concavity_worst` | **最差凹度**：细胞核轮廓中凹陷部分的严重程度。最大值越大，恶性可能性越高。 |
            """)
else:
    st.warning("⚠️ 模型加载失败，请确保 `lightgbm_model.pkl`, `scaler.pkl`, `feature_info.json` 文件已正确放置。")

# 页脚
st.markdown("---")
st.caption("""
*注意：本工具旨在辅助临床决策，不能替代执业医师的专业诊断。所有预测结果均应结合完整的临床资料进行解读[citation:2]。*

""")
