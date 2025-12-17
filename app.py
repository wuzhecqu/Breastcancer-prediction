import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

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
        # 加载模型和标准化器
        model = joblib.load('lightgbm_model.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # 加载特征信息
        try:
            with open('feature_info.json', 'r', encoding='utf-8') as f:
                feature_info = json.load(f)
            selected_features = feature_info.get('selected_features', [
                'radius_worst', 'concave points_mean', 'radius_se',
                'concavity_worst', 'area_worst', 'compactness_mean'
            ])
        except:
            # 如果feature_info.json不存在或格式错误，使用默认特征
            selected_features = [
                'radius_worst', 'concave points_mean', 'radius_se',
                'concavity_worst', 'area_worst', 'compactness_mean'
            ]
            feature_info = {'selected_features': selected_features}
        
        # 创建背景数据用于SHAP解释器
        # 使用简化的小样本背景数据
        background = np.zeros((5, len(selected_features)))
        background_df = pd.DataFrame(background, columns=selected_features)
        background_scaled = scaler.transform(background_df)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model, background_scaled)
        
        # 获取期望值 - 处理不同格式
        expected_val = explainer.expected_value
        
        # 调试信息
        print(f"Expected value type: {type(expected_val)}")
        print(f"Expected value shape: {np.shape(expected_val)}")
        print(f"Expected value: {expected_val}")
        
        # 处理expected_value的格式
        if isinstance(expected_val, np.ndarray):
            if len(expected_val) == 2:
                # 二分类，返回两个值的情况
                base_value = float(expected_val[1])  # 恶性类的基础值
            elif len(expected_val) == 1:
                # 只有一个值的情况
                base_value = float(expected_val[0])
            else:
                base_value = float(expected_val[0])
        else:
            # 单个标量值
            base_value = float(expected_val)
        
        print(f"Base value for SHAP: {base_value}")
        
        return model, scaler, explainer, base_value, feature_info, selected_features
        
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        # 返回默认值
        selected_features = [
            'radius_worst', 'concave points_mean', 'radius_se',
            'concavity_worst', 'area_worst', 'compactness_mean'
        ]
        return None, None, None, 0.0, {}, selected_features

# 加载模型组件
model, scaler, explainer, base_value, feature_info, selected_features = load_artifacts()

# ------------------ 侧边栏：用户输入 ------------------
st.sidebar.header("🔬 输入患者特征值")

# 为每个特征创建输入滑块
feature_inputs = {}
for feat in selected_features:
    # 根据特征定义合理的范围和默认值
    if feat == 'radius_worst':
        min_val, max_val, default_val, step_val = 10.0, 30.0, 15.0, 0.1
    elif feat == 'concave points_mean':
        min_val, max_val, default_val, step_val = 0.0, 0.2, 0.05, 0.001
    elif feat == 'radius_se':
        min_val, max_val, default_val, step_val = 0.2, 2.0, 0.5, 0.01
    elif feat == 'concavity_worst':
        min_val, max_val, default_val, step_val = 0.0, 0.5, 0.1, 0.01
    elif feat == 'area_worst':
        min_val, max_val, default_val, step_val = 500.0, 2000.0, 800.0, 10.0
    elif feat == 'compactness_mean':
        min_val, max_val, default_val, step_val = 0.05, 0.3, 0.15, 0.001
    else:
        min_val, max_val, default_val, step_val = 0.0, 1.0, 0.5, 0.01
    
    # 创建滑块
    value = st.sidebar.slider(
        label=f"{feat}",
        min_value=float(min_val),
        max_value=float(max_val),
        value=float(default_val),
        step=float(step_val),
        help=f"范围: {min_val} - {max_val}"
    )
    feature_inputs[feat] = value

st.sidebar.markdown("---")
predict_button = st.sidebar.button("🚀 进行诊断预测", type="primary", use_container_width=True)

# ------------------ 主页面 ------------------
st.title("🩺 乳腺癌诊断预测与可解释性分析")
st.markdown("基于LightGBM模型的乳腺癌良恶性预测系统，提供SHAP可解释性分析。")

if predict_button and model is not None:
    with st.spinner('正在分析特征并生成预测...'):
        try:
            # 1. 准备输入数据
            input_df = pd.DataFrame([feature_inputs])
            input_df = input_df[selected_features]  # 确保列顺序
            input_scaled = scaler.transform(input_df)
            
            # 2. 进行预测
            # LightGBM预测恶性概率
            probability = model.predict_proba(input_scaled)[0][1]  # 获取恶性类概率
            prediction = 1 if probability > 0.5 else 0
            prediction_label = "恶性 (M)" if prediction == 1 else "良性 (B)"
            
            # 3. 计算SHAP值
            shap_values = explainer.shap_values(input_scaled)
            
            # 调试信息
            print(f"SHAP values type: {type(shap_values)}")
            print(f"SHAP values length if list: {len(shap_values) if isinstance(shap_values, list) else 'Not list'}")
            
            # 处理SHAP值的格式 - 根据警告信息，SHAP值现在是列表格式
            if isinstance(shap_values, list):
                if len(shap_values) == 2:
                    # 二分类，有两个数组 [良性SHAP值, 恶性SHAP值]
                    shap_val_for_instance = shap_values[1][0]  # 恶性类的SHAP值
                elif len(shap_values) == 1:
                    # 只有一个数组
                    shap_val_for_instance = shap_values[0][0]
                else:
                    # 其他情况
                    shap_val_for_instance = shap_values[0][0]
            else:
                # 不是列表，直接使用
                shap_val_for_instance = shap_values[0]
            
            print(f"SHAP values for instance: {shap_val_for_instance}")
            print(f"SHAP values shape: {np.shape(shap_val_for_instance)}")
            
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
            
            # ------------------ SHAP力力图 ------------------
            st.header("🧠 模型决策解释 (SHAP力力图)")
            st.markdown(f"**基础值**: {base_value:.4f} (模型在训练数据上的平均预测)")
            
            # 创建力力图 - 使用安全的绘图方式
            try:
                fig, ax = plt.subplots(figsize=(10, 4))
                
                # 使用force_plot
                shap.force_plot(
                    base_value=base_value,
                    shap_values=shap_val_for_instance,
                    features=input_df.iloc[0],
                    feature_names=selected_features,
                    matplotlib=True,
                    show=False,
                    text_rotation=15
                )
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.clf()
                
                st.caption("""
                **解读**：红色特征将预测推向恶性，蓝色特征将预测推向良性。
                所有特征影响力的总和将预测值从"基础值"推到了最终的预测概率。
                """)
                
            except Exception as e:
                st.warning(f"无法生成SHAP力力图: {e}")
                st.info("使用替代的可视化方式...")
            
            # ------------------ 特征影响分析 ------------------
            st.header("📈 特征影响分析")
            
            # 创建特征影响DataFrame
            impact_df = pd.DataFrame({
                '特征': selected_features,
                'SHAP值': shap_val_for_instance,
                '特征值': input_df.iloc[0].values,
                '绝对影响': np.abs(shap_val_for_instance),
                '影响方向': ['增加风险' if v > 0 else '降低风险' for v in shap_val_for_instance]
            })
            
            # 按绝对影响排序
            impact_df = impact_df.sort_values('绝对影响', ascending=False)
            
            # 显示表格
            st.subheader("特征影响明细表")
            st.dataframe(
                impact_df[['特征', '特征值', 'SHAP值', '影响方向']].style.format({
                    '特征值': '{:.3f}',
                    'SHAP值': '{:.4f}'
                }),
                use_container_width=True
            )
            
            # ------------------ 可视化特征影响 ------------------
            st.subheader("特征影响力条形图")
            
            fig = go.Figure()
            
            # 添加条形图
            colors = ['red' if x > 0 else 'blue' for x in impact_df['SHAP值']]
            
            fig.add_trace(go.Bar(
                x=impact_df['SHAP值'],
                y=impact_df['特征'],
                orientation='h',
                marker_color=colors,
                text=[f'{x:.4f}' for x in impact_df['SHAP值']],
                textposition='auto',
                name='SHAP值'
            ))
            
            fig.update_layout(
                title='各特征对预测的影响 (SHAP值)',
                xaxis_title='SHAP值 (对恶性概率的影响)',
                yaxis_title='特征',
                height=400,
                showlegend=False
            )
            
            # 添加零线
            fig.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
            
            st.plotly_chart(fig, use_container_width=True)
            
            # ------------------ 临床解读 ------------------
            st.header("💡 临床解读")
            
            # 找出最重要的风险因素和保护因素
            top_risk = impact_df[impact_df['SHAP值'] > 0].head(2)
            top_protective = impact_df[impact_df['SHAP值'] < 0].head(2)
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.subheader("主要风险驱动因素")
                if not top_risk.empty:
                    for _, row in top_risk.iterrows():
                        st.markdown(f"**{row['特征']}** = {row['特征值']:.3f}")
                        st.markdown(f"贡献: **+{row['SHAP值']:.4f}**")
                        st.markdown("该特征值显著增加了恶性风险")
                else:
                    st.info("未识别出明显的风险因素")
            
            with col_b:
                st.subheader("主要良性指标")
                if not top_protective.empty:
                    for _, row in top_protective.iterrows():
                        st.markdown(f"**{row['特征']}** = {row['特征值']:.3f}")
                        st.markdown(f"贡献: **{row['SHAP值']:.4f}**")
                        st.markdown("该特征值有助于降低恶性风险")
                else:
                    st.info("未识别出明显的保护因素")
            
            # 建议
            st.subheader("后续步骤建议")
            if probability > 0.7:
                st.warning("""
                **高风险 - 强烈建议进一步检查：**
                1. 立即咨询乳腺外科或肿瘤科专家
                2. 考虑进行穿刺活检以明确诊断
                3. 进行乳腺超声或钼靶检查
                4. 定期随访监测
                """)
            elif probability > 0.3:
                st.warning("""
                **中风险 - 建议进一步评估：**
                1. 咨询专科医生进行评估
                2. 考虑进行影像学检查
                3. 密切观察，3-6个月后复查
                """)
            else:
                st.info("""
                **低风险 - 建议常规随访：**
                1. 按照常规筛查计划进行
                2. 保持健康生活方式
                3. 定期进行乳房自查
                4. 如有变化及时就医
                """)
            
            # ------------------ 特征值汇总 ------------------
            with st.expander("📋 查看详细的输入特征值"):
                st.dataframe(input_df.T.rename(columns={0: '输入值'}))
                
        except Exception as e:
            st.error(f"预测过程中出现错误: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            st.info("请检查模型文件和输入数据是否正确。")

elif not predict_button:
    # 初始页面
    st.info("👈 请在左侧侧边栏输入特征值，然后点击 **'进行诊断预测'** 按钮。")
    
    # 显示特征说明
    if feature_info and 'selected_features' in feature_info:
        st.subheader("模型使用的关键特征")
        st.write("以下6个特征用于预测乳腺癌良恶性：")
        for i, feat in enumerate(selected_features, 1):
            st.write(f"{i}. **{feat}**")

else:
    st.error("⚠️ 模型加载失败，请确保模型文件存在并格式正确。")

# 页脚
st.markdown("---")
st.caption("""
*注意：本工具旨在辅助临床决策，不能替代执业医师的专业诊断。所有预测结果均应结合完整的临床资料进行解读。*
""")
