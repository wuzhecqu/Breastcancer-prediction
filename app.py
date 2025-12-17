# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import json
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="乳腺癌诊断预测系统",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("🏥 乳腺癌诊断预测系统")
st.markdown("""
基于LightGBM模型的乳腺癌良恶性预测系统，提供SHAP可解释性分析。
使用6个关键特征进行预测，模型在验证集上准确率达到97.8%。
""")

# 设置6个关键特征
selected_features = [
    'radius_worst',
    'concave points_mean', 
    'radius_se',
    'concavity_worst',
    'area_worst',
    'compactness_mean'
]

# 特征描述
feature_descriptions = {
    'radius_worst': '肿瘤最差半径 - 最重要的恶性指标',
    'concave points_mean': '平均凹点数量 - 凹点越多恶性可能性越大',
    'radius_se': '半径标准误 - 反映细胞大小的一致性',
    'concavity_worst': '最差凹度 - 凹度越大恶性可能性越高',
    'area_worst': '最差面积 - 在某些情况下与良性相关',
    'compactness_mean': '平均紧致度 - 与细胞形状规则性相关'
}

# ====================== 1. 加载模型 ======================
@st.cache_resource
def load_model():
    """加载模型和预处理对象"""
    try:
        # 加载模型
        model = joblib.load('lightgbm_model.pkl')
        
        # 加载标准化器
        scaler = joblib.load('scaler.pkl')
        
        # 加载特征信息
        try:
            with open('feature_info.json', 'r', encoding='utf-8') as f:
                feature_info = json.load(f)
        except:
            feature_info = {
                'selected_features': selected_features,
                'feature_importance': [
                    {"feature": "radius_worst", "importance": 0.5081},
                    {"feature": "area_worst", "importance": 0.3233},
                    {"feature": "concave points_mean", "importance": 0.1373},
                    {"feature": "radius_se", "importance": 0.1327},
                    {"feature": "compactness_mean", "importance": 0.1465},
                    {"feature": "concavity_worst", "importance": 0.1030}
                ]
            }
        
        st.sidebar.success("✅ 模型加载成功")
        return model, scaler, feature_info
        
    except Exception as e:
        st.sidebar.error(f"❌ 模型加载失败: {e}")
        return None, None, {}

# 加载模型
model, scaler, feature_info = load_model()

# ====================== 2. 侧边栏导航 ======================
st.sidebar.header("导航")
option = st.sidebar.selectbox(
    "选择页面",
    ["🔍 单样本预测", "📊 特征分析", "ℹ️ 使用说明"]
)

# ====================== 3. 单样本预测页面 ======================
if option == "🔍 单样本预测":
    st.header("单样本预测")
    st.markdown("请输入患者的6个关键特征值进行乳腺癌诊断预测。")
    
    # 创建两列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("正相关特征 (值越大越可能是恶性)")
        radius_worst = st.slider(
            "radius_worst (最差半径)", 
            min_value=5.0, max_value=40.0, value=15.0, step=0.1,
            help="典型范围: 6-36, 平均值: 16-20"
        )
        
        concave_points_mean = st.slider(
            "concave points_mean (平均凹点)", 
            min_value=0.0, max_value=0.3, value=0.05, step=0.001,
            help="典型范围: 0.0-0.2, 平均值: 0.05-0.08"
        )
        
        radius_se = st.slider(
            "radius_se (半径标准误)", 
            min_value=0.1, max_value=3.0, value=0.5, step=0.01,
            help="典型范围: 0.1-3.0, 平均值: 0.3-0.6"
        )
        
        concavity_worst = st.slider(
            "concavity_worst (最差凹度)", 
            min_value=0.0, max_value=1.0, value=0.1, step=0.01,
            help="典型范围: 0.0-0.8, 平均值: 0.1-0.2"
        )
    
    with col2:
        st.subheader("负相关特征 (值越大越可能是良性)")
        area_worst = st.slider(
            "area_worst (最差面积)", 
            min_value=200.0, max_value=3000.0, value=800.0, step=10.0,
            help="典型范围: 200-2500, 平均值: 800-1200"
        )
        
        compactness_mean = st.slider(
            "compactness_mean (平均紧致度)", 
            min_value=0.02, max_value=0.4, value=0.15, step=0.001,
            help="典型范围: 0.02-0.35, 平均值: 0.1-0.2"
        )
    
    if st.button("🚀 开始预测", type="primary", use_container_width=True):
        if model is None:
            st.error("模型未加载成功，请检查模型文件")
        else:
            with st.spinner("正在分析中..."):
                # 准备输入数据
                input_data = {
                    'radius_worst': radius_worst,
                    'concave points_mean': concave_points_mean,
                    'radius_se': radius_se,
                    'concavity_worst': concavity_worst,
                    'area_worst': area_worst,
                    'compactness_mean': compactness_mean
                }
                
                input_df = pd.DataFrame([input_data])
                input_scaled = scaler.transform(input_df)
                
                # 预测
                try:
                    # 尝试不同的预测方法
                    if hasattr(model, 'predict_proba'):
                        probability = model.predict_proba(input_scaled)[0, 1]
                        print(f"使用predict_proba，概率: {probability}")
                    else:
                        # 对于原生LightGBM Booster
                        try:
                            raw_pred = model.predict(input_scaled, raw_score=True)
                            if isinstance(raw_pred, np.ndarray) and len(raw_pred) > 0:
                                raw_score = float(raw_pred[0])
                            else:
                                raw_score = float(raw_pred)
                            probability = 1 / (1 + np.exp(-raw_score))
                            print(f"使用raw_score转换，概率: {probability}")
                        except:
                            # 备选方法
                            pred = model.predict(input_scaled)
                            probability = float(pred[0]) if pred[0] <= 1 else 0.5
                            print(f"使用直接预测，概率: {probability}")
                except Exception as pred_error:
                    st.error(f"预测失败: {pred_error}")
                    probability = 0.5
                
                # 确保概率在0-1之间
                probability = max(0.0, min(1.0, probability))
                
                prediction = 1 if probability > 0.5 else 0
                prediction_label = "恶性 (M)" if prediction == 1 else "良性 (B)"
                
                # 显示预测结果
                st.subheader("📊 预测结果")
                
                # 创建三列布局显示结果
                col_result1, col_result2, col_result3 = st.columns(3)
                
                with col_result1:
                    if prediction == 1:
                        st.error(f"**诊断结果: {prediction_label}**")
                    else:
                        st.success(f"**诊断结果: {prediction_label}**")
                
                with col_result2:
                    st.metric("恶性概率", f"{probability:.2%}")
                
                with col_result3:
                    # 风险等级
                    if probability < 0.3:
                        risk_level = "低风险"
                        color = "green"
                    elif probability < 0.7:
                        risk_level = "中风险" 
                        color = "orange"
                    else:
                        risk_level = "高风险"
                        color = "red"
                    st.markdown(f"**风险等级**: :{color}[{risk_level}]")
                
                # 风险仪表盘
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probability * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "恶性风险 (%)"},
                    gauge={
                        'axis': {'range': [0, 100]},
                        'bar': {'color': "darkred"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 70], 'color': "lightyellow"},
                            {'range': [70, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "black", 'width': 4},
                            'thickness': 0.75,
                            'value': 50
                        }
                    }
                ))
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
                
                # ================= SHAP可解释性分析 =================
                st.subheader("🧠 模型决策解释 (SHAP分析)")
                
                try:
                    # 创建SHAP解释器
                    background = np.zeros((10, len(selected_features)))
                    background_df = pd.DataFrame(background, columns=selected_features)
                    background_scaled = scaler.transform(background_df)
                    
                    explainer = shap.TreeExplainer(model, background_scaled)
                    
                    # 计算SHAP值
                    shap_values = explainer.shap_values(input_scaled)
                    
                    # 处理SHAP输出格式
                    if isinstance(shap_values, list):
                        if len(shap_values) >= 2:
                            # 二分类，取恶性类的SHAP值
                            shap_vals = shap_values[1][0]
                        else:
                            shap_vals = shap_values[0][0]
                    else:
                        shap_vals = shap_values[0]
                    
                    # 创建SHAP值数据框
                    shap_df = pd.DataFrame({
                        '特征': selected_features,
                        'SHAP值': shap_vals,
                        '特征值': input_df.iloc[0].values,
                        '影响方向': ['增加风险' if v > 0 else '降低风险' for v in shap_vals]
                    })
                    
                    # 按绝对值排序
                    shap_df['绝对值'] = np.abs(shap_df['SHAP值'])
                    shap_df = shap_df.sort_values('绝对值', ascending=False)
                    
                    # 显示SHAP表格
                    st.dataframe(
                        shap_df[['特征', '特征值', 'SHAP值', '影响方向']].style.format({
                            '特征值': '{:.3f}',
                            'SHAP值': '{:.4f}'
                        }),
                        use_container_width=True
                    )
                    
                    # 可视化SHAP值
                    fig_shap = px.bar(shap_df, 
                                     x='SHAP值', 
                                     y='特征',
                                     orientation='h',
                                     color='影响方向',
                                     color_discrete_map={
                                         '增加风险': '#EF553B', 
                                         '降低风险': '#636EFA'
                                     },
                                     title='各特征对本次预测的影响 (SHAP值)')
                    
                    fig_shap.add_vline(x=0, line_width=1, line_dash="dash", line_color="black")
                    fig_shap.update_layout(height=400)
                    st.plotly_chart(fig_shap, use_container_width=True)
                    
                    # 尝试生成力力图
                    # try:
                    #     st.subheader("SHAP力力图")
                    #     # 获取基础值
                    #     expected_value = explainer.expected_value
                    #     if isinstance(expected_value, np.ndarray):
                    #         if len(expected_value) >= 2:
                    #             base_value = float(expected_value[1])
                    #         else:
                    #             base_value = float(expected_value[0])
                    #     else:
                    #         base_value = float(expected_value)
                        
                    #     # 创建力力图
                    #     fig, ax = plt.subplots(figsize=(10, 4))
                    #     shap.force_plot(
                    #         base_value=base_value,
                    #         shap_values=shap_vals,
                    #         features=input_df.iloc[0],
                    #         feature_names=selected_features,
                    #         matplotlib=True,
                    #         show=False
                    #     )
                    #     plt.tight_layout()
                    #     st.pyplot(fig)
                    #     plt.clf()
                        
                    #     st.caption("""
                    #     **力力图解读**:
                    #     - 红色箭头: 增加恶性风险的特征
                    #     - 蓝色箭头: 降低恶性风险的特征  
                    #     - 基础值: 模型在所有患者上的平均预测
                    #     - 最终值: 当前患者的预测概率
                    #     """)
                        
                    # except Exception as e:
                    #     st.info("力力图生成跳过，SHAP条形图已提供完整的特征影响分析")
                    
                    # 临床解读
                    st.subheader("💡 临床解读")
                    
                    # 找出最重要的风险和保护因素
                    top_risk = shap_df[shap_df['SHAP值'] > 0].head(2)
                    top_protective = shap_df[shap_df['SHAP值'] < 0].head(2)
                    
                    col_interpret1, col_interpret2 = st.columns(2)
                    
                    with col_interpret1:
                        st.markdown("**主要风险因素:**")
                        if not top_risk.empty:
                            for _, row in top_risk.iterrows():
                                st.markdown(f"**{row['特征']}** = {row['特征值']:.3f}")
                                st.markdown(f"- SHAP值: **+{row['SHAP值']:.4f}**")
                                st.markdown(f"- 解释: {feature_descriptions.get(row['特征'], '')}")
                        else:
                            st.info("无显著风险因素")
                    
                    with col_interpret2:
                        st.markdown("**主要保护因素:**")
                        if not top_protective.empty:
                            for _, row in top_protective.iterrows():
                                st.markdown(f"**{row['特征']}** = {row['特征值']:.3f}")
                                st.markdown(f"- SHAP值: **{row['SHAP值']:.4f}**")
                                st.markdown(f"- 解释: {feature_descriptions.get(row['特征'], '')}")
                        else:
                            st.info("无显著保护因素")
                    
                    # 临床建议
                    st.subheader("📋 临床建议")
                    if probability > 0.7:
                        st.warning("""
                        **高风险 (恶性概率 > 70%)**:
                        1. **立即就诊**: 建议尽快咨询乳腺外科或肿瘤科专家
                        2. **进一步检查**: 考虑进行穿刺活检明确诊断
                        3. **影像学检查**: 建议乳腺超声、钼靶或MRI检查
                        4. **密切随访**: 定期复查监测病情变化
                        """)
                    elif probability > 0.3:
                        st.warning("""
                        **中风险 (恶性概率 30%-70%)**:
                        1. **专科咨询**: 建议咨询乳腺专科医生
                        2. **定期复查**: 建议3-6个月后复查
                        3. **生活方式**: 保持健康生活方式，避免压力
                        4. **自我监测**: 定期进行乳房自查
                        """)
                    else:
                        st.success("""
                        **低风险 (恶性概率 < 30%)**:
                        1. **常规筛查**: 按照年龄指南进行常规乳腺癌筛查
                        2. **健康生活**: 保持均衡饮食，适度运动
                        3. **定期自查**: 每月进行乳房自查
                        4. **及时就医**: 如有新发症状及时就诊
                        """)
                    
                except Exception as e:
                    st.error(f"SHAP分析失败: {e}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("模型预测功能正常，但可解释性分析暂时不可用")

# ====================== 4. 特征分析页面 ======================
elif option == "📊 特征分析":
    st.header("特征分析")
    
    tab1, tab2 = st.tabs(["📈 特征重要性", "ℹ️ 特征说明"])
    
    with tab1:
        st.subheader("特征重要性分析")
        
        # 方法1：从feature_info.json获取重要性
        if 'feature_importance' in feature_info:
            importance_data = feature_info['feature_importance']
            importance_df = pd.DataFrame(importance_data)
            importance_df = importance_df.sort_values('importance', ascending=True)
            
            fig = px.bar(importance_df, 
                         x='importance', 
                         y='feature',
                         orientation='h',
                         title="特征重要性排序 (基于Lasso系数)",
                         color='importance',
                         color_continuous_scale='Viridis')
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("""
            **特征重要性说明**:
            - 重要性基于Lasso回归的系数绝对值计算
            - 系数绝对值越大，特征对预测的影响越大
            - 正系数表示与恶性正相关，负系数表示与恶性负相关
            """)
        
        # 方法2：尝试从LightGBM Booster获取重要性
        elif model is not None:
            try:
                # LightGBM Booster获取特征重要性的正确方法
                if hasattr(model, 'feature_importance'):
                    # 对于原生Booster
                    importances = model.feature_importance(importance_type='gain')
                elif hasattr(model, 'feature_importances_'):
                    # 对于scikit-learn接口
                    importances = model.feature_importances_
                else:
                    # 尝试其他方法
                    try:
                        importances = model.booster_.feature_importance(importance_type='gain')
                    except:
                        # 使用固定重要性值
                        importances = np.ones(len(selected_features))
                
                importance_df = pd.DataFrame({
                    'feature': selected_features,
                    'importance': importances
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(importance_df, 
                             x='importance', 
                             y='feature',
                             orientation='h',
                             title="LightGBM特征重要性",
                             color='importance',
                             color_continuous_scale='Viridis')
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.warning(f"无法获取特征重要性: {e}")
                
                # 显示默认重要性
                st.info("显示基于Lasso回归的默认特征重要性")
                default_importance = {
                    'radius_worst': 0.5081,
                    'area_worst': 0.3233,
                    'concave points_mean': 0.1373,
                    'radius_se': 0.1327,
                    'compactness_mean': 0.1465,
                    'concavity_worst': 0.1030
                }
                
                importance_df = pd.DataFrame({
                    'feature': list(default_importance.keys()),
                    'importance': list(default_importance.values())
                }).sort_values('importance', ascending=True)
                
                fig = px.bar(importance_df, 
                             x='importance', 
                             y='feature',
                             orientation='h',
                             title="特征重要性 (基于Lasso系数)",
                             color='importance',
                             color_continuous_scale='Viridis')
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("特征重要性信息不可用")
    
    with tab2:
        st.subheader("特征详细说明")
        
        # 创建特征说明表格
        feature_info_data = []
        for feature in selected_features:
            feature_info_data.append({
                '特征': feature,
                '描述': feature_descriptions.get(feature, '暂无描述'),
                '与恶性的关系': '正相关' if feature in ['radius_worst', 'concave points_mean', 'radius_se', 'concavity_worst'] else '负相关',
                '典型范围': {
                    'radius_worst': '6.0-36.0',
                    'concave points_mean': '0.0-0.2',
                    'radius_se': '0.1-3.0',
                    'concavity_worst': '0.0-0.8',
                    'area_worst': '200-2500',
                    'compactness_mean': '0.02-0.35'
                }.get(feature, '未知'),
                'Lasso系数': {
                    'radius_worst': '+0.508',
                    'concave points_mean': '+0.137',
                    'radius_se': '+0.133',
                    'concavity_worst': '+0.103',
                    'area_worst': '-0.323',
                    'compactness_mean': '-0.147'
                }.get(feature, '未知')
            })
        
        feature_info_df = pd.DataFrame(feature_info_data)
        st.dataframe(feature_info_df, use_container_width=True)
        
        # 详细说明
        st.markdown("""
        ### 🎯 特征临床意义详解
        
        1. **radius_worst (最差半径)**
           - **Lasso系数**: +0.508 (最重要的恶性指标)
           - **临床意义**: 肿瘤在多个切片中的最大半径
           - **恶性特征**: 恶性肿瘤通常生长不规则，半径更大
           - **参考值**: 
             - 良性: 通常 < 15mm
             - 恶性: 通常 > 20mm
        
        2. **concave points_mean (平均凹点数量)**
           - **Lasso系数**: +0.137
           - **临床意义**: 细胞核轮廓中凹点的平均数量
           - **恶性特征**: 恶性细胞核膜不规则，凹点更多
           - **解释**: 凹点是细胞核膜内陷形成的凹陷
        
        3. **radius_se (半径标准误)**
           - **Lasso系数**: +0.133
           - **临床意义**: 细胞核半径的标准误差
           - **恶性特征**: 恶性细胞大小不一，标准误更大
           - **生物学意义**: 反映细胞异质性
        
        4. **concavity_worst (最差凹度)**
           - **Lasso系数**: +0.103
           - **临床意义**: 细胞核轮廓中最深的凹陷程度
           - **恶性特征**: 恶性细胞核膜凹陷更深
        
        5. **area_worst (最差面积)**
           - **Lasso系数**: -0.323 (最强的良性指标)
           - **临床意义**: 肿瘤最大截面的面积
           - **注意**: 虽然面积通常与大小正相关，但在此模型中为负系数
           - **可能解释**: 某些大面积的肿瘤可能是良性增生
        
        6. **compactness_mean (平均紧致度)**
           - **Lasso系数**: -0.147
           - **临床意义**: 周长²/面积，衡量形状接近圆形的程度
           - **恶性特征**: 恶性细胞通常更不规则（紧致度更高）
           - **模型观察**: 在此数据集中与良性相关
        """)

# ====================== 5. 使用说明页面 ======================
elif option == "ℹ️ 使用说明":
    st.header("使用说明")
    
    st.markdown("""
    ## 📖 乳腺癌诊断预测系统使用指南
    
    ### 1. 系统概述
    本系统基于LightGBM机器学习模型，使用威斯康星乳腺癌数据集训练。
    通过6个关键细胞核特征预测肿瘤的良恶性，并提供SHAP可解释性分析。
    
    ### 2. 主要功能
    
    #### 🔍 单样本预测
    - **功能**: 输入单个患者的6个特征值进行实时预测
    - **步骤**:
      1. 在侧边栏选择"单样本预测"
      2. 调整6个特征的滑块值
      3. 点击"开始预测"按钮
    - **输出**:
      - 诊断结果 (良性/恶性)
      - 恶性概率
      - 风险等级
      - SHAP可解释性分析
      - 临床建议
    
    #### 📊 特征分析
    - **功能**: 分析模型的特征重要性
    - **包含**:
      - 基于Lasso系数的特征重要性
      - 特征详细说明
    
    ### 3. 6个关键特征说明
    
    本模型使用以下6个经过Lasso特征选择的关键特征:
    
    | 特征 | 类型 | Lasso系数 | 临床意义 |
    |------|------|-----------|----------|
    | `radius_worst` | 正相关 | +0.508 | 肿瘤最差半径，最重要的恶性指标 |
    | `concave points_mean` | 正相关 | +0.137 | 平均凹点数量，凹点越多恶性可能越大 |
    | `radius_se` | 正相关 | +0.133 | 半径标准误，反映细胞大小变化 |
    | `concavity_worst` | 正相关 | +0.103 | 最差凹度，凹度越大恶性可能越高 |
    | `area_worst` | 负相关 | -0.323 | 最差面积，在某些情况下与良性相关 |
    | `compactness_mean` | 负相关 | -0.147 | 平均紧致度，与细胞形状规则性相关 |
    
    ### 4. 结果解读指南
    
    #### 风险等级分类:
    - **低风险 (<30%)**: 恶性可能性较低，建议常规随访
    - **中风险 (30%-70%)**: 需要进一步评估，建议专科咨询
    - **高风险 (>70%)**: 恶性可能性较高，建议立即就医
    
    #### SHAP值解读:
    - **正SHAP值**: 增加恶性风险
    - **负SHAP值**: 降低恶性风险
    - **绝对值大小**: 表示影响程度
    
    ### 5. 技术信息
    
    - **模型算法**: LightGBM (梯度提升决策树)
    - **训练数据**: 威斯康星乳腺癌数据集
    - **特征选择**: Lasso回归 (系数绝对值>0.1)
    - **特征数量**: 6个关键特征
    - **可解释性**: SHAP (SHapley Additive exPlanations)
    
    ### 6. 重要声明
    
    ⚠️ **免责声明**:
    1. 本系统为辅助诊断工具，不能替代专业医生诊断
    2. 所有预测结果仅供参考
    3. 临床决策应结合完整的临床资料
    4. 如有疑问，请及时咨询医疗专业人员
    
    ### 7. 故障排除
    
    **常见问题**:
    1. **模型加载失败**: 检查`lightgbm_model.pkl`和`scaler.pkl`文件是否存在
    2. **SHAP分析失败**: 可能是内存不足，尝试重启应用
    3. **预测结果异常**: 检查输入特征值是否在合理范围内
    
    **技术支持**: 如有问题，请联系系统管理员
    """)

# ====================== 6. 页脚 ======================
st.sidebar.markdown("---")
st.sidebar.info("""
**乳腺癌诊断辅助系统**  

🏥 基于机器学习的临床决策支持工具  
📊 提供SHAP可解释性分析  
⚠️ 仅供医疗专业人员参考使用
""")

# 添加刷新按钮
if st.sidebar.button("🔄 刷新应用"):
    st.rerun()

