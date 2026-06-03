"""
GS - Generative AI for Engineering
Tema: Economia Espacial - Predição de Risco de Colisão Orbital

Deploy com Streamlit - Pipeline completo com XGBoost e Random Forest.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="GS — Risco Orbital",
    page_icon="🛰️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.header-box {
    background: linear-gradient(135deg, #1e3a5f 0%, #1a2a4a 100%);
    border-left: 5px solid #3b82f6;
    border-radius: 6px;
    padding: 1.2rem 1.6rem;
    margin-bottom: 1.2rem;
}
.header-box h2 { color: #93c5fd; margin: 0 0 0.3rem 0; font-size: 1.3rem; }
.header-box p  { color: #94a3b8; margin: 0; font-size: 0.85rem; }

.result-box {
    border-radius: 8px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.info-box {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.85rem;
    color: #94a3b8;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def gerar_dados_e_treinar():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, r2_score, mean_squared_error
    from xgboost import XGBClassifier, XGBRegressor
    

    np.random.seed(42)
    N = 1400

    altitude_km       = np.random.uniform(200, 2000, N)
    inclinacao_graus  = np.random.uniform(0, 98, N)
    excentricidade    = np.random.beta(1.2, 12, N) * 0.3
    distancia_miss    = np.clip(np.random.exponential(2.0, N) + 0.01, 0.001, 25)
    velocidade_rel    = np.clip(np.random.normal(10, 3, N), 0.5, 16)
    rcs_obj1          = np.random.lognormal(0.5, 1.5, N)
    rcs_obj2          = np.random.lognormal(-0.5, 1.8, N)
    cov_radial        = np.random.exponential(500, N)
    cov_transversal   = np.random.exponential(2000, N)
    cov_normal        = np.random.exponential(300, N)
    tempo_tca         = np.random.uniform(0.5, 72, N)
    obj1_ativo        = np.random.binomial(1, 0.55, N)
    obj2_debris       = np.random.binomial(1, 0.70, N)
    solar_f107        = np.clip(np.random.normal(130, 40, N), 65, 240)

    rcs_comb = rcs_obj1 + rcs_obj2
    cov_comb = np.sqrt(cov_radial**2 + cov_normal**2)
    log_pc = (
        -4.5 - 3.2*np.log1p(distancia_miss) + 0.8*np.log1p(rcs_comb)
        + 0.6*np.log1p(cov_comb/1000) + 0.3*np.log1p(velocidade_rel)
        - 0.02*tempo_tca + np.random.normal(0, 0.4, N)
    )
    pc = np.clip(1/(1+np.exp(-log_pc)), 1e-8, 0.9999)
    risco = pd.cut(pc, bins=[-np.inf,1e-5,1e-4,1e-3,np.inf], labels=[0,1,2,3]).astype(int)

    cov_vol = (cov_radial * cov_transversal * cov_normal)**(1/3)
    razao   = distancia_miss*1000 / np.sqrt(cov_radial**2+cov_normal**2).clip(1)
    ke      = rcs_comb * velocidade_rel**2

    df = pd.DataFrame({
        'altitude_km': altitude_km, 'inclinacao_graus': inclinacao_graus,
        'excentricidade': excentricidade, 'distancia_miss_km': distancia_miss,
        'velocidade_relativa_kms': velocidade_rel, 'rcs_objeto1_m2': rcs_obj1,
        'rcs_objeto2_m2': rcs_obj2, 'cov_radial': cov_radial,
        'cov_transversal': cov_transversal, 'cov_normal': cov_normal,
        'tempo_tca_horas': tempo_tca, 'obj1_ativo': obj1_ativo,
        'obj2_debris': obj2_debris, 'indice_solar_f107': solar_f107,
        'volume_covariancia': cov_vol, 'razao_dist_incerteza': razao,
        'energia_cinetica': ke,
        'log_distancia_miss_km':   np.log1p(distancia_miss),
        'log_cov_radial':          np.log1p(cov_radial),
        'log_cov_transversal':     np.log1p(cov_transversal),
        'log_cov_normal':          np.log1p(cov_normal),
        'log_rcs_objeto1_m2':      np.log1p(rcs_obj1),
        'log_rcs_objeto2_m2':      np.log1p(rcs_obj2),
        'log_volume_covariancia':  np.log1p(cov_vol),
        'log_energia_cinetica':    np.log1p(ke),
    })

    FEATURES = list(df.columns)
    X = df[FEATURES]; y_clf = risco; y_reg = pc
    

    X_tr, X_te, yc_tr, yc_te, yr_tr, yr_te = train_test_split(
        X, y_clf, y_reg, test_size=0.2, random_state=42, stratify=y_clf)

    scaler = StandardScaler()
    X_tr_sc = pd.DataFrame(scaler.fit_transform(X_tr), columns=FEATURES)
    X_te_sc = pd.DataFrame(scaler.transform(X_te),     columns=FEATURES)

    # XGBoost Classifier
    xgb_clf = XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric='mlogloss', random_state=42
    )
    xgb_clf.fit(X_tr_sc, yc_tr)
    cv_xgb = cross_val_score(xgb_clf, X_tr_sc, yc_tr, cv=5, scoring='accuracy', n_jobs=-1)

    # Random Forest Classifier
    rf_clf = RandomForestClassifier(
        n_estimators=200, max_depth=12, class_weight='balanced',
        random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_tr_sc, yc_tr)
    cv_rf = cross_val_score(rf_clf, X_tr_sc, yc_tr, cv=5, scoring='accuracy', n_jobs=-1)

    # XGBoost Regressor
    xgb_reg = XGBRegressor(
        n_estimators=500, max_depth=4, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    xgb_reg.fit(X_tr_sc, yr_tr)

    metricas = {
        'xgb_cv_acc':   round(cv_xgb.mean(), 4),
        'xgb_cv_std':   round(cv_xgb.std(), 4),
        'rf_cv_acc':    round(cv_rf.mean(), 4),
        'rf_cv_std':    round(cv_rf.std(), 4),
        'xgb_test_acc': round(accuracy_score(yc_te, xgb_clf.predict(X_te_sc)), 4),
        'rf_test_acc':  round(accuracy_score(yc_te, rf_clf.predict(X_te_sc)), 4),
        'xgb_reg_r2':   round(r2_score(yr_te, xgb_reg.predict(X_te_sc)), 4),
    }

    return {
        'df': df, 'pc': pc, 'risco': risco,
        'X_tr_sc': X_tr_sc, 'X_te_sc': X_te_sc,
        'yc_tr': yc_tr, 'yc_te': yc_te, 'yr_te': yr_te,
        'xgb_clf': xgb_clf, 'rf_clf': rf_clf, 'xgb_reg': xgb_reg,
        'scaler': scaler, 'metricas': metricas, 'FEATURES': FEATURES,
    }


with st.spinner("Treinando modelos... (primeira execução leva ~30 segundos)"):
    dados = gerar_dados_e_treinar()

df         = dados['df']
pc_full    = dados['pc']
risco_full = dados['risco']
FEATURES   = dados['FEATURES']
xgb_clf    = dados['xgb_clf']
rf_clf     = dados['rf_clf']
xgb_reg    = dados['xgb_reg']
scaler     = dados['scaler']
metricas   = dados['metricas']

ROTULOS = {0: 'Baixo', 1: 'Médio', 2: 'Alto', 3: 'Crítico'}
CORES   = {0: '#22c55e', 1: '#f59e0b', 2: '#f97316', 3: '#ef4444'}
ICONES  = {0: '🟢', 1: '🟡', 2: '🟠', 3: '🔴'}

st.markdown("""
<div class="header-box">
  <h2>🛰️ Predição de Risco de Colisão Orbital</h2>
  <p>GS · GenAI for Engineering · Economia Espacial</p>
</div>
""", unsafe_allow_html=True)

aba1, aba2, aba3, aba4 = st.tabs([
    "🎯 Predição",
    "📊 Análise dos Dados",
    "📈 Comparação de Modelos",
    "🔍 SHAP — Interpretabilidade"
])


# ── ABA 1: PREDIÇÃO ──────────────────────────────────────────────────────────
with aba1:
    st.subheader("Simulação de Conjunção Orbital")
    st.write("Preencha os parâmetros da conjunção para obter a predição de risco dos dois modelos treinados.")

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.markdown("**Parâmetros Orbitais**")
        altitude   = st.slider("Altitude (km)", 200, 2000, 550, step=10)
        inclinacao = st.slider("Inclinação (°)", 0.0, 98.0, 53.0, step=0.5)
        excentr    = st.slider("Excentricidade", 0.0, 0.30, 0.002, step=0.001, format="%.3f")

    with col_b:
        st.markdown("**Parâmetros da Conjunção**")
        dist_miss  = st.slider("Miss Distance (km)", 0.001, 25.0, 1.5, step=0.01, format="%.3f")
        vel_rel    = st.slider("Velocidade Relativa (km/s)", 0.5, 16.0, 10.5, step=0.1)
        tca        = st.slider("Tempo até TCA (horas)", 0.5, 72.0, 24.0, step=0.5)

    with col_c:
        st.markdown("**Objetos e Incerteza**")
        rcs1    = st.slider("RCS Objeto 1 (m²)", 0.01, 50.0, 3.0, step=0.1)
        rcs2    = st.slider("RCS Objeto 2 (m²)", 0.01, 50.0, 1.0, step=0.1)
        cov_r   = st.slider("Covariância Radial (m²)", 10.0, 5000.0, 500.0, step=10.0)
        ativo1  = st.checkbox("Objeto 1 é satélite ativo", value=True)
        debris2 = st.checkbox("Objeto 2 é debris", value=True)

    if st.button("▶ Executar Predição"):
        cov_t_v  = cov_r * 4
        cov_n_v  = cov_r * 0.6
        cov_vol  = (cov_r * cov_t_v * cov_n_v)**(1/3)
        razao    = dist_miss*1000 / np.sqrt(cov_r**2+cov_n_v**2).clip(1)
        rcs_comb = rcs1 + rcs2
        ke       = rcs_comb * vel_rel**2

        entrada = pd.DataFrame([{
            'altitude_km': altitude, 'inclinacao_graus': inclinacao,
            'excentricidade': excentr, 'distancia_miss_km': dist_miss,
            'velocidade_relativa_kms': vel_rel, 'rcs_objeto1_m2': rcs1,
            'rcs_objeto2_m2': rcs2, 'cov_radial': cov_r,
            'cov_transversal': cov_t_v, 'cov_normal': cov_n_v,
            'tempo_tca_horas': tca, 'obj1_ativo': int(ativo1),
            'obj2_debris': int(debris2), 'indice_solar_f107': 130.0,
            'volume_covariancia': cov_vol, 'razao_dist_incerteza': razao,
            'energia_cinetica': ke,
            'log_distancia_miss_km':  np.log1p(dist_miss),
            'log_cov_radial':         np.log1p(cov_r),
            'log_cov_transversal':    np.log1p(cov_t_v),
            'log_cov_normal':         np.log1p(cov_n_v),
            'log_rcs_objeto1_m2':     np.log1p(rcs1),
            'log_rcs_objeto2_m2':     np.log1p(rcs2),
            'log_volume_covariancia': np.log1p(cov_vol),
            'log_energia_cinetica':   np.log1p(ke),
        }])[FEATURES]

        entrada_sc = pd.DataFrame(scaler.transform(entrada), columns=FEATURES)

        pred_xgb  = int(xgb_clf.predict(entrada_sc)[0])
        prob_xgb  = xgb_clf.predict_proba(entrada_sc)[0]
        pred_rf   = int(rf_clf.predict(entrada_sc)[0])
        prob_rf   = rf_clf.predict_proba(entrada_sc)[0]
        pred_pc   = float(np.clip(xgb_reg.predict(entrada_sc)[0], 0, 1))

        st.markdown("---")
        st.markdown("#### Resultado da Predição")

        c1, c2, c3 = st.columns(3)
        for col, pred, probs, nome in [
            (c1, pred_xgb, prob_xgb, "XGBoost"),
            (c2, pred_rf,  prob_rf,  "Random Forest"),
        ]:
            cor = CORES[pred]
            with col:
                st.markdown(f"""
                <div class="result-box" style="background:{cor}20; border:2px solid {cor};">
                    <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.3rem;">{nome}</div>
                    <div style="font-size:2rem;">{ICONES[pred]}</div>
                    <div style="font-size:1.3rem;font-weight:700;color:{cor};">{ROTULOS[pred]}</div>
                    <div style="font-size:0.75rem;color:#94a3b8;margin-top:0.4rem;">
                        confiança: {max(probs):.1%}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with c3:
            st.markdown(f"""
            <div class="result-box" style="background:#1e293b;border:1px solid #334155;">
                <div style="font-size:0.8rem;color:#94a3b8;margin-bottom:0.3rem;">Pc estimado (XGBoost Regressor)</div>
                <div style="font-size:1.6rem;font-weight:700;color:#60a5fa;">{pred_pc:.2e}</div>
                <div style="font-size:0.72rem;color:#64748b;margin-top:0.4rem;">
                    {'⚠️ acima do limiar de manobra (1e-4)' if pred_pc > 1e-4 else '✅ abaixo do limiar de manobra'}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("**Distribuição de probabilidade por classe**")
        df_probs = pd.DataFrame({
            'Classe':        [f"{ICONES[i]} {ROTULOS[i]}" for i in range(4)],
            'XGBoost':       [f"{p:.1%}" for p in prob_xgb],
            'Random Forest': [f"{p:.1%}" for p in prob_rf],
        })
        st.dataframe(df_probs, use_container_width=True, hide_index=True)

        st.markdown("**Dados informados**")
        st.dataframe(entrada.round(4), use_container_width=True)


# ── ABA 2: EDA ───────────────────────────────────────────────────────────────
with aba2:
    st.subheader("Análise Exploratória do Dataset")
    st.write(f"Dataset sintético gerado via IA generativa (Claude): **{len(df):,} amostras** × **{len(FEATURES)} variáveis**.")

    contagem = pd.Series(risco_full).value_counts().sort_index()
    m1, m2, m3, m4 = st.columns(4)
    for col, (lvl, rot) in zip([m1,m2,m3,m4], ROTULOS.items()):
        n = contagem.get(lvl, 0)
        col.metric(f"{ICONES[lvl]} {rot}", f"{n:,}", f"{n/len(df)*100:.1f}%")

    st.markdown("---")
    ca, cb = st.columns(2)

    with ca:
        fig = px.scatter(
            df.assign(risco=risco_full, pc=pc_full).sample(500, random_state=1),
            x='distancia_miss_km', y='pc', color='risco',
            color_discrete_map={i: CORES[i] for i in range(4)},
            log_x=True, log_y=True,
            labels={'distancia_miss_km':'Miss Distance (km)','pc':'Probabilidade de Colisão'},
            title='Miss Distance vs Probabilidade de Colisão',
            opacity=0.5, height=350
        )
        fig.update_layout(legend_title_text='Risco')
        st.plotly_chart(fig, use_container_width=True)

    with cb:
        fig2 = go.Figure()
        for lvl in range(4):
            mask = np.array(risco_full) == lvl
            fig2.add_trace(go.Box(
                y=df.loc[mask, 'velocidade_relativa_kms'],
                name=f"{ICONES[lvl]} {ROTULOS[lvl]}",
                marker_color=CORES[lvl], boxmean=True
            ))
        fig2.update_layout(
            title='Velocidade Relativa por Nível de Risco',
            yaxis_title='km/s', showlegend=False, height=350
        )
        st.plotly_chart(fig2, use_container_width=True)

    cols_corr = ['distancia_miss_km','velocidade_relativa_kms','rcs_objeto1_m2',
                 'cov_radial','tempo_tca_horas','altitude_km']
    corr = df[cols_corr].corr()
    fig3 = px.imshow(corr.round(2), text_auto=True, color_continuous_scale='RdBu_r',
                     title='Matriz de Correlação entre Variáveis Numéricas', height=380)
    st.plotly_chart(fig3, use_container_width=True)

    with st.expander("Ver amostra do dataset (20 linhas)"):
        st.dataframe(
            df.assign(risco=risco_full, pc_foster=pc_full.round(6)).sample(20, random_state=7).round(4),
            use_container_width=True
        )


# ── ABA 3: COMPARAÇÃO ────────────────────────────────────────────────────────
with aba3:
    st.subheader("Validação e Comparação de Desempenho")
    st.write("""
    Foram treinados dois modelos de classificação (**XGBoost** e **Random Forest**) e um de regressão (**XGBoost Regressor**).
    A validação foi feita com **cross-validation estratificado de 5 folds** no treino e avaliação final no teste (20%).
    """)

    st.markdown("#### Classificação — Nível de Risco")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class="info-box">
            <b>XGBoost Classifier</b><br>
            n_estimators=500 · max_depth=4 · learning_rate=0.1 · subsample=0.8
        </div>
        """, unsafe_allow_html=True)
        st.metric("CV Accuracy (5-fold)", f"{metricas['xgb_cv_acc']:.2%}",
                  f"± {metricas['xgb_cv_std']:.2%}")
        st.metric("Test Accuracy", f"{metricas['xgb_test_acc']:.2%}")

    with col2:
        st.markdown("""
        <div class="info-box">
            <b>Random Forest Classifier</b><br>
            n_estimators=200 · max_depth=12 · class_weight='balanced'
        </div>
        """, unsafe_allow_html=True)
        st.metric("CV Accuracy (5-fold)", f"{metricas['rf_cv_acc']:.2%}",
                  f"± {metricas['rf_cv_std']:.2%}")
        st.metric("Test Accuracy", f"{metricas['rf_test_acc']:.2%}")

    melhor = "XGBoost" if metricas['xgb_cv_acc'] >= metricas['rf_cv_acc'] else "Random Forest"
    st.success(f"✅ **Modelo escolhido para deploy:** {melhor} (melhor CV Accuracy)")

    st.markdown("---")
    st.markdown("#### Regressão — Probabilidade de Colisão (Pc)")
    st.metric("XGBoost Regressor — R²", f"{metricas['xgb_reg_r2']:.4f}")

    # Gráfico comparativo
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='XGBoost',
        x=['CV Accuracy', 'Test Accuracy'],
        y=[metricas['xgb_cv_acc'], metricas['xgb_test_acc']],
        marker_color='#3b82f6',
        text=[f"{v:.3f}" for v in [metricas['xgb_cv_acc'], metricas['xgb_test_acc']]],
        textposition='outside'
    ))
    fig_comp.add_trace(go.Bar(
        name='Random Forest',
        x=['CV Accuracy', 'Test Accuracy'],
        y=[metricas['rf_cv_acc'], metricas['rf_test_acc']],
        marker_color='#f59e0b',
        text=[f"{v:.3f}" for v in [metricas['rf_cv_acc'], metricas['rf_test_acc']]],
        textposition='outside'
    ))
    fig_comp.update_layout(
        title='Comparação de Desempenho — XGBoost vs Random Forest',
        barmode='group', yaxis=dict(range=[0, 1.1]), height=370
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # Feature importance (XGBoost nativa)
    st.markdown("#### Importância de Variáveis — XGBoost (Feature Importance nativa)")
    imp = pd.Series(xgb_clf.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(12)
    fig_imp = go.Figure(go.Bar(
        x=imp.values, y=imp.index, orientation='h',
        marker_color='#3b82f6',
        text=[f"{v:.3f}" for v in imp.values], textposition='outside'
    ))
    fig_imp.update_layout(
        title='Feature Importance — XGBoost (Top 12)',
        xaxis_title='Importância', height=420, margin=dict(l=210)
    )
    st.plotly_chart(fig_imp, use_container_width=True)


# ── ABA 4: SHAP ──────────────────────────────────────────────────────────────
with aba4:
    st.subheader("Interpretabilidade com SHAP")
    st.write("""
    SHAP (SHapley Additive exPlanations) é baseado na teoria dos jogos cooperativos.
    Cada feature recebe um valor φᵢ que representa sua contribuição marginal para a predição.
    O XGBoost tem suporte nativo a SHAP via TreeExplainer, tornando o cálculo eficiente.
    """)
    st.markdown("> **Fórmula:** `f(x) = φ₀ + φ₁x₁ + φ₂x₂ + ... + φₙxₙ`")

    if st.button("▶ Calcular SHAP Values (amostra de 200 exemplos)"):
        import shap

        sample_idx = np.random.choice(len(dados['X_te_sc']), 200, replace=False)
        X_shap = dados['X_te_sc'].iloc[sample_idx]

        with st.spinner("Calculando valores SHAP..."):
            explainer = shap.TreeExplainer(xgb_clf)
            shap_vals = explainer.shap_values(X_shap)

        if isinstance(shap_vals, list):
            sv_critico = shap_vals[3]
        else:
            sv_critico = shap_vals[:, :, 3]

        media_abs = np.abs(sv_critico).mean(axis=0)

        imp_shap = pd.Series(
            media_abs,
            index=FEATURES
        ).sort_values(ascending=True).tail(15)

        st.markdown("#### SHAP — Importância Global das Features (Classe: Risco Crítico)")
        fig_shap = go.Figure(go.Bar(
            x=imp_shap.values, y=imp_shap.index, orientation='h',
            marker=dict(color=imp_shap.values, colorscale='Viridis',
                        showscale=True, colorbar=dict(title='|SHAP médio|')),
            text=[f"{v:.4f}" for v in imp_shap.values], textposition='outside'
        ))
        fig_shap.update_layout(
            xaxis_title='Valor SHAP médio |φ|',
            height=520, margin=dict(l=230)
        )
        st.plotly_chart(fig_shap, use_container_width=True)
        st.success("✅ Valores SHAP calculados com sucesso!")
    else:
        st.info("Clique no botão acima para calcular. O processo leva alguns segundos.")
