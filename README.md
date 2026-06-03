# Trabalho Global Solution - Generative AI for Engineering
## Predição de Risco de Colisão Orbital na Economia Espacial

### Integrantes:
João Rodrigo Solano Nogueira RM551319  
Julia Amorim Bezerra RM99609  
Lana Giulia Auada Leite RM551143  
Tony Willian da Silva Segalin RM550667 

---

## Problema

O crescimento acelerado da Economia Espacial, impulsionado pelo aumento do número de satélites, megaconstelações e detritos espaciais em órbita terrestre, elevou significativamente o risco de colisões orbitais. Esses eventos podem causar prejuízos milionários, interrupção de serviços de comunicação e navegação, além de gerar novos fragmentos espaciais que aumentam ainda mais o risco operacional.

Este projeto propõe uma solução baseada em Inteligência Artificial e Machine Learning para prever o nível de risco de conjunções orbitais e estimar a probabilidade de colisão (Probability of Collision), auxiliando operadores na tomada de decisão sobre possíveis manobras de mitigação.

## Fonte dos Dados

Conforme permitido pela proposta do projeto, foi utilizado um conjunto de dados sintético gerado com auxílio de IA generativa (Claude).

O dataset contém mais de 1.000 registros e mais de 10 variáveis relacionadas à dinâmica orbital, incluindo:

- Altitude orbital
- Inclinação orbital
- Excentricidade
- Distância mínima de aproximação (Miss Distance)
- Velocidade relativa
- Radar Cross Section (RCS)
- Covariâncias orbitais
- Tempo até o TCA (Time of Closest Approach)
- Índice de atividade solar F10.7

Além das variáveis originais, foram criadas novas variáveis derivadas por meio de Feature Engineering.

## Metodologia

- Geração do dataset sintético
- Análise exploratória (EDA)
- Pré-processamento e feature engineering
- Treinamento dos modelos (Random Forest + Gradient Boosting)
- Validação, comparação e escolha do melhor modelo
- Interpretabilidade com SHAP

## Modelos Testados 

Classificação do Nível de Risco: XGBoost Classifier e Random Forest Classifier
Regressão da Probabilidade de Colisão (Pc): XGBoost Regressor

## Resultados

| Modelo | Tarefa | CV Accuracy | Test Accuracy |
|---|---|---|---|
| XGBoost | Classificação | ~91% | ~90% |
| Random Forest | Classificação | ~88% | ~89% |

| Modelo | Tarefa | Métrica | Resultado |
|---|---|---|---|
| XGBoost Regressor | Regressão | R² | 0.6576 |

## Interpretabilidade com SHAP

As variáveis mais influentes identificadas foram:
- distancia_miss_km	- Distância mínima entre os objetos; quanto menor, maior o risco
- razao_dist_incerteza - Relação entre a distância e a incerteza orbital
- tempo_tca_horas - Tempo restante até a máxima aproximação
- rcs_objeto1_m2 - Tamanho efetivo do primeiro objeto
- rcs_objeto2_m2 - Tamanho efetivo do segundo objeto
- energia_cinetica - Potencial destrutivo da colisão
- indice_solar_f107 - Influência da atividade solar nas órbitas
Os resultados demonstraram coerência com os princípios físicos da dinâmica orbital, indicando que o modelo aprendeu relações plausíveis para a estimativa de risco.

## Estrutura do Projeto

```
space_project/
│
├── notebooks/
│   └── pipeline_orbital_risk.ipynb   ← Pipeline completo (Etapas 1–6)
│
├── app/
│   └── app.py                        ← Deploy via Streamlit
│
├── data/
│   └── conjuncoes_orbitais.csv       ← Gerado ao rodar o notebook
│
├── models/
│   └── *.joblib                      ← Modelos salvos após treino
│
├── docs/
│   └── *.png                         ← Gráficos gerados
│
└── README.md
```

---

## Como Rodar

### 1. Instalar dependências

```bash
pip install streamlit scikit-learn shap plotly pandas numpy matplotlib seaborn joblib jupyter
```

### 2. Rodar o notebook (pipeline completo)

```bash
jupyter notebook notebooks/pipeline_orbital_risk.ipynb
```

Rodando manualmente, xecute todas as células em ordem. O notebook cobre:
- Etapa 1: Geração do dataset sintético
- Etapa 2: Análise exploratória (EDA)
- Etapa 3: Pré-processamento e feature engineering
- Etapa 4: Treinamento dos modelos (Random Forest + Gradient Boosting)
- Etapa 5: Validação, comparação e escolha do melhor modelo
- Etapa 6: Interpretabilidade com SHAP

### 3. Rodar o app de deploy (Streamlit)

```bash
cd app
streamlit run app.py
```
ou
```bash
python -m streamlit run app.py
```

Acesse no navegador: `http://localhost:8501`

O app possui 4 abas:
- **Predição** — simule uma conjunção orbital e veja o risco predito
- **Análise dos Dados** — visualizações EDA do dataset
- **Comparação de Modelos** — métricas e feature importance
- **SHAP** — interpretabilidade do modelo

---
