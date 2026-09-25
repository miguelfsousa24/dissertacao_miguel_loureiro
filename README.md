# Data Mining Aplicado à Transformação Digital de uma PME Portuguesa de E-commerce: Um Caso de Estudo Integrado

**Dissertação de Mestrado em Ciência de Dados** | Universidade de Leiria e Oeste

**Âmbito:** Aplicação integrada de Data Mining e Machine Learning a uma PME portuguesa do setor pet e agrícola que atua em contexto de e-commerce, com forte predominância no segmento de Aves, sobre dados transacionais reais no período temporal compreendido entre fevereiro de 2016 e março de 2026.

---

## 1. Contexto e motivação

A crescente digitalização do comércio de retalho tem contribuído para aumentar substancialmente o volume de dados disponíveis nas Pequenas e Médias Empresas (PMEs). Contudo, a capacidade analítica destas organizações continua, muitas vezes, aquém do potencial disponível nos próprios dados. Esta dissertação materializa, num caso de estudo real, a aplicação de técnicas de Data Mining como um instrumento essencial de melhoria na tomada de decisão em PMEs.

A empresa em estudo é uma PME portuguesa do setor pet e agrícola com atuação em contexto de e-commerce, cuja base transacional cobre ~10 anos de atividade na loja online.

A dissertação desenvolve três componentes analíticas complementares orientadas a três objetivos de negócio distintos:

|Objetivo de Negócio|Objetivo de Data Mining|Vertente Analítica|
|-|-|-|
|**ON1** - Maximizar o valor gerado pela base de clientes ativos, antecipando aqueles que detêm maior propensão de recompra a 90 dias|**ODM1** - Prever a recompra a 90 dias|Classificação supervisionada|
|**ON2** - Personalizar a comunicação por perfil|**ODM2** - Segmentar a base de clientes|Clustering (portfólio + ciclo de vida)|
|**ON3** - Estimular o cross-sell|**ODM3** - Descobrir padrões de co-compra|Regras de associação (Apriori)|

A metodologia adotada é a CRISP-DM (*Cross-Industry Standard Process for Data Mining*), organizando o trabalho nas suas seis fases (Compreensão do Negócio, Compreensão dos Dados, Preparação dos Dados, Modelação, Avaliação e Implementação). A fase de Implementação será desenvolvida numa fase posterior à elaboração da dissertação, dada a necessidade de um processo mais gradual de realização de campanhas para que os resultados sejam avaliados através de comparação sustentada e cuidada com períodos anteriores.

---

## 2. Estrutura do repositório

```
├── README.md                                             # Este ficheiro
│
├── Dissertação/
│   └── Dissertação_Miguel Loureiro_2220134.docx                  # Documento principal que inclui a Fase 1 CRISP-DM
│
├── Notebooks/
│   ├── 1_Compreensão_de_Dados.ipynb                      # Fase 2 CRISP-DM
│   ├── 2_Preparação_de_Dados.ipynb                       # Fase 3 CRISP-DM
│   ├── 3_Modelação_Baseline_Classificação.ipynb          # Fase 4 - Classificação (baseline)
│   ├── 4_Modelação_Gridsearch_Classificação.ipynb        # Fase 4 - Classificação (otimização)
│   ├── 5_Clustering_K_Means_Portfólio_Produto.ipynb      # Fase 4 - Clustering portfólio (K-Means)
│   ├── 6_Clustering_Hierárquico.ipynb                    # Fase 4 - Clustering portfólio (Hierárquico)
│   ├── 7_Clustering_RFM.ipynb                            # Fase 4 - Clustering ciclo de vida / RFM
│   ├── 8_Regras_de_Associação_Histórico_Completo.ipynb   # Fase 4 - Regras Apriori (histórico)
│   └── 9_Associacao_Apriori_12meses.ipynb                # Fase 4 - Regras Apriori (12 meses)
│
├── Dataset ilustrativo/
│   └── vendas_amostra_anonimizada.csv                    # Amostra pseudonimizada transacional para compreensão da estrutura
│
├── Datasets/                                             
│   ├── df_linhas.csv                                     # Produzido pelo NB1 (linha de produto) - Não incluído no repositório, pois contém dados pessoais (ver secção 13)
│   ├── df_vendas.csv                                     # Produzido pelo NB1 (venda) - Não incluído no repositório, pois contém dados pessoais (ver secção 13)
│   ├── df_clientes.csv                                   # Produzido pelo NB1 (cliente) - Não incluído no repositório, pois contém dados pessoais (ver secção 13)
│   ├── dataset_modelo_compra.csv                         # Produzido pelo NB2 (modelação para classificação)
│   ├── df_clientes_enriquecido.csv                       # Produzido pelo NB2 (modelação para clustering) - Não incluído no repositório, pois contém dados pessoais (ver secção 13)
│   ├── dataset_features_clustering.csv                   # Produzido pelo NB5 (features de clustering de portfólio)
│   ├── dataset_clusters_kmeans.csv                       # Produzido pelo NB5 (atribuição de clusters K-Means)
│   ├── dataset_clusters_hierarquico.csv                  # Produzido pelo NB6 (atribuição de clusters Hierárquico)
│   └── dataset_clusters_rfm.csv                          # Produzido pelo NB7 (atribuição de clusters RFM)
│
├── Modelos/
│   ├── regressao_logistica_pipeline.pkl                  # Baseline (NB3)
│   ├── svm_pipeline.pkl                                  # Baseline (NB3)
│   ├── decision_tree_pipeline.pkl                        # Baseline (NB3)
│   ├── random_forest_pipeline.pkl                        # Baseline (NB3)
│   ├── regressao_logistica_otimizado_pipeline.pkl        # Otimizado pós Grid Search (NB4)
│   ├── svm_otimizado_pipeline.pkl                        # Otimizado pós Grid Search (NB4)
│   ├── decision_tree_otimizado_pipeline.pkl              # Otimizado pós Grid Search (NB4)
│   ├── random_forest_otimizado_pipeline.pkl              # Otimizado pós Grid Search (NB4)
│   ├── modelo_final_compra_pipeline.pkl                  # Modelo final selecionado - cópia da Decision Tree otimizada (NB4)
│   ├── kmeans_final.pkl                                  # K-Means portfólio (NB5)
│   ├── scaler_kmeans.pkl                                 # Scaler do K-Means portfólio (NB5)
│   ├── scaler_clusters_hc.pkl                            # Scaler do Clustering Hierárquico (NB6)
│   ├── kmeans_rfm.pkl                                    # K-Means RFM / ciclo de vida (NB7)
│   └── scaler_rfm.pkl                                    # Scaler do K-Means RFM (NB7)
│
└── Resultados/
    ├── resultados_classificacao_baseline_cv.csv          # Produzido pelo NB3 (métricas CV dos 4 baselines)
    ├── resultados_classificacao_baseline_teste.csv       # Produzido pelo NB3 (métricas em teste dos 4 baselines)
    ├── features_selecionadas_classificacao.csv           # Produzido pelo NB4 (features selecionadas por modelo via SelectKBest)
    ├── resultados_classificacao_baseline_vs_gridsearch.csv  # Produzido pelo NB4 (comparação baseline vs. otimizado)
    ├── resultados_classificacao_modelos_otimizados.csv   # Produzido pelo NB4 (métricas finais dos 4 modelos otimizados)
    ├── regras_associacao_historico.csv                   # Produzido pelo NB8 (27 regras, histórico completo)
    ├── regras_associacao_sensibilidade.csv             # Produzido pelo NB8 (análise de sensibilidade, 133 regras, min_support=0,0005)
    └── regras_associacao_12m.csv                         # Produzido pelo NB9 (77 regras, últimos 12 meses)
```

---

## 3. Caracterização do dataset

|Atributo|Valor|
|-|-|
|Fonte|Plataforma de e-commerce da empresa|
|Período|13 de fevereiro de 2016 a 31 de março de 2026 (~10 anos)|
|Granularidade original|Linha de produto|
|Volume original|153.100 registos × 17 atributos|
|Volume após limpeza|152.976 registos (124 inconsistências `Qt = 0` removidas)|
|Vendas únicas|55.971|
|Clientes únicos|13.074|
|Produtos únicos|8.506|
|Distribuição por segmento|Aves (64,34%), Cães (18,23%), Gatos (7,28%), Roedores e restantes segmentos nicho (cerca de 10%)|

### Variáveis principais

* **Identificadores:** `Ref Venda`, `Ref Cliente`, `Ref Produto`, `Nome do Cliente`
* **Contacto:** `E-Mail`, `Telemóvel`
* **Temporais:** `Data`
* **Produto:** `Nome do Produto`, `Tipo`, `Categoria`, `Subcategoria`, `Marca`
* **Transacionais:** `Qt`, `Preço Produto c/IVA`, `Total Compra c/IVA`, `Desconto`, `Pagamento`

---

## 4. Pipeline de execução

Os notebooks devem ser executados pela ordem numérica, pois cada um tira partido dos outputs resultantes do anterior. Assim, a pipeline está organizada de forma a que os ficheiros produzidos sejam reutilizados pelos seguintes.

```
NB1 (Compreensão dos Dados)
    │
    ├── df_linhas.csv ─────────────────────────┐
    ├── df_vendas.csv                          │
    └── df_clientes.csv ─────────┐             │
                                 ▼             ▼
                          NB2 (Preparação dos Dados)
                                 │
                                 ├── dataset_modelo_compra.csv ────┐
                                 └── df_clientes_enriquecido.csv ──┼───┐
                                                                   │   │
                                                                   ▼   │
                                                          NB3 (Baseline)
                                                                   │
                                                                   ▼
                                                          NB4 (Grid Search)
                                                                       │
                                                                       │   ┌──────────────────────┐
                                                                       │   │                      │
                                                                       └───┤  NB5 (K-Means port.) │
                                                                           │  NB6 (Hierárquico)   │
                                                                           │  NB7 (K-Means RFM)   │
                                                                           └──────────────────────┘
                                                                                       │
df_linhas.csv ─────────────────────────────────────────────────────────────────────────┤
                                                                                       ▼
                                                                              NB8 (Apriori histórico)
                                                                              NB9 (Apriori 12 meses)
```

### Dependências entre notebooks

|Notebook|Inputs|Outputs principais|
|-|-|-|
|**NB1**|`vendas_16_26.csv` (ou `vendas_amostra_anonimizada.csv` - ver secção 13) |`df_linhas.csv`, `df_vendas.csv`, `df_clientes.csv`|
|**NB2**|`df_linhas`, `df_vendas`, `df_clientes`|`dataset_modelo_compra.csv`, `df_clientes_enriquecido.csv`|
|**NB3**|`dataset_modelo_compra.csv`|4 baselines (.pkl) + `Resultados/resultados_classificacao_baseline_cv.csv` + `Resultados/resultados_classificacao_baseline_teste.csv`|
|**NB4**|`dataset_modelo_compra.csv` (baselines são recriados internamente para comparação)|4 pipelines otimizados (.pkl) + `modelo_final_compra_pipeline.pkl` + `Resultados/features_selecionadas_classificacao.csv` + `Resultados/resultados_classificacao_baseline_vs_gridsearch.csv` + `Resultados/resultados_classificacao_modelos_otimizados.csv`|
|**NB5**|`df_linhas.csv` + `df_clientes_enriquecido.csv`|`kmeans_final.pkl`, `scaler_kmeans.pkl`, `dataset_features_clustering.csv`, `dataset_clusters_kmeans.csv`|
|**NB6**|`dataset_features_clustering.csv` + `dataset_clusters_kmeans.csv` (outputs do NB5)|`scaler_clusters_hc.pkl`, `dataset_clusters_hierarquico.csv`|
|**NB7**|`df_clientes_enriquecido.csv` + `dataset_clusters_kmeans.csv` (NB5, usado no cruzamento da secção 10)|`kmeans_rfm.pkl`, `scaler_rfm.pkl`, `dataset_clusters_rfm.csv`|
|**NB8**|`df_linhas.csv`|`Resultados/regras_associacao_historico.csv` (análise principal, 27 regras) + `Resultados/regras_associacao_sensibilidade.csv` (análise de sensibilidade, 133 regras)|
|**NB9**|`df_linhas.csv`|`Resultados/regras_associacao_12m.csv` (77 regras = 16 persistentes + 61 emergentes)|

---

## 5. Resumo metodológico de cada notebook

### NB1 - Compreensão dos Dados

Carregamento, inspeção e tratamento inicial do dataset. Avaliação de qualidade (valores omissos, duplicados, outliers, valores negativos). Análise exploratória temporal, transacional e de produto. Construção dos datasets agregados.

### NB2 - Preparação dos Dados

Construção da janela temporal (180 dias de features + 90 dias de target) para evitar data leakage. Criação das variáveis RFM, feature engineering (12 variáveis comportamentais), análise de correlação e seleção (remoção de `Frequency_FW` por colinearidade perfeita). Criação da variável alvo `Target_Compra`.

### NB3 - Modelação | Baseline de Classificação

Aplicação dos quatro algoritmos (Regressão Logística, SVM, Decision Tree, Random Forest) com hiperparâmetros default em pipelines `StandardScaler + estimador`. Avaliação por validação cruzada estratificada (10 folds) sobre o conjunto de treino e por avaliação final no conjunto de teste independente.

### NB4 - Modelação | Grid Search e Seleção de Variáveis

Otimização de hiperparâmetros via `GridSearchCV` (com `scoring='recall'`), seguindo-se a seleção de variáveis com `SelectKBest` e a seleção do modelo final com base no conjunto de treino através da métrica RecallCV obtida em validação cruzada. É ainda feita uma avaliação final da capacidade de generalização no conjunto de teste.

### NB5, NB6 - Modelação | Clustering de Portfólio

K-Means (NB5) e Clustering Hierárquico com linkage Ward (NB6) sobre as mesmas 21 features (15 shares por Tipo, 3 de diversidade, 3 de padrão de consumo). Determinação de k através de Elbow + Silhouette + Dendrograma. Comparação inter-método para análise de coerência entre segmentações.

### NB7 - Modelação | Clustering RFM

K-Means sobre as três features RFM (Recency, Frequency, Monetary) para segmentação por ciclo de vida do cliente. Cruzamento com NB5 para identificar interceções estratégicas.

### NB8 - Modelação | Regras de Associação (Histórico)

Apriori sobre as 55.971 transações de todo o período transacional da empresa. Parâmetros escolhidos: `min_support=0.001`, `min_confidence=0.10`, `min_lift=2.0`. As 27 regras identificadas são exportadas para `Resultados/regras_associacao_historico.csv`. O notebook inclui ainda uma análise de sensibilidade com `min_support=0.0005`, exportada para `Resultados/regras_associacao_sensibilidade.csv` (133 regras, das quais as 27 principais são um subconjunto integral). As análises realizadas são fundamentadas por uma normalização prévia dos nomes de produtos, dado que ao longo do tempo se verificam ligeiras atualizações na respetiva nomenclatura (ex.: '-' vs '»'). A normalização tem em consideração a versão mais recente da nomenclatura do produto.

### NB9 - Modelação | Regras de Associação (Últimos 12 Meses)

Apriori sobre as transações dos últimos 12 meses (abr 2025 - mar 2026), com os mesmos parâmetros do NB8. Comparação temporal que identifica regras persistentes, emergentes e descontinuadas. As 77 regras identificadas são exportadas para `Resultados/regras_associacao_12m.csv`. São identificadas 16 regras persistentes, 61 emergentes e 11 descontinuadas.

---

## 6. Critérios de sucesso e principais resultados

### ODM1 - Classificação

|Critério|Limiar|Resultado|Estado|
|-|-|-|-|
|Recall|≥ 0,70|0,8242|✓ Cumprido|
|F1-Score|≥ 0,60|0,6939|✓ Cumprido|
|ROC-AUC|≥ 0,70|0,7536|✓ Cumprido|
|Gap CV-Teste|≤ 0,05|+0,0090|✓ Cumprido|

**Modelo final:** Decision Tree otimizada com `max_depth=3`, `criterion=gini`, `class_weight='balanced'`, sobre 7 features selecionadas.

### ODM2 - Clustering

|Critério|Limiar|Resultado|Estado|
|-|-|-|-|
|Silhouette portfólio|≥ 0,30|0,3836 (K-Means, k=7) / 0,3202 (Hierárquico, k=6)|✓ Cumprido|
|Silhouette RFM|≥ 0,50|0,5132|✓ Cumprido|
|Concordância inter-método (portfólio)|≥ 80%|47,8% - 100,0%|⚠ Cumprido parcialmente|

**Segmentações obtidas:**

* 7 clusters por portfólio em K-Means e 6 em método Hierárquico
* 4 clusters por ciclo de vida (RFM): Champions (1,5%), Regulares (9,3%), Ocasionais (54,3%), Adormecidos (34,9%)

**Nota sobre a concordância inter-método:** de sete arquétipos comparados entre K-Means (NB5) e Hierárquico (NB6), seis apresentam concordância entre 86,3% e 100,0%, no entanto o segmento associado aos Avicultores VIP regista apenas 47,8% (275 dos 575 clientes identificados como VIP pelo K-Means mantêm-se como VIP no Hierárquico, sendo os restantes 300 reclassificados, dos quais 284 são absorvidos pelo cluster de Avicultores Gerais). Este resultado, discutido no NB6, evidencia uma fronteira estatisticamente ténue entre avicultores VIP e avicultores regulares e não cumpre isoladamente o critério de ≥80% definido para este segmento.

**Descoberta integrada:** os 81 Champions (Cluster 2 da segmentação RFM, NB7) pertencem na sua totalidade ao cluster de portfólio "Avicultores VIP" (Cluster 3 do K-Means de portfólio, NB5), indicador antecipatório de transição para o estatuto Champion.

### ODM3 - Regras de Associação

|Critério|Limiar|Resultado|Estado|
|-|-|-|-|
|Support mínimo|≥ 0,001|0,001 - 0,005|✓ Cumprido|
|Confidence mínimo|≥ 0,10|0,11 - 0,81|✓ Cumprido|
|Lift mínimo|≥ 2,0|6,7 - 231,7|✓ Cumprido|
|Nº regras NB8 (histórico)|≥ 10|27 regras|✓ Cumprido|
|Nº regras NB9 (12 meses)|≥ 30|77 regras|✓ Cumprido|

**Análise temporal:** 16 regras persistentes, 61 emergentes, 11 descontinuadas. Resultado atribuível à normalização dos nomes de produtos, que tem em conta a renomeação do catálogo. Além disso, a descontinuação de regras resulta da reformulação de gama e descontinuações de produtos.

---

## 7. Ambiente e dependências

### Plataforma de execução

Todos os notebooks foram desenvolvidos e testados em Google Colab, com acesso ao Google Drive para persistência dos datasets e modelos.

### Versão do Python

Python 3.13.15

### Bibliotecas principais

```
pandas (2.2.3)
numpy (2.1.3)
matplotlib (3.10.0)
seaborn (0.13.2)
scikit-learn (1.6.1)
scipy (1.16.3)
apyori (1.1.2)
joblib (1.5.3)

```

A maioria das bibliotecas vem pré-instalada no Google Colab. A única exceção é `apyori`, que é instalada na primeira célula do NB8 e NB9.

---

## 8. Instruções de reprodução

### Em Google Colab

1. **Carregar os notebooks para o Drive:** Criar uma pasta `MyDrive/dissertacao/notebooks/` e copiar os 9 notebooks.
2. **Carregar os dados:** O dataset original `vendas_16_26.csv` não se encontra no repositório devido à inclusão de dados pessoais (ver secção 13. notas finais). Em alternativa, é possível copiar a amostra pseudonimizada `vendas_amostra_anonimizada.csv` de 150 registos e 35 clientes distintos para `MyDrive/`, no entanto os resultados obtidos a partir deste processo não replicam os documentados ao longo da dissertação.
3. **Executar os notebooks pela ordem numérica:** Cada notebook produz outputs (CSV e PKL) que são consumidos pelos seguintes.
4. **Caminhos:** Todos os notebooks assumem `/content/drive/MyDrive/...` como raiz. Ajustar se a estrutura do Drive for diferente.
5. **Subpasta `modelos/`:** Os modelos (`.pkl`) são guardados em `MyDrive/modelos/`, criada automaticamente pelos notebooks na primeira exportação.
6. **Ficheiros de resultados:** Os notebooks NB3, NB4, NB8 e NB9 exportam os seus CSVs de resultados (métricas de classificação e regras de associação) diretamente para a raiz de `MyDrive/`. Neste repositório, esses ficheiros estão reorganizados na pasta `Resultados/` apenas como forma de organização; no caso de reprodução da pipeline em Colab a partir do zero, deve-se mover esses CSVs para `Resultados/` manualmente após cada exportação.

### Em ambiente local

Alternativamente, os notebooks podem ser executados localmente (Jupyter Notebook / VS Code / JupyterLab). Para tal:

1. Instalar as bibliotecas listadas na secção 7 num ambiente Python 3.13.15.
2. Substituir, em cada notebook, o path base `/content/drive/MyDrive/` pelo caminho local escolhido (por exemplo, a raiz da pasta do repositório clonado).
3. Remover ou comentar as células de montagem do Google Drive (`from google.colab import drive; drive.mount(...)`), presentes na secção 2 de cada notebook.
4. Executar os notebooks pela ordem numérica, tal como no fluxo Colab.

---

## 9. Decisões metodológicas-chave

### Janela temporal (180 + 90 dias)

* **Janela de features (180 dias):** sustentada pela média observada de `Dias_Entre_Compras` (177 dias), cobrindo um ciclo médio completo de recompra.
* **Janela de target (90 dias):** fundamentada pela mediana observada (95 dias), capturando o cliente típico e alinhando-se com o trimestre operacional.
* **Data Leakage:** `Recency_FW` é calculada até ao final da janela de features (não até `data_max`), eliminando a contaminação temporal.

### Universo de modelação

* **Classificação:** clientes com pelo menos uma compra na janela de features → 2.257 clientes
* **Clustering:** clientes recorrentes (Frequency ≥ 2) → 5.521 clientes
* **Regras de associação:** todas as transações (55.971 no histórico, 9.363 nos últimos 12 meses)

### Métrica primária na Classificação

**Recall**, em coerência com a prioridade declarada na fase de Compreensão do Negócio pela minimização de Falsos Negativos, pois o custo de um FN (perda de oportunidade de nutrir cliente recomprador não detetado) é superior ao de um Falso Positivo (custo residual de enviar campanha personalizada a cliente com reduzida probabilidade de recomprar no período temporal em análise).

### Tratamento de desbalanceamento

`class_weight='balanced'` em todos os algoritmos sensíveis a desbalanceamento representado pela proporção 63,4% (classe 0) / 36,6% (classe 1).

### Normalização

`StandardScaler` aplicado dentro das pipelines de cross-validation (ajuste apenas sobre os folds de treino). Nunca aplicado globalmente antes da divisão treino/teste.

---

## 10. Limitações conhecidas

1. **Duplicação de identificadores de cliente** (~4,47%) devida à ausência de obrigatoriedade de registo na plataforma, levando à eventual subestimação ligeira da frequência e do valor gasto individuais.
2. **Ausência de variáveis demográficas e geográficas** que enriqueceriam o perfil dos clientes.
3. **Taxa de Falsos Positivos do modelo final** (31,7%) aceite como contrapartida da minimização de FN, embora implique o custo residual de comunicação a clientes com menor probabilidade de responder a iniciativas de nutrição e fidelização.
4. **Valores moderados de Silhouette score** na abordagem de clustering de portfólio (0,3202 - 0,3836), sendo expectável em features de composição (shares que somam 1), mas indicia fronteiras parcialmente sobrepostas entre clusters.
5. **Concordância inter-método reduzida no segmento Avicultores VIP** (47,8%, abaixo do critério de sucesso de ≥80% definido para o ODM2), refletindo uma fronteira estatisticamente ténue entre avicultores VIP e avicultores regulares — ver nota na secção 6.
6. **Janelas temporais não otimizadas**, sendo que a sensibilidade do desempenho à definição das janelas não foi avaliada neste trabalho.
7. **Ausência de validação externa em produção**, pois os modelos não foram testados em campanhas reais.

---

## 11. Estrutura da dissertação

|Capítulo|Conteúdo|
|-|-|
|1. Introdução|Contextualização, problema, objetivos, RQs, estrutura|
|2. Enquadramento Teórico|Conceitos de Transformação Digital, PMEs, E-commerce, BI, Data Mining|
|3. Revisão da Literatura|Organizada por RQ; síntese crítica e lacunas|
|4. Metodologia|CRISP-DM, descrição das fases, objetivos de DM e critérios de sucesso|
|5. Implementação e Resultados|Operacionalização das fases 2-5 do CRISP-DM|
|6. Conclusões|Síntese, contribuições, limitações, trabalho futuro|

---

## 12. Autor

**Miguel Loureiro | 2220134**
Mestrando em Ciência de Dados
Universidade de Leiria e Oeste

**Orientação:** Ricardo Malheiro

---

## 13. Notas finais

Este repositório foi construído para compreensão global do contexto, estrutura e organização da dissertação. Os notebooks são autossuficientes, pois cada um contém a sua contextualização CRISP-DM, justificação das decisões metodológicas e síntese conclusiva. Esta opção permite que o leitor possa abrir qualquer notebook isoladamente e compreender o seu propósito sem necessidade de consultar o documento principal.

### Privacidade e confidencialidade dos dados

A base de dados recolhida da plataforma de e-commerce e utilizada para o desenvolvimento desta dissertação não é publicada no repositório, na medida em que o cumprimento do Regulamento Geral de Proteção de Dados (RGPD) e o acordo com a empresa não permitem a disponibilização de dados pessoais dos clientes e respetivas transações. Esta restrição aplica-se ao ficheiro original (`vendas_16_26.csv`) e a todos os ficheiros dele diretamente derivados que integram dados pessoais identificáveis, nomeadamente, `df_linhas.csv`, `df_vendas.csv`, `df_clientes.csv` e `df_clientes_enriquecido.csv` (produzidos pelos NB1 e NB2). Por este motivo, estes ficheiros não são integrados na pasta `Datasets/` listada na secção 2 e, consequentemente, no repositório publicado.

Contudo, pretende-se que a estrutura dos ficheiros utilizados nos diferentes notebooks seja facilmente reconhecida e compreendida, pelo que é disponibilizada uma amostra ilustrativa pseudonimizada com o nome `vendas_amostra_anonimizada.csv`, na qual os identificadores pessoais diretos foram substituídos por valores fictícios (`Nome do Cliente` → `Nome_Fictício_XXXX`, `E-Mail` → `clienteXXXX@exemplo.pt`, `Telemóvel` → `900 000 XXX`, `Ref Cliente` → `Cliente_XXXX`). Importa salientar que todos os atributos transacionais permaneceram com a mesma estrutura da base de dados originalmente exportada da plataforma de e-commerce.

De realçar que qualquer resultado obtido com base na amostra pseudonimizada não replica os documentados nesta dissertação, na medida em que todas as análises e procedimentos se sustentaram na base de dados original completa.

Os restantes ficheiros publicados no repositório (notebooks, modelos `.pkl` em `modelos/` e métricas/regras em `Resultados/`) contêm apenas identificadores internos, variáveis agregadas e resultados estatísticos, não permitindo a identificação de clientes individuais.

O acesso à base de dados original e aos ficheiros com dados pessoais pode ser solicitado ao autor, mediante acordo de confidencialidade com a empresa parceira.

A necessidade de leitura formal do trabalho (justificações teóricas detalhadas, posicionamento na literatura, discussão integrada) pode ser satisfeita através da consulta do documento `Dissertacao_Miguel_Loureiro_2220134.docx`.
