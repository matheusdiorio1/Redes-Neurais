# NeuroStartup — Previsão de Cancelamento de Clientes

## Integrantes
| Nome | RA |
|------|----|
| Matheus Diório | 2743334 |
| Eduardo Dias Carotenuto | 3064556 |
| Daniel Gomes Santiago | 2681939 |
| Fauzer Silva | 2768291 |
| Everton Tiburcio | 2775959 |

## Problema
Uma operadora de telecom quer identificar clientes com risco de cancelar para oferecer retenção antes da saída.
Dataset: [Telco Customer Churn (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## Como rodar
1. Baixe `WA_Fn-UseC_-Telco-Customer-Churn.csv` do Kaggle e coloque na mesma pasta do notebook.
2. `pip install numpy pandas matplotlib scikit-learn tensorflow ipywidgets joblib`
3. Abra `neurostartup_churn.ipynb` e rode todas as células na ordem.

## Resultados

Avaliação no conjunto de teste (aberto uma única vez, 1.409 clientes, 374 cancelaram):

| Modelo | AUC | Recall | Precisão |
|---|---|---|---|
| Baseline: regressão logística | 0.842 | 78,3% | 50,7% |
| Rede neural Keras (32 neurônios) | 0.841 | 75,7% | 51,9% |

Tentativas de arquitetura registradas em `experimentos.csv` (16, 32 e 64 neurônios);
a de 32 foi escolhida pela AUC na validação.

**Leitura das curvas:** nas três redes, o erro de treino continuou caindo, mas o de validação
parou de melhorar depois de algumas épocas. O early stopping interrompeu o treino nesse ponto
e restaurou os pesos da melhor época, evitando overfitting.

**Frase de negócio:** dos 374 clientes que cancelaram no teste, a rede identificou 283 (76%).
A cada 100 clientes sinalizados, cerca de 52 realmente cancelariam. O baseline simples
identificou 78% e teve desempenho equivalente.

**Conclusão:** neste conjunto, a rede neural não superou a regressão logística. Como os
dados são tabulares e pequenos, um modelo simples já captura quase todo o padrão. Em
produção, a regressão logística seria a escolha mais prática (mais simples, mais fácil de
explicar e de manter), e a rede só se justificaria com mais dados ou mais variáveis.

## Tela do produto
![print da tela](print_tela.png)

## Arquivos
- `neurostartup_churn.ipynb`: notebook completo (rodado)
- `experimentos.csv`: todas as tentativas
- `modelo_churn.keras` e `preprocessador.joblib`: modelo salvo
- `MODEL_CARD.md`: a "bula" do modelo

## Declaração de uso de IA
(mantenha o texto que você já tem)
