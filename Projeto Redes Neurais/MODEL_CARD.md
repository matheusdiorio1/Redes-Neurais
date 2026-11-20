# Model Card — Previsão de Cancelamento (Telco)
- **Uso pretendido:** apontar clientes com alto risco de cancelar para o time de retenção oferecer ajuda ou desconto.
- **Não usar para:** negar atendimento, piorar condições ou cobrar mais de quem tem "risco alto"; nem decidir sozinho, sem revisão humana.
- **Dados:** 7043 clientes (Telco Customer Churn, Kaggle); 27% cancelaram.
- **Modelo:** rede neural Keras (32 neurônios, dropout, early stopping), comparada a regressão logística.
- **Desempenho no teste:** AUC 0.841, recall 76%, precisão 52% (baseline: AUC 0.842, recall 78%).
- **Limitações:** dados de uma única operadora; não sabe o motivo do cancelamento; limiar 0.5 pode ser ajustado conforme o custo da oferta; re-treinar com dados novos.
