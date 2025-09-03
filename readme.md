git clone https://github.com/seu-usuario/analise-filmes-pproductions.git
cd analise-filmes-pproductions

python -m venv venv
source venv/bin/activate  # 
# ou
venv\Scripts\activate  # Windows

pip install -r requirements.txt

jupyter notebook desafio_indicium_imdb_analysis.ipynb
python predict_imdb.py


analise-filmes-pproductions/
│
├── data/
│   └── desafio_indicium_imdb.csv   # Dataset original
│
├── models/
│   └── imdb_predictor.pkl          # Modelo treinado
│
├── notebooks/
│   └── desafio_indicium_imdb_analysis.ipynb  # Análise completa
│
├── reports/
│   └── analysis_report.pdf         # Relatório da análise
│
├── requirements.txt                # Dependências do projeto
├── predict_imdb.py                # Script de previsão
└── README.md                      # Este arquivo



# Indicium IMDB – EDA & Modelagem (entrega rápida)

## Artefatos incluídos
- `EDA_Indicium_IMDB_Report.pdf` – relatório com gráficos essenciais.
- `imdb_rating_model.pkl` – modelo de regressão (Ridge) para prever `IMDB_Rating`.
- `overview_to_genre_model.pkl` – classificador de gênero a partir da `Overview` (TF‑IDF + Regressão Logística).
- `regression_metrics.csv` – métricas (RMSE, MAE, R²).
- `key_answers.json` – arquivo com respostas-chave (modelo escolhido, métricas e predição do exemplo).

## Como instalar
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
pip install -r requirements.txt
```

## Como treinar/rodar
Crie um script baseado no notebook/roteiro usado aqui (pipeline de EDA + treinamento) e aponte para o CSV original. 
O pipeline:
1. Faz limpeza e engenharia de atributos (ano, duração, metascore, votos, bilheteria, certificado, gênero primário).
2. Treina um **Ridge** para nota IMDB (regressão), com `OneHotEncoder` e imputação.
3. Treina um classificador **Logistic Regression** (TF‑IDF) para inferir gênero a partir de `Overview` (classificação multiclasse).

## Métrica do modelo de nota IMDB
- RMSE (mesma unidade da nota, penaliza mais erros maiores), acompanhado de MAE e R².

