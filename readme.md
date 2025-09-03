## Desafios final sobre banco de dados ##

https://github.com/alexandrekapete/desafio_indicius_ProjetoFinal.git

## Descrição e breve do projeto ## 

Você foi alocado em um time da Indicium contratado por um estúdio de Hollywood chamado PProductions, e agora deve fazer uma análise em cima de um banco de dados cinematográfico para orientar qual tipo de filme deve ser o próximo a ser desenvolvido. Lembre-se que há muito dinheiro envolvido, então a análise deve ser muito detalhada e levar em consideração o máximo de fatores possíveis (a introdução de dados externos é permitida - e encorajada).

python -m venv venv
source venv/bin/activate  # 
# Como clonar o repositorio ##

# 1.. Copiar a URL do repositório
Vá até o repositório que deseja clonar (ex: no GitHub).
Clique no botão Code (ou Código).
Copie a URL HTTPS (ex: https://github.com/usuario/repositorio.git).
# 2.. Abrir o VS Code
Abra o Visual Studio Code.
# 3.. Clonar o repositório
No VS Code, abra o Painel de Controle do Git clicando no ícone de ramificação no lado esquerdo (ou pressione Ctrl+Shift+G).
Clique nos três pontinhos ... no topo do painel Git.
Selecione Clone.
## Cole a URL do repositório copiada e pressione Enter.
Escolha a pasta onde deseja salvar o repositório clonado.
Aguarde o download dos arquivos.
Passo 4: Abrir a pasta do repositório clonado
Após o clone, o VS Code perguntará se deseja abrir a pasta do repositório clonado ##.
## Clique em Abrir.

## 5.. Usar o repositório
Agora você pode editar, executar e versionar o código localmente.
Use o terminal integrado para comandos Git adicionais, se desejar.


## EStrutura do versioamento do banco de dados ##

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

