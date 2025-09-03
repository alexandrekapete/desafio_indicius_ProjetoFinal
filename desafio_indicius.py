import pandas as pd
import numpy as np
import joblib

# Carregar modelo
model = joblib.load('../models/imdb_rating_model.pkl')

# A qui foram coletado Dados do filme para previsão
filme = {
    'Meta_score': 80.0,
    'Runtime': 142,
    'No_of_Votes': 2343110,
    'Gross': 28341469,
    'Genre': 'Drama'
}

# Pré-processamento
filme['No_of_Votes_log'] = np.log1p(filme['No_of_Votes'])
filme['Gross_log'] = np.log1p(filme['Gross'])

# Criar dataframe com as mesmas features do modelo
genres = [g.replace('Genre_', '') for g in model.feature_names_in_ if g.startswith('Genre_')]
data = {
    'Meta_score': [filme['Meta_score']],
    'Runtime': [filme['Runtime']],
    'No_of_Votes_log': [filme['No_of_Votes_log']],
    'Gross_log': [filme['Gross_log']],
}
for g in genres:
    data[f'Genre_{g}'] = [1 if g == filme['Genre'] else 0]

X_new = pd.DataFrame(data)

# Prever nota IMDB
nota_predita = model.predict(X_new)[0]
print(f'Nota IMDB prevista: {nota_predita:.2f}')
