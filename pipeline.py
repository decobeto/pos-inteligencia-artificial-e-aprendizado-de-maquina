import pandas as pd
import re
from nltk.corpus import stopwords
import spacy

# Carrega o modelo de língua portuguesa do spaCy
nlp = spacy.load("pt_core_news_sm")
# Carrega a lista de stopwords em português
stop_words = set(stopwords.words('portuguese'))

def preprocessar_texto(texto):
    # Etapa 2: Normalização (minúsculas e remoção de caracteres)
    texto = texto.lower()
    texto = re.sub(r'[^a-zA-Z\s]', '', texto, re.I|re.A)
    
    # Etapa 3: Tokenização (implícita no processamento do spaCy)
    # Etapa 5: Lematização
    doc = nlp(texto)
    tokens_lematizados = [token.lemma_ for token in doc if token.text not in stop_words and not token.is_space]
    
    # Etapa 4: Remoção de Stopwords (feita durante a lematização)
    
    return tokens_lematizados

# Exemplo de aplicação em uma coluna do dataframe
# df['texto_processado'] = df['texto_completo'].apply(preprocessar_texto)