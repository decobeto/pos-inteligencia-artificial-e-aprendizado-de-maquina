import pandas as pd
import re
import nltk
import spacy
from unidecode import unidecode
import time

print("Iniciando a configuração do ambiente...")
try:
    nlp = spacy.load("pt_core_news_sm")
    print("Modelo 'pt_core_news_sm' do spaCy já está carregado.")
except OSError:
    print("Modelo 'pt_core_news_sm' não encontrado. Baixando e instalando...")
    import spacy.cli
    spacy.cli.download("pt_core_news_sm")
    nlp = spacy.load("pt_core_news_sm")
    print("Modelo baixado e carregado com sucesso.")

try:
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    print("Pacote 'stopwords' do NLTK já está disponível.")
except LookupError:
    print("Pacote 'stopwords' não encontrado. Baixando...")
    nltk.download('stopwords')
    stopwords_pt = nltk.corpus.stopwords.words('portuguese')
    print("Pacote baixado com sucesso.")

def preprocessar_texto(texto):
    if not isinstance(texto, str):
        return []

    texto = texto.lower()
    
    texto = re.sub(r'\d+', '', texto)
    texto = re.sub(r'[^\w\s]', '', texto)
    
    texto = unidecode(texto)

    doc = nlp(texto)
    tokens_processados = [
        token.lemma_ 
        for token in doc 
        if token.text not in stopwords_pt and not token.is_space and len(token.text) > 2
    ]
    
    return tokens_processados

print("\n--- Carregando o Dataset ---")
nome_do_arquivo = 'olist_order_reviews_dataset.csv'
try:
    df = pd.read_csv(nome_do_arquivo)
    print(f"Arquivo '{nome_do_arquivo}' carregado com sucesso!")
    print(f"O dataset possui {len(df)} linhas.")
    
    print("\nColunas originais e 5 primeiras linhas (antes do processamento):")
    print(df[['review_comment_title', 'review_comment_message']].head())

except FileNotFoundError:
    print(f"ERRO: O arquivo '{nome_do_arquivo}' não foi encontrado.")
    print("Verifique se ele está na mesma pasta que este script.")
    exit()

print("\n--- Iniciando o Pré-processamento dos Textos ---")
start_time = time.time()

df['texto_completo'] = df['review_comment_title'].fillna('') + ' ' + df['review_comment_message'].fillna('')

df.dropna(subset=['texto_completo'], inplace=True)
df = df[df['texto_completo'].str.strip() != '']

df['texto_processado'] = df['texto_completo'].apply(preprocessar_texto)

end_time = time.time()
print(f"Pré-processamento concluído em {end_time - start_time:.2f} segundos.")
print("\n--- Resultado do Processamento (Antes e Depois) ---")
print(df[['texto_completo', 'texto_processado']].head(10))
print("\nScript finalizado com sucesso!")