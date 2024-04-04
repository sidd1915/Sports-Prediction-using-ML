import pandas as pd
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

def word_embed():
    import pandas as pd
    from gensim.models import Word2Vec
    from nltk.tokenize import word_tokenize

    # Load preprocessed data
    df = pd.read_excel(
        'preprocessed_complaints.xlsx')  # Assuming you have preprocessed your data and saved it to 'preprocessed_complaints.xlsx'

    # Tokenize preprocessed text
    tokenized_text = df['preprocessed_text'].apply(word_tokenize)

    # Train Word2Vec model
    word2vec_model = Word2Vec(sentences=tokenized_text, vector_size=100, window=5, min_count=1, workers=4)

    # Save Word2Vec model
    word2vec_model.save("word2vec_model.model")

    # Load Word2Vec model
    # word2vec_model = Word2Vec.load("word2vec_model.model")

    # Example usage of Word2Vec embeddings
    word_embeddings = word2vec_model.wv

    # Get word vector for a specific word
    word_vector = word_embeddings['loan']

    print("Word vector for 'loan':", word_vector)
