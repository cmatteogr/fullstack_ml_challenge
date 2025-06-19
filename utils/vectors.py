import numpy as np

def tokens_to_vector(tokens, model):
    # Get vectors for each word in the sentence
    word_vectors = [model.wv[word] for word in tokens if word in model.wv]
    # If no valid words, return a zero vector
    if not word_vectors:
        return np.zeros(model.vector_size)
    # Aggregate word vectors (e.g., by averaging)
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector


def get_tokens_from_sentences(nlp, sentence_pack):
    return [[token.text for token in nlp(sentence)] for sentence in sentence_pack]