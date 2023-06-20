import tensorflow as tf
import tensorflow_hub as hub

embed = hub.KerasLayer('Word2Vec/Wiki-words-250-with-normalization_2')
embeddings = embed(["A long sentence.", "single-word", "http://example.com"])
print(embeddings.shape, embeddings.dtype)