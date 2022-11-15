from data.EmbeddingDataset import EmbeddingsDataset
from data.embedWrapping import GoogleNewsEmbeddor

embedding = GoogleNewsEmbeddor("./data/embeddings/GoogleNews-vectors-negative300.bin", DEBUG=True)
dataset = EmbeddingsDataset("./train.csv", embedding)

print(dataset[0][0].shape)