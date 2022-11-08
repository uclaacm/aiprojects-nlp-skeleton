import re
from gensim.models import KeyedVectors

class GoogleNewsEmbeddor:
    def __init__(self, data_path, DEBUG=False):
        self.DEBUG = DEBUG
        # load embedding
        self.data_path = data_path
        if(self.DEBUG):
            print("="*10)
            print("Loading embeddings")
        self.embeddings = KeyedVectors.load_word2vec_format(data_path, binary=True)
        if(self.DEBUG):
            print("Finished loading embeddings")
            print("="*10)

    def preprocess(self, data):
        if(self.DEBUG):
            print("="*10)
            print("Starting data preprocessing")
        data = list(map(self.__preprocess_calls, data))
        if(self.DEBUG):
            print("Finished data preprocessing")
            print("="*10)
        return data

    def __preprocess_calls(self, sentence):
        sentence = sentence.lower()
        sentence = self.remove_punctuation(sentence)
        sentence = self.remove_number(sentence)
        words = self.remove_stopwords(sentence)
        return words

    def convert_to_embedding(self, words):
        vector = [self.embeddings[word] for word in words if word in self.embeddings]
        return vector

    def remove_punctuation(self, sentence):
        # adds space to \ ' - 
        sentence = re.sub(r"[\\'-]", " ", sentence)
        
        sentence = re.sub(r"&", " & ", sentence)

        # completely deletes the rest
        sentence = re.sub(r"[^\w\s^\\'-]", "", sentence)
        
        return sentence
    
    def remove_number(self, sentence):
        sentence = re.sub(r"[\d{2,}]", "#", sentence)
        return sentence

    def remove_stopwords(self, sentence):
        stop_words = ["a", "to", "of", "and"]
        words = sentence.split()

        filtered = [word for word in words if not word in stop_words]
        
        return filtered
    
    # TODO: add misspelled correcter