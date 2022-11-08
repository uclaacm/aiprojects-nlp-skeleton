import torch
import pandas as pd
import numpy as np

class EmbeddingsDataset(torch.utils.data.Dataset):
    """
    https://www.kaggle.com/code/christofhenkel/how-to-preprocessing-when-using-embeddings
    ----------
    
    """

    # TODO: dataset constructor.
    def __init__(self, data_path, embeddings, preprocessed_path=None):
        '''
        data_path (str): path for the csv file that contains the data that you want to use
        '''

        # check if pre processed path exists, if so return

        self.embeddings = embeddings
        # Preprocess the data. These are just library function calls so it's here for you
        self.df = pd.read_csv(data_path)
        self.raw_train = self.df.question_text.tolist()
        self.processed_train = self.embeddings.preprocess(self.raw_train)
        
    def __getitem__(self, i):
        return self.embeddings.convert_to_embedding(self.processed_train[i])

    def __len__(self):
        return len(self.processed_train)