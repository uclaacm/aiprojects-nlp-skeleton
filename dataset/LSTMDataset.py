import torch

class LSTMDataset(torch.utils.data.Dataset):
    def __init__(self, df, vocab, max_seq_length, pad_token, unk_token):
        # make a list of our labels
        self.labels = df.target.tolist()

        # make a dictionary converting each word to its id in the vocab, as well
        # as the reverse lookup
        self.word2idx = {term:idx for idx,term in enumerate(vocab)}
        self.idx2word = {idx:word for word,idx in self.word2idx.items()}

        self.pad_token,self.unk_token = pad_token,unk_token

        self.input_ids = []
        self.sequence_lens = []
        self.labels = []

        for i in range(df.shape[0]):
            # clean up each sentence and turn it into tensor containing the
            # token ids of each word. Also add padding to make them all the
            # same length as the longest sequence
            input_ids,sequence_len = self.convert_text_to_input_ids(
                df.iloc[i].question_text,
                pad_to_len = max_seq_length)

            self.input_ids.append(input_ids.reshape(-1))
            self.sequence_lens.append(sequence_len)
            self.labels.append(df.iloc[i].target)

        #sanity checks
        assert len(self.input_ids) == df.shape[0]
        assert len(self.sequence_lens) == df.shape[0]
        assert len(self.labels) == df.shape[0]

    def convert_text_to_input_ids(self,text,pad_to_len):
        # truncate excess words (beyond the length we should pad to)
        words = text.strip().split()[:pad_to_len]

        # add padding till we've reached desired length
        deficit = pad_to_len - len(words)
        words.extend([self.pad_token]*deficit)

        # replace words with their id
        for i in range(len(words)):
            if words[i] not in self.word2idx:
                # if word is not in vocab, then use <unk> token
                words[i] = self.word2idx[self.unk_token]
            else:
                # else find the id associated with the word
                words[i] = self.word2idx[words[i]]
        return torch.Tensor(words).long(),pad_to_len - deficit

    def __len__(self):
        # Make dataset compatible with len() function
        return len(self.input_ids)

    def __getitem__(self, i):
        # for the ith indexm return a dictionary containing id, length and label
        sample_dict = dict()
        sample_dict['input_ids'] = self.input_ids[i].reshape(-1)
        sample_dict['sequence_len'] = torch.tensor(self.sequence_lens[i]).long()
        sample_dict['labels'] = torch.tensor(self.labels[i]).type(torch.FloatTensor)
        return sample_dict