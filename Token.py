# reference:
#   http://dl4us.com
# uses Keras
#   https://keras.io/ja

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

class Token:
    def __init__(self, filename):
        self.tokenizer = Tokenizer(filters="")
        texts = []
        for line in open(filename, encoding='utf-8'):
            texts.append('<s> ' + line.strip() + ' </s>')
        
        self.tokenizer.fit_on_texts(texts)
        self.sequences = self.tokenizer.texts_to_sequences(texts)
        self.sequences = pad_sequences(self.sequences, padding='post')
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.length = len(self.sequences[0])
        self.detokenizer = dict(map(reversed, self.tokenizer.word_index.items()))

    def tokenize(self, filename):
        texts = []
        for line in open(filename, encoding='utf-8'):
            texts.append('<s> ' + line.strip() + ' </s>')

        s = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(s, self.length, padding='post')
        
    def detokenize(self, seq):
        output = [self.detokenizer[i] for i in seq if i]
        return output[1:-1]
