from keras.callbacks import EarlyStopping
from Token import Token
from NMT import NMT
#from NMTatt import NMT

en = Token('train.en')
ja = Token('train.ja')
en_train = en.sequences
ja_train = ja.sequences
en_test = en.tokenize('test.en')
ja_test = ja.tokenize('test.ja')

en2ja = NMT(en.length, en.vocab_size, ja.length, ja.vocab_size)

en2ja.train(en_train, ja_train, en_test, ja_test,
            batch_size=128, epochs=20,
            callbacks=[EarlyStopping(patience=10)])

en2ja.save('en2ja.hdf5')

ja2en = NMT(ja.length, ja.vocab_size, en.length, en.vocab_size)

ja2en.train(ja_train, en_train, ja_test, en_test,
            batch_size=128, epochs=20,
            callbacks=[EarlyStopping(patience=10)])

ja2en.save('ja2en.hdf5')
