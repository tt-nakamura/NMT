from Token import Token
from NMT import NMT
#from NMTatt import NMT

en = Token('train.en')
ja = Token('train.ja')
en_test = en.tokenize('test.en')
ja_test = ja.tokenize('test.ja')

en2ja = NMT(en.length, en.vocab_size, ja.length, ja.vocab_size)
en2ja.load('en2ja.hdf5')

ja2en = NMT(ja.length, ja.vocab_size, en.length, en.vocab_size)
ja2en.load('ja2en.hdf5')

for e,j in zip(en_test, ja_test):
    jt = en2ja.translate(e)
    et = ja2en.translate(j)
    print(' '.join(en.detokenize(e)), end='')
    print(' '.join(ja.detokenize(jt)), end='')
    print(' '.join(ja.detokenize(j)), end='')
    print(' '.join(en.detokenize(et)))
