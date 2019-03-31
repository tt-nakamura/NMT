# reference:
#   http://dl4us.com
# uses Keras
#   https://keras.io/ja

import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense

class NMT:
    def __init__(self, x_len, x_vocab_size, y_len, y_vocab_size, emb_dim=256, hid_dim=256):
        """
        x_len, y_len = length of sentences in source and target language
        x_vocab_size, y_vocab_size = vocaburary of source and target language
        emb_dim = dimension of embedding layer
        hid_dim = dimension of hidden layer
        """
        encoder_inputs = Input(shape=(x_len,))
        encoder_embedding = Embedding(x_vocab_size, emb_dim, mask_zero=True)
        encoder_embedded = encoder_embedding(encoder_inputs)
        encoder_lstm = LSTM(hid_dim, return_state=True)
        _, *encoder_states = encoder_lstm(encoder_embedded)

        decoder_inputs = Input(shape=(y_len,))
        decoder_embedding = Embedding(y_vocab_size, emb_dim)
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
        decoded_seq, _, _ = decoder_lstm(decoder_embedded, initial_state=encoder_states)
        decoder_dense = Dense(y_vocab_size, activation='softmax')
        decoder_outputs = decoder_dense(decoded_seq)

        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        encoder_model = Model(encoder_inputs, encoder_states)

        decoder_states_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]
        decoder_inputs = Input(shape=(1,))
        decoder_embedded = decoder_embedding(decoder_inputs)
        decoded_seq, *decoder_states = decoder_lstm(decoder_embedded, initial_state=decoder_states_inputs)
        decoder_outputs = decoder_dense(decoded_seq)

        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model


    def train(self, x_train, y_train, x_test, y_test, **kwarg):
        """
        x_train, y_train = data for training source and target language
        x_test, y_test = data for testing source and target language
        kwarg = keyword argnuments passed to keras.model.fit
        """
        self.model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy')
        train_target = np.roll(y_train, -1, axis=1)[...,np.newaxis]; train_target[:,-1] = 0
        test_target = np.roll(y_test, -1, axis=1)[...,np.newaxis]; test_target[:,-1] = 0
        self.model.fit([x_train, y_train], train_target, validation_data=([x_test, y_test], test_target), **kwarg)
        
    def translate(self, input_seq, max_output_length=100):
        bos = input_seq[0]
        eos = input_seq[np.count_nonzero(input_seq)-1]

        input_seq = input_seq.reshape(1,-1)
        states_value = self.encoder_model.predict(input_seq)

        target_seq = np.array([bos])
        output_seq = [bos]

        while len(output_seq) < max_output_length:
            output_tokens, *states_value = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            output_seq.append(sampled_token_index)

            if sampled_token_index == eos: break

            target_seq = np.array([sampled_token_index])

        return output_seq

    def save(self, filename):
        self.model.save_weights(filename)
        
    def load(self, filename):
        self.model.load_weights(filename)
