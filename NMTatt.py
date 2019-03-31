# reference:
#   http://dl4us.com
# uses Keras
#   https://keras.io/ja

import numpy as np
from keras.models import Model
from keras.layers import Input, Embedding, LSTM, Dense, dot, Activation, concatenate

class NMT:
    def __init__(self, x_len, x_vocab_size, y_len, y_vocab_size, emb_dim=256, hid_dim=256, att_dim=256):
        """
        x_len, y_len = length of sentences in source and target language
        x_vocab_size, y_vocab_size = vocaburary of source and target language
        emb_dim = dimension of embedding layer
        hid_dim = dimension of hidden layer
        att_dim = dimension of attention layer
        """
        encoder_inputs = Input(shape=(x_len,))
        encoder_embeding = Embedding(x_vocab_size, emb_dim, mask_zero=True)
        encoder_embeded = encoder_embeding(encoder_inputs)
        encoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
        encoded_seq, *encoder_states = encoder_lstm(encoder_embeded)

        decoder_inputs = Input(shape=(y_len,))
        decoder_embeding = Embedding(y_vocab_size, emb_dim)
        decoder_embeded = decoder_embeding(decoder_inputs)
        decoder_lstm = LSTM(hid_dim, return_sequences=True, return_state=True)
        decoded_seq, _, _ = decoder_lstm(decoder_embeded, initial_state=encoder_states)

        score_dense = Dense(hid_dim, use_bias=False)
        score = score_dense(decoded_seq)                        # shape: (y_len, hid_dim) -> (y_len, hid_dim)
        score = dot([score, encoded_seq], axes=(2,2))           # shape: [(y_len, hid_dim), (x_len, hid_dim)] -> (y_len, x_len)
        attention = Activation('softmax')(score)                # shape: (y_len, x_len) -> (y_len, x_len)
        context = dot([attention, encoded_seq], axes=(2,1))     # shape: [(y_len, x_len), (x_len, hid_dim)] -> (y_len, hid_dim)
        concat = concatenate([context, decoded_seq], axis=2)    # shape: [(y_len, hid_dim), (y_len, hid_dim)] -> (y_len, 2*hid_dim)
        attention_dense = Dense(att_dim, activation='tanh')
        attentional = attention_dense(concat)                   # shape: (y_len, 2*hid_dim) -> (y_len, att_dim)
        output_dense = Dense(y_vocab_size, activation='softmax')
        attention_outputs = output_dense(attentional)           # shape: (y_len, att_dim) -> (y_len, y_vocab_size)

        model = Model([encoder_inputs, decoder_inputs], attention_outputs)

        encoder_model = Model(encoder_inputs, [encoded_seq] + encoder_states)

        decoder_states_inputs = [Input(shape=(hid_dim,)), Input(shape=(hid_dim,))]
        decoder_inputs = Input(shape=(1,))
        decoder_embeded = decoder_embeding(decoder_inputs)
        decoded_seq, *decoder_states = decoder_lstm(decoder_embeded, initial_state=decoder_states_inputs)
        decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoded_seq] + decoder_states)

        encoded_seq_in, decoded_seq_in = Input(shape=(x_len, hid_dim)), Input(shape=(1, hid_dim))
        score = score_dense(decoded_seq_in)
        score = dot([score, encoded_seq_in], axes=(2,2))
        attention = Activation('softmax')(score)
        context = dot([attention, encoded_seq_in], axes=(2,1))
        concat = concatenate([context, decoded_seq_in], axis=2)
        attentional = attention_dense(concat)
        attention_outputs = output_dense(attentional)
        attention_model = Model([encoded_seq_in, decoded_seq_in], attention_outputs)

        self.model = model
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.attention_model = attention_model


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
        encoded_seq, *states_value = self.encoder_model.predict(input_seq)

        target_seq = np.array([bos])
        output_seq = [bos]
    
        while len(output_seq) < max_output_length:
            decoded_seq, *states_value = self.decoder_model.predict([target_seq] + states_value)
            output_tokens = self.attention_model.predict([encoded_seq, decoded_seq])
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            output_seq.append(sampled_token_index)
        
            if sampled_token_index == eos: break

            target_seq = np.array([sampled_token_index])

        return output_seq

    def save(self, filename):
        self.model.save_weights(filename)
        
    def load(self, filename):
        self.model.load_weights(filename)
