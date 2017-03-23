from __future__ import print_function
import numpy as np

np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Dropout, Activation, Embedding, Merge, merge, Input
from keras.layers.advanced_activations import ParametricSoftplus, ELU, ThresholdedReLU, SReLU
# from keras.regularizers import l1l2
# from keras.datasets import imdb
#from keras.callbacks import Callback, EarlyStopping, History, ModelCheckpoint
#from itertools import islice
import argparse
import sys
import os
import gensim
#from six import iteritems
import w2v_handler
#from huffmax import HuffmaxClassifier



# Relevant exempel kode: https://keras.io/getting-started/functional-api-guide/


class Word2VecInKeras:
    def __init__(self, vector_dim=100):
        self.unique_word_count = None
        self.vector_dim = vector_dim
        self.keras_model = None
        self.vocab = None
        self.x_2d_np_array = None
        self.y_2d_np_array = None


    def make_training_data(self, load_text_filename, window_size=5, min_word_count=5, sg=1, one_hot=1):
        """
        Make keras training data in a CBOW or Skip-Gram fashion from a sequence of sentences.
        Each sentence is a list of string tokens, which are looked up in the
        vocab dictionary.
        """
        sentences = []
        if load_text_filename == None:
            print('\nNo input file given using Brown corpus for training ...')
            from nltk.corpus import brown
            sentences = list(brown.sents()) #[:10000]
        else:
            print('\nGetting all sentences from "' + load_text_filename + '" as list of string tokens ...')
            sentences = []
            with open(load_text_filename, 'rb') as file:
                for line in file:
                    line = line.decode('utf-8').strip()
                    sentence = line.split()
                    sentences.append(sentence)
        print(str(len(sentences)) + ' sentences loaded ...')

        print('\nMaking vocabulary ...')
        gensim_model = gensim.models.Word2Vec(min_count=min_word_count, window=window_size)
        gensim_model.build_vocab(sentences, trim_rule=None)
        self.vocab = gensim_model.wv.vocab
        self.unique_word_count = len(self.vocab)

        print('\nMaking training data ...')
        #self.x_index_list = []
        #self.y_index_list = []
        self.x_2d_np_array = np.zeros((len(sentences), self.unique_word_count), dtype=np.int8)
        self.y_2d_np_array = np.zeros((len(sentences), self.unique_word_count), dtype=np.int8)
        train_word_count = 0
        for i, sentence in enumerate(sentences):
            word_vocabs = [self.vocab[w] for w in sentence if w in self.vocab and self.vocab[w].sample_int > gensim_model.random.rand() * 2 ** 32]
            for pos, word in enumerate(word_vocabs):
                reduced_window = gensim_model.random.randint(gensim_model.window)  # `b` in the original word2vec code

                # Now go over all words from the (reduced) window, predicting each one in turn
                start = max(0, pos - gensim_model.window + reduced_window)

                if one_hot: # Each training example consists of one hot on the input(x) and output(y)
                    for pos2, word2 in enumerate(word_vocabs[start:(pos + gensim_model.window + 1 - reduced_window)], start):
                        # Don't train on the `word` itself
                        if pos2 != pos and word2 is not None:
                            if sg:
                                self.x_2d_np_array[i][word.index] = 1
                                self.y_2d_np_array[i][word2.index] = 1
                            else:
                                self.x_2d_np_array[i][word2.index] = 1
                                self.y_2d_np_array[i][word.index] = 1

                else: # Each training example consists of possibly N hots on the input(x) or output(y)
                    word2_index_list = [word2.index for pos2, word2 in enumerate(word_vocabs[start:(pos + gensim_model.window + 1 - reduced_window)], start) if (pos2 != pos and word2 is not None)]
                    if sg:
                        self.x_2d_np_array[i][word.index] = 1
                        for index in word2_index_list:
                            self.y_2d_np_array[i][index] = 1
                    else:
                        for index in word2_index_list:
                            self.x_2d_np_array[i][index] = 1
                        self.y_2d_np_array[i][word.index] = 1
            train_word_count += len(word_vocabs)


    def build_keras_model(self, activation='softmax', loss='categorical_crossentropy', optimizer='adam'):
        #See: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        print('\nBuild keras model ...')

        input_layer = Input(shape=(self.unique_word_count,), name='input_layer')
        hidden_layer = Dense(output_dim=self.vector_dim, trainable=True, activation='linear', name='hidden_layer', bias=True)(input_layer)
        output_layer = Dense(output_dim=self.unique_word_count, trainable=True, activation=activation, name='output_layer', bias=True)(hidden_layer)

        self.keras_model = Model(input=[input_layer], output=[output_layer])
        self.keras_model.compile(loss=loss, optimizer=optimizer)


    def train_keras_model(self, batch_size, nb_epoch, verbose=1):
        # verbose: 0=silent, 1=normal, 2=minimal
        print('\nTraining ...')
        self.keras_model.fit(self.x_2d_np_array, self.y_2d_np_array, batch_size=batch_size, nb_epoch=nb_epoch, verbose=verbose)


    def save_weights_as_word2vec_model(self, save_model_filename, layer_name='hidden_layer'):
        print('\nSave weights as a word2vec model ...')
        #edge_weights, bias_weights = self.keras_model.get_layer('hidden_layer').get_weights()
        edge_weights = self.keras_model.get_layer(layer_name).get_weights()[0]

        #context_weights = np.array(edge_weights, dtype=np.float32).transpose()
        context_weights = edge_weights

        print('Vocab: ' + str(len(context_weights))) # 50
        print('Dim: ' + str(len(context_weights[0]))) # 112
        #print('Type: ' + str(type(context_weights)))

        save_filename, file_extension = os.path.splitext(save_model_filename)
        w2v_handler.save_word2vec_format(context_weights, self.vocab, save_filename=save_model_filename, fvocab=save_filename+'.vocab', binary=True)


    def batch_data_generator(self, batch_size):
        start_index = 0
        while True:
            end_index = start_index + batch_size
            if end_index > len(self.x_2d_np_array):
                end_index = len(self.x_2d_np_array)
                if end_index == start_index:
                    start_index = 0
                    continue
            x_np_batch = self.x_2d_np_array[start_index:end_index]
            y_np_batch = self.y_2d_np_array[start_index:end_index]
            start_index = end_index
            yield (x_np_batch, y_np_batch)


    def test_query_model(self, model_filename, q_word_list, topn=10):
        w2v_model = w2v_handler.W2vModel()
        w2v_model.load_w2v_model(fname=model_filename, binary=True)
        print('\nTesting the model ...')
        for q in q_word_list:
            print('\n' + q)
            if w2v_model.in_vocab(q):
                results = w2v_model.most_similar(q, topn=topn)
                for r in results:
                    print('\t%s\t%0.4f' % (r[0], r[1]))
            else:
                print('\tnot in vocab!')



if __name__ == "__main__":
    """
    ## LOCAL TESTING ###################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='my_keras_word2vec.py')
    parser.add_argument('-input', type=str, help='Text filename for training.', default='data/documents/lorem.txt') #default=None)
    parser.add_argument('-output', type=str, help='Save as word2vec model, using binary format (recommend using filetype .bin).', default='data/models/lorem-test.bin') # required=True)
    parser.add_argument('-dim', type=int, help='Vector dimensionality; default=100.', default=100)
    parser.add_argument('-window', type=int, help='Window size; default=5.', default=5)
    parser.add_argument('-min_count', type=int, help='Minimum word count for inclusion; default=1.', default=1)
    parser.add_argument('-sg', type=int, help='Conduct Skip-Gram (word inn, context out) training, if not CBOW (context inn, word out) will be used; default=1 (Skip-Gram).', default=1)
    parser.add_argument('-one_hot', type=int, help='Use one hot input and output, if not N hot is allowed; default=1.', default=1)
    parser.add_argument('-activation', type=str, help='Activation function for the output layer; default="softmax".', default='softmax')
    parser.add_argument('-loss', type=str, help='Loss function for the model; default="categorical_crossentropy".', default='categorical_crossentropy')
    parser.add_argument('-optimizer', type=str, help='Optimizer for the model; default="adam".', default='adam')
    parser.add_argument('-batch_size', type=int, help='Batch size; default=8.', default=8)
    parser.add_argument('-epoch', type=int, help='Number of training epochs; default=2.', default=2)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    """
    ## EVEX RUN ########################%%%%%%%%%%%%%%%%%%%%%
    parser = argparse.ArgumentParser(description='my_keras_word2vec.py')
    parser.add_argument('-input', type=str, help='Text filename for training.', default=None)
    parser.add_argument('-output', type=str, help='Save as word2vec model, using binary format (recommend using filetype .bin).', required=True)
    parser.add_argument('-dim', type=int, help='Vector dimensionality; default=100.', default=100)
    parser.add_argument('-window', type=int, help='Window size; default=5.', default=5)
    parser.add_argument('-min_count', type=int, help='Minimum word count for inclusion; default=5.', default=5)
    parser.add_argument('-sg', type=int, help='Conduct Skip-Gram (word inn, context out) training, if not CBOW (context inn, word out) will be used; default=1 (Skip-Gram).', default=1)
    parser.add_argument('-one_hot', type=int, help='Use one hot input and output, if not N hot is allowed; default=0.', default=1)
    parser.add_argument('-activation', type=str, help='Activation function for the output layer; default="softmax".', default='softmax')
    parser.add_argument('-loss', type=str, help='Loss function for the model; default="categorical_crossentropy".', default='categorical_crossentropy')
    parser.add_argument('-optimizer', type=str, help='Optimizer for the model; default="adam".', default='adam') # sgd
    parser.add_argument('-batch_size', type=int, help='Batch size; default=32.', default=32)
    parser.add_argument('-epoch', type=int, help='Number of training epochs; default=1.', default=1)
    ####################################%%%%%%%%%%%%%%%%%%%%%
    args = parser.parse_args(sys.argv[1:])


    if not os.path.isdir(os.path.dirname(args.output)):
        print('\nThe directory for the given -output file, "' + args.output + '", does not exist. Try again!')
        sys.exit(-1)


    wv = Word2VecInKeras(vector_dim=args.dim)

    wv.make_training_data(load_text_filename=args.input, window_size=args.window, min_word_count=args.min_count, sg=args.sg, one_hot=args.one_hot)

    wv.build_keras_model(activation=args.activation, loss=args.loss, optimizer=args.optimizer)

    wv.train_keras_model(batch_size=args.batch_size, nb_epoch=args.epoch)

    wv.save_weights_as_word2vec_model(args.output)

    wv.test_query_model(model_filename=args.output, q_word_list=['America', 'American', 'she', 'house', 'face'], topn=10)

