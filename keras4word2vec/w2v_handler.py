from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import logging
import gensim
import numpy
import scipy.spatial.distance as dist
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#import local_word2vec
from numpy import float32 as REAL
from gensim import utils
from six import iteritems

__author__ = 'hansmoe'



# word2vec testings
# Example: https://radimrehurek.com/gensim/models/word2vec.html

# Normalize vector to unit length
def norm(v):
    return gensim.matutils.unitvec(numpy.array(v, dtype=REAL))

# Calculate cosine similarity between two vectors
def cossim(v1, v2):
    return 1 - dist.cosine(v1, v2)


def make_vocab(word_dict):
    """
    Take a list of words and convert into a vocabulary variable suitable for the word2vec tool.
    The order of the words are stored in the given order.
    :param word_list:
    :return: vocab
    """
    vocab = {}
    for i, (word, count) in enumerate(sorted(word_dict.items(), key=lambda x: x[1], reverse=True)):
        vocab[word] = Vocab(count=count, index=i)
    return vocab


class Vocab(object):
    """
    A single vocabulary item, used internally for collecting per-word frequency/sampling info,
    and for constructing binary trees (incl. both word leaves and inner nodes).

    """
    def __init__(self, **kwargs):
        self.count = 0
        self.__dict__.update(kwargs)

    def __lt__(self, other):  # used for sorting in a priority queue
        return self.count < other.count

    def __str__(self):
        vals = ['%s:%r' % (key, self.__dict__[key]) for key in sorted(self.__dict__) if not key.startswith('_')]
        return "%s(%s)" % (self.__class__.__name__, ', '.join(vals))


'''
def save_word2vec_format(vectors, fname, vocab, fvocab=None, binary=False):
    """
    Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.

     `fname` is the file used to save the vectors in
     `fvocab` is an optional file used to save the vocabulary
     `binary` is an optional boolean indicating whether the data is to be saved
     in binary word2vec format (default: False)

    """
    vector_size = vectors.shape[1]
    if fvocab is not None:
        #logger.info("storing vocabulary in %s" % (fvocab))
        with utils.smart_open(fvocab, 'wb') as vout:
            for word, vocab in sorted(iteritems(vocab), key=lambda item: -item[1].count):
                vout.write(utils.to_utf8("%s %s\n" % (word, vocab.count)))
    #logger.info("storing %sx%s projection weights into %s" % (len(vocab), vector_size, fname))
    #assert (len(vocab), vector_size) == vectors.shape
    with utils.smart_open(fname, 'wb') as fout:
        fout.write(utils.to_utf8("%s %s\n" % vectors.shape))
        # store in sorted order: most frequent words at the top
        for word, voc in sorted(iteritems(vocab), key=lambda item: -item[1].count):
            row = vectors[voc.index]
            if binary:
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))
'''

def save_word2vec_format(vectors, vocab, save_filename, fvocab=None, binary=True):
    """
    Store the input-hidden weight matrix in the same format used by the original
    C word2vec-tool, for compatibility.
     `vectors` is the numpy 2d array of context vectors to save.
     `vocab` is the vocabulary for the vectors.
     `fname` is the file used to save the vectors in
     `fvocab` is an optional file used to save the vocabulary
     `binary` is an optional boolean indicating whether the data is to be saved
     in binary word2vec format (default: False)
    """
    if fvocab is not None:
        # logger.info("storing vocabulary in %s" % (fvocab))
        with utils.smart_open(fvocab, 'wb') as vout:
            for word, voc in sorted(iteritems(vocab), key=lambda item: -item[1].count):
            #for word, vocab in sorted(vocab.items(), key=lambda item: -item[1].count):
                vout.write(utils.to_utf8("%s %s\n" % (word, voc.count)))
    # logger.info("storing %sx%s projection weights into %s" % (len(self.vocab), self.vector_size, fname))
    # assert (len(vocab), self.vector_size) == self.syn0.shape
    with utils.smart_open(save_filename, 'wb') as fout:
        fout.write(utils.to_utf8("%s %s\n" % (len(vectors), len(vectors[0])))) # vectors.shape
        # store in sorted order: most frequent words at the top
        for word, voc in sorted(iteritems(vocab), key=lambda item: -item[1].count):
        #for word, voc in sorted(vocab.items(), key=lambda item: -item[1].count):
            row = vectors[voc.index]
            if binary:
                fout.write(utils.to_utf8(word) + b" " + row.tostring())
            else:
                fout.write(utils.to_utf8("%s %s\n" % (word, ' '.join("%f" % val for val in row))))


class W2vModel:
    def __init__(self):
        self.model = None

    def load_w2v_model(self, fname, fvocab=None, binary=True):
        self.model = gensim.models.KeyedVectors.load_word2vec_format(fname=fname, fvocab=fvocab, binary=binary, encoding="utf-8", unicode_errors='ignore')
        #self.model = gensim.models.Word2Vec.load_word2vec_format(fname=fname, fvocab=fvocab, binary=binary, encoding="utf-8", unicode_errors='ignore')
        #self.model = local_word2vec.Word2Vec_local.load_word2vec_format(fname=fname, fvocab=fvocab, binary=binary, encoding="utf-8")

    # Calculate cosine similarity between two list of words
    def n_similarity_sum(self, wordlist1, wordlist2, normalize=True):
        if normalize:
            v1 = [norm(self.model[word]) for word in wordlist1]
            v2 = [norm(self.model[word]) for word in wordlist2]
            return numpy.dot(gensim.matutils.unitvec(numpy.array(v1).mean(axis=0)), gensim.matutils.unitvec(numpy.array(v2).mean(axis=0)))
        else:
            v1 = [self.model[word] for word in wordlist1]
            v2 = [self.model[word] for word in wordlist2]
            return numpy.dot(gensim.matutils.unitvec(numpy.array(v1).mean(axis=0)), gensim.matutils.unitvec(numpy.array(v2).mean(axis=0)))

    # Sum a list of words into a sentence vector
    def sent2vec_sum(self, wordlist, normalize=True):
        sent_vec = None
        for i_word in wordlist:
            if i_word in self.model:
                i_w_vec = self.model[i_word]
                if normalize:
                    i_w_vec = norm(i_w_vec)
                if sent_vec is not None:
                    #sent_vec += i_w_vec
                    #sent_vec = sent_vec + i_w_vec
                    sent_vec = numpy.add(sent_vec, i_w_vec)
                else:
                    sent_vec = i_w_vec
        return sent_vec

    # Get the vector for the given word
    def get_vec(self, word):
        return self.model[word]

    def most_similar(self, word_list, topn=10):
        return self.model.most_similar(positive=word_list, topn=topn)

    def most_similar_normalize(self, word_list, topn=10):
        sent_vec = self.sent2vec_sum(word_list, normalize=True)
        return self.model.most_similar(positive=[sent_vec], topn=topn)

    def most_similar_min_count(self, word_list, topn=10, min_count=1):
        local_topn = topn
        while True:
            sim_list = self.model.most_similar(positive=word_list, topn=len(self.model.vocab))
            final_sim_list = []
            for word, sim in sim_list:
                if self.model.vocab[word].count >= min_count:
                    final_sim_list.append((word, sim))
                    if len(final_sim_list) >= topn:
                        return final_sim_list
            if local_topn > len(self.model.vocab):
                return final_sim_list
            local_topn *= 2
        '''
        #return self.model.most_similar(positive=word_list, topn=topn)
        sim_list = self.model.most_similar(positive=word_list, topn=len(self.model.vocab))
        final_sim_list = []
        for word, sim in sim_list:
            if self.model.vocab[word].count >= min_count:
                final_sim_list.append((word, sim))
        return final_sim_list[:topn]
        '''

    def most_similar_dont_start_with(self, word_list, topn=10, min_count=1, dont_start_with='__label__'):
        local_topn = topn * 100
        while True:
            most_similar = self.model.most_similar(positive=word_list, topn=local_topn)
            final_sim_list = []
            for word, sim in most_similar:
                if (not word.startswith(dont_start_with)) and self.model.vocab[word].count >= min_count:
                    final_sim_list.append((word, sim))
                    if len(final_sim_list) >= topn:
                        return final_sim_list
            if local_topn > len(self.model.vocab):
                return final_sim_list
            local_topn *= 2
        '''
        most_similar = self.model.most_similar(positive=word_list, topn=len(self.model.vocab))
        final_sim_list = []
        for word, sim in most_similar:
            #if self.model.vocab[word].count >= min_count:
            if (not word.startswith(dont_start_with)) and self.model.vocab[word].count >= min_count:
                final_sim_list.append((word, sim))
        return final_sim_list[:topn]
        '''

    def most_similar_only_start_with(self, word_list, topn=10, min_count=1, start_with='__label__'):
        local_topn = topn * 100
        while True:
            most_similar = self.model.most_similar(positive=word_list, topn=local_topn)
            final_sim_list = []
            for word, sim in most_similar:
                if word.startswith(start_with) and self.model.vocab[word].count >= min_count:
                    final_sim_list.append((word, sim))
                    if len(final_sim_list) >= topn:
                        return final_sim_list
            if local_topn > len(self.model.vocab):
                return final_sim_list
            local_topn *= 2

        #return filter(lambda x: x[0].startswith((start_with)), most_similar)
        '''
        most_similar = self.model.most_similar(positive=word_list, topn=len(self.model.vocab))
        final_sim_list = []
        for word, sim in most_similar:
            #if self.model.vocab[word].count >= min_count:
            if word.startswith(start_with) and self.model.vocab[word].count >= min_count:
                final_sim_list.append((word, sim))
        return final_sim_list[:topn]
        '''

    def in_vocab(self, word):
        return word in self.model.vocab

    def get_word_count(self, word):
        if word in self.model.vocab:
            return self.model.vocab[word].count
        return 0

    def get_dim(self):
        #return self.model.vector_size
        return self.model.syn0.shape[1]

    def get_vocab(self):
        return self.model.vocab




#Main ...
if __name__ == "__main__":
    print("Hello World!")
    ####################################%%%%%%%%%%%%%%%%%%%%%

    '''
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    # train word2vec on the two sentences
    model = gensim.models.Word2Vec(sentences, size=200, window=5, min_count=100, workers=4)
    #model = gensim.models.Word2Vec(sentences, min_count=1)

    #model.build_vocab(sentences)
    #model.sort_vocab()
    #model.finalize_vocab()
    #model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)

    print(model.similarity('first', 'sentence'))
    '''

    m = W2vModel()
    # Word2vec from existing model
    print(" ... loading w2v model ... ")
    #model = gensim.models.Word2Vec.load_word2vec_format(fname='semantic_models/w2v-doctors-nurses-sp-skip-dim200-win5.bin', binary=True)

    #model.load_w2v_model(fname='/Users/hansmoe/Dropbox/MyProgramming/PythonProjects/PyCharm/Gensim_RI/data/models/ri-news.bin', binary=True)
    #model.load_w2v_model(fname='/Users/hansmoe/Dropbox/MyProgramming/PythonProjects/PyCharm/Gensim_RI/data/models/ri-news.bin', fvocab="/Users/hansmoe/Dropbox/MyProgramming/PythonProjects/PyCharm/Gensim_RI/data/models/ri-news.vocab", binary=True)
    #model = gensim.models.Word2Vec.load_word2vec_format(fname='data/iki_models/w2v-doctors-nurses-sp-skip-dim200-win5.bin', fvocab="data/iki_models/vocab-doctors-nurses-sp.txt", binary=True)

    #m.load_w2v_model(fname='data/iki_models/w2v-doctors-nurses-sp-skip-dim200-win5.bin', fvocab="data/iki_models/vocab-doctors-nurses-sp.txt", binary=True)
    m.load_w2v_model(fname="data/models/leena-filtered.bin", binary=True)

    print(" ... done.")
    #print(model.similarity('kipu', 'kipuilu'))
    #print(type(model['kipu']))


    s0 = m.sent2vec_sum(['kipu'], normalize=True)
    #s0 = s0.encode('utf-8')
    most_similar = m.most_similar([s0], topn=10)
    for word, sim in most_similar:
        print(word + "; " + str(sim))




    '''
    #print(model['tullebukk'])
    v1 = norm(m.get_vec('kipu')) # Normalize
    v2 = norm(m.get_vec('sekavuus')) # Normalize

    v1_2 = numpy.add(v1, v2) # Combine the vectors

    #print(model.most_similar(positive=[v1_2], topn=10)) # Query the model for the topn most similar words

    print(cossim(v1_2, v1)) # Calculate cosine similarity between two vectors

    s1 = m.sent2vec_sum(['kipu', 'sekavuus'], normalize=True)
    s2 = m.sent2vec_sum(['kipu'], normalize=True)
    print(cossim(s1, s2))

    print(m.n_similarity_sum(['kipu', 'sekavuus'], ['kipu'], normalize=True))
    #print(model.n_similarity(['kipu', 'sekavuus'], ['kipu']))
    '''




