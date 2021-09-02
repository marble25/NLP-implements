import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

corpus = [
  'he is a king',
  'she is a queen',
  'he is a man',
  'she is a woman',
  'warsaw is poland capital',
  'berlin is germany capital',
  'paris is france capital',
]

class PreProcessing:
  def __init__(self):
    self.window_size = 2
    self.idx_pairs = []

  def _tokenize_corpus(self, corpus):
    tokens = [word.split() for word in corpus]
    return tokens

  def make_vocabulary_list(self, corpus):
    self.tokenized_corpus = self._tokenize_corpus(corpus)
    self.vocabulary = list(set(itertools.chain.from_iterable(self.tokenized_corpus))) # make unique set of tokens
    self.vocabulary_size = len(self.vocabulary)

    self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocabulary)}

  def make_center_context_word_pairs(self):
    # for each sentence
    for sentence in self.tokenized_corpus:
      indices = [self.word2idx[word] for word in sentence]
      for center_word_pos in range(len(indices)):
        for w in range(-self.window_size, self.window_size + 1): # relative index
          context_word_pos = center_word_pos + w
          # make soure not jump out sentence
          if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
            continue
          context_word_idx = indices[context_word_pos]
          self.idx_pairs.append((indices[center_word_pos], context_word_idx)) # (center_word_idx, context_word_idx)

    self.idx_pairs = np.array(self.idx_pairs)

class Model:
  def __init__(self):
    self.preprocessing = PreProcessing()
    self.embedding_dims = 5
    self.num_epochs = 100
    self.learning_rate = 0.001

  def prepare_corpus(self, corpus):
    self.preprocessing.make_vocabulary_list(corpus)
    self.preprocessing.make_center_context_word_pairs()
    self.vocabulary_size = self.preprocessing.vocabulary_size
    self.W1 = Variable(torch.randn(self.embedding_dims, self.vocabulary_size).float(), requires_grad=True) # initialize with random number (size: embedding_dims * vocabulary_size)
    self.W2 = Variable(torch.randn(self.vocabulary_size, self.embedding_dims).float(), requires_grad=True) # initialize with random number (size: vocabulary_size * embedding_dims)

  def get_input_layer(self, word_idx):
    x = torch.zeros(self.vocabulary_size).float()
    x[word_idx] = 1.0
    return x

  def train(self):
    idx_pairs = self.preprocessing.idx_pairs
    for epo in range(self.num_epochs):
      loss_val = 0
      for data, target in idx_pairs:
        x = Variable(self.get_input_layer(data)).float() # one-hot vector
        y_true = Variable(torch.from_numpy(np.array([target])).long())

        z1 = torch.matmul(self.W1, x)
        z2 = torch.matmul(self.W2, z1)

        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1,-1), y_true)
        loss_val += loss.item()
        loss.backward() # calculate gradient

        self.W1.data -= self.learning_rate * self.W1.grad.data # update w1 using gradient
        self.W2.data -= self.learning_rate * self.W2.grad.data

        self.W1.grad.data.zero_() # set gradient 0
        self.W2.grad.data.zero_()
      if (epo+1) % 10 == 0:    
        print(f'Loss at epo {epo+1}: {loss_val/len(idx_pairs)}')

model = Model()
model.prepare_corpus(corpus)
model.train()