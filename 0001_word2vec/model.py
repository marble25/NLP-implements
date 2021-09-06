import itertools
import collections
import datetime
import math
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor
import re

class PreProcessing:
  def __init__(self):
    self.window_size = 2
    self.idx_pairs = []
    self.counter = collections.Counter()
    self.portion = {}
    self.vocabulary = []
    self.threshold = 0.0000001

  def get_corpus_from_dataset(self, dataset):
    corpus = []
    for row in dataset:
      text = re.sub(r'[^a-zA-Z ]', "", row)
      if text: corpus.append(text.lower())

    print(f'Length of Corpus: {len(corpus)}')
    return corpus


  def _tokenize_corpus(self, corpus):
    tokens = [word.split() for word in corpus]
    return tokens

  def _subsampling(self):
    total_tokens = sum(self.counter.values())
    self.total_vocabs = 0

    for word in self.counter:
      p = 1 - math.sqrt(self.threshold / (self.counter[word] / total_tokens))
      rand_num = random.random()

      if p < rand_num: continue # deleted by probability p
      self.vocabulary.append(word)
      self.total_vocabs += self.counter[word]
    
    self.vocabulary_size = len(self.vocabulary)
    print(f'Length of Vocabulary: {self.vocabulary_size}')

  def make_vocabulary_list(self, corpus):
    self.tokenized_corpus = self._tokenize_corpus(corpus)
    for idx, token in enumerate(self.tokenized_corpus):
      self.counter += collections.Counter(token)
    
    self._subsampling()

    self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocabulary)}

    for word in self.vocabulary:
      self.portion[self.word2idx[word]] = self.counter[word] / self.total_vocabs

    assert 'king' in self.word2idx
    assert 'queen' in self.word2idx
    assert 'woman' in self.word2idx
    assert 'man' in self.word2idx

  def make_center_context_word_pairs(self):
    # for each sentence
    for sentence in self.tokenized_corpus:
      indices = [self.word2idx[word] for word in sentence if word in self.vocabulary] # word must be in vocabulary
      for center_word_pos in range(len(indices)):
        for w in range(-self.window_size, self.window_size + 1): # relative index
          context_word_pos = center_word_pos + w
          # make soure not jump out sentence
          if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
            continue
          context_word_idx = indices[context_word_pos]
          self.idx_pairs.append((indices[center_word_pos], context_word_idx)) # (center_word_idx, context_word_idx)

    self.idx_pairs = np.array(self.idx_pairs)
    print(f'Length of idx_pairs : {len(self.idx_pairs)}')

class Model:
  def __init__(self):
    self.preprocessing = PreProcessing()
    self.embedding_dims = 300
    self.num_epochs = 1
    self.learning_rate = 0.001
    self.negative_sampling_count = 20

    self.GPU_NUM = 0

    device = torch.device(f'cuda:{self.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

  def prepare_corpus(self, dataset):
    corpus = self.preprocessing.get_corpus_from_dataset(dataset)
    self.preprocessing.make_vocabulary_list(corpus)
    self.preprocessing.make_center_context_word_pairs()
    self.vocabulary_size = self.preprocessing.vocabulary_size
    self.W1 = Tensor(torch.randn(self.embedding_dims, self.vocabulary_size).float()).cuda() # initialize with random number (size: embedding_dims * vocabulary_size)
    self.W2 = Tensor(torch.randn(self.vocabulary_size, self.embedding_dims).float()).cuda() # initialize with random number (size: vocabulary_size * embedding_dims)
    self.W1.requires_grad_()
    self.W2.requires_grad_()

  def get_input_layer(self, word_idx):
    x = torch.zeros(self.vocabulary_size).float()
    x[word_idx] = 1.0
    return x

  def train(self):
    idx_pairs = self.preprocessing.idx_pairs
    portion_key = list(self.preprocessing.portion.keys())
    portion_values = list(self.preprocessing.portion.values())
    for epo in range(self.num_epochs):
      loss_val = 0
      for idx, (data, target) in enumerate(idx_pairs):
        x = Tensor(self.get_input_layer(data)).float().cuda() # one-hot vector
        y_true = LongTensor(torch.from_numpy(np.array([target])).long()).cuda()

        z1 = torch.matmul(self.W1, x)
        z2 = torch.matmul(self.W2, z1)

        log_softmax = F.log_softmax(z2, dim=0)
        loss = F.nll_loss(log_softmax.view(1, -1), y_true)

        # # negative sampling
        # updating_indices = np.random.choice(portion_key, self.negative_sampling_count, replace=False, p=portion_values)
        # updating_indices = np.append(updating_indices, target)
        # updating = torch.zeros(self.vocabulary_size).float().cuda()
        # for i in updating_indices:
        #   updating[i] = 1.0
        
        # sampled_log_softmax = log_softmax * updating
        # loss = F.nll_loss(sampled_log_softmax.view(1,-1), y_true, reduction='sum') / (self.negative_sampling_count + 1)
        loss_val += loss.item()
        loss.backward() # calculate gradient

        self.W1.data -= self.learning_rate * self.W1.grad.data # update w1 using gradient
        self.W2.data -= self.learning_rate * self.W2.grad.data

        self.W1.grad.data.zero_() # set gradient 0
        self.W2.grad.data.zero_()
        if (idx+1) % 10000 == 0:
          print(f'[{datetime.datetime.now().time()}] Epoch {epo+1} - Row {idx+1} - Loss {loss_val/(idx+1)}')
      print(f'[{datetime.datetime.now().time()}] Loss at Epoch {epo+1}: {loss_val/len(idx_pairs)}')

start = datetime.datetime.now()
print(f'Start at: {start}')

# corpus_file = open('/content/gdrive/MyDrive/colab/training-monolingual/news.2011.en.shuffled', 'r')
corpus_file = open('training-monolingual/news.2011.en.shuffled', 'r')
corpus = corpus_file.readlines()

model = Model()
model.prepare_corpus(corpus[:100000])
model.train()

def similarity(model, v, u):
  return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))

word1 = 'king'
word2 = 'man'
word3 = 'woman'

with torch.no_grad():
  x1 = Tensor(model.get_input_layer(model.preprocessing.word2idx[word1])).float().cuda()
  x2 = Tensor(model.get_input_layer(model.preprocessing.word2idx[word2])).float().cuda()
  x3 = Tensor(model.get_input_layer(model.preprocessing.word2idx[word3])).float().cuda()

  wv1 = torch.matmul(model.W1, x1)
  wv2 = torch.matmul(model.W1, x2)
  wv3 = torch.matmul(model.W1, x3)
  wv4 = wv1 - wv2 + wv3

  max_similarity = -999999
  max_word = ''
  for word in model.preprocessing.vocabulary:
    x = Tensor(model.get_input_layer(model.preprocessing.word2idx[word])).float().cuda()
    wv = torch.matmul(model.W1, x)
    similarity = F.cosine_similarity(wv4, wv, dim=0).cpu()
    if similarity > max_similarity: max_word = word

    del x, wv, similarity
    torch.cuda.empty_cache()

  print(max_word)
  print(f'Elapsed Time: {datetime.datetime.now() - start}')
