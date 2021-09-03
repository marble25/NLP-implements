import itertools
import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, LongTensor
import re
from datasets import load_dataset

class PreProcessing:
  def __init__(self):
    self.window_size = 2
    self.idx_pairs = []

  def get_corpus_from_dataset(self, dataset):
    corpus = []
    for row in dataset:
      text = re.sub(r'[^a-zA-Z ]', "", row)
      if text: corpus.append(text)

    print(f'Length of Corpus: {len(corpus)}')
    return corpus


  def _tokenize_corpus(self, corpus):
    tokens = [word.split() for word in corpus]
    return tokens

  def make_vocabulary_list(self, corpus):
    self.tokenized_corpus = self._tokenize_corpus(corpus)
    self.vocabulary = list(set(itertools.chain.from_iterable(self.tokenized_corpus))) # make unique set of tokens
    self.vocabulary_size = len(self.vocabulary)

    print(f'Length of Vocabulary: {self.vocabulary_size}')

    self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocabulary)}

    assert 'king' in self.word2idx
    assert 'queen' in self.word2idx
    assert 'woman' in self.word2idx
    assert 'man' in self.word2idx

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
    print(f'Length of idx_pairs : {len(self.idx_pairs)}')

class Model:
  def __init__(self):
    self.preprocessing = PreProcessing()
    self.embedding_dims = 100
    self.num_epochs = 50
    self.learning_rate = 0.1

    self.GPU_NUM = 0

    device = torch.device(f'cuda:{self.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device)

  def prepare_corpus(self, dataset):
    corpus = self.preprocessing.get_corpus_from_dataset(dataset)
    self.preprocessing.make_vocabulary_list(corpus)
    self.preprocessing.make_center_context_word_pairs()
    self.vocabulary_size = self.preprocessing.vocabulary_size
    self.W1 = Tensor(torch.randn(self.embedding_dims, self.vocabulary_size).float()) # initialize with random number (size: embedding_dims * vocabulary_size)
    self.W2 = Tensor(torch.randn(self.vocabulary_size, self.embedding_dims).float()) # initialize with random number (size: vocabulary_size * embedding_dims)
    self.W1.requires_grad_()
    self.W2.requires_grad_()

  def get_input_layer(self, word_idx):
    x = torch.zeros(self.vocabulary_size).float()
    x[word_idx] = 1.0
    return x

  def train(self):
    idx_pairs = self.preprocessing.idx_pairs
    for epo in range(self.num_epochs):
      loss_val = 0
      for idx, (data, target) in enumerate(idx_pairs):
        x = Tensor(self.get_input_layer(data)).float() # one-hot vector
        y_true = LongTensor(torch.from_numpy(np.array([target])).long())

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
        if (idx+1) % 10000 == 0:
          print(f'[{datetime.datetime.now().time()}] Epoch {epo+1} - Row {idx+1}')
      print(f'[{datetime.datetime.now().time()}] Loss at Epoch {epo+1}: {loss_val/len(idx_pairs)}')
      if self.learning_rate >= 0.001: self.learning_rate = self.learning_rate * 0.96

start = datetime.datetime.now()

dataset = load_dataset('wikitext', 'wikitext-2-raw-v1')['train']
dataset = dataset.filter(lambda sample: 'king' in sample['text'] or 
                                        'queen' in sample['text'] or 
                                        'man' in sample['text'] or 
                                        'woman' in sample['text'])[:150]

model = Model()
model.prepare_corpus(dataset['text'])
model.train()

def similarity(model, v, u):
  return torch.dot(v,u)/(torch.norm(v)*torch.norm(u))

word1 = 'king'
word2 = 'man'
word3 = 'woman'
wv1 = torch.matmul(model.W1,model.get_input_layer(model.preprocessing.word2idx[word1]))
wv2 = torch.matmul(model.W1,model.get_input_layer(model.preprocessing.word2idx[word2]))
wv3 = torch.matmul(model.W1,model.get_input_layer(model.preprocessing.word2idx[word3]))
wv4 = wv1 - wv2 + wv3

similarities = {}
for word in model.preprocessing.vocabulary:
  wv = torch.matmul(model.W1,model.get_input_layer(model.preprocessing.word2idx[word]))
  similarities[word] = similarity(model, wv4, wv)

max_word = max(similarities.keys(), key=(lambda k: similarities[k]))
print(max_word)
print(f'Elapsed Time: {datetime.datetime.now() - start}')