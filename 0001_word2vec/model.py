import itertools

corpus = [
  'he is a king',
  'she is a queen',
  'he is a man',
  'she is a woman',
  'warsaw is poland capital',
  'berlin is germany capital',
  'paris is france capital',
]

def tokenize_corpus(corpus):
  tokens = [word.split() for word in corpus]
  return tokens

tokenized_corpus = tokenize_corpus(corpus)
vocabulary = list(set(itertools.chain.from_iterable(tokenized_corpus))) # make unique set of tokens
vocabulary_size = len(vocabulary)

word2idx = {word: idx for idx, word in enumerate(vocabulary)}
idx2word = {idx: word for idx, word in enumerate(vocabulary)}

import numpy as np

window_size = 2
idx_pairs = []
# for each sentence
for sentence in tokenized_corpus:
  indices = [word2idx[word] for word in sentence]
  for center_word_pos in range(len(indices)):
    for w in range(-window_size, window_size + 1): # relative index
      context_word_pos = center_word_pos + w
      # make soure not jump out sentence
      if context_word_pos < 0 or context_word_pos >= len(indices) or center_word_pos == context_word_pos:
        continue
      context_word_idx = indices[context_word_pos]
      idx_pairs.append((indices[center_word_pos], context_word_idx)) # (center_word_idx, context_word_idx)

idx_pairs = np.array(idx_pairs)

import torch
import torch.nn.functional as F
from torch.autograd import Variable

def get_input_layer(word_idx):
  x = torch.zeros(vocabulary_size).float()
  x[word_idx] = 1.0
  return x

embedding_dims = 5
W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True) # initialize with random number (size: embedding_dims * vocabulary_size)
W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True) # initialize with random number (size: vocabulary_size * embedding_dims)
num_epochs = 100
learning_rate = 0.001

for epo in range(num_epochs):
  loss_val = 0
  for data, target in idx_pairs:
    x = Variable(get_input_layer(data)).float() # one-hot vector
    y_true = Variable(torch.from_numpy(np.array([target])).long())

    z1 = torch.matmul(W1, x)
    z2 = torch.matmul(W2, z1)

    log_softmax = F.log_softmax(z2, dim=0)
    loss = F.nll_loss(log_softmax.view(1,-1), y_true)
    loss_val += loss.item()
    loss.backward() # calculate gradient

    W1.data -= learning_rate * W1.grad.data # update w1 using gradient
    W2.data -= learning_rate * W2.grad.data

    W1.grad.data.zero_() # set gradient 0
    W2.grad.data.zero_()
  if (epo+1) % 10 == 0:    
    print(f'Loss at epo {epo+1}: {loss_val/len(idx_pairs)}')