import numpy as np
import torch
import re
import collections
import torch
import datetime

class Preprocessing:
  def __init__(self):
    self.vocabulary = []
    self.counter = collections.Counter()
    self.vocabulary_size = 0
    self.total_vocab = 0
    self.word2idx = {}
    self.idx2word = {}

  def _cleaning(self, corpus):
    new_corpus = []
    for sentence in corpus:
      text = re.sub(r'[^a-zA-Z ]', "", sentence)
      if text: new_corpus.append(text.lower())
    
    print(f'Number of Lines: {len(new_corpus)}')
    return new_corpus

  def _tokenize(self, corpus):
    return [tokenized_sentence.split() for tokenized_sentence in corpus]

  def _make_vocabulary_list(self):
    for sentence in self.tokenized_corpus:
      self.counter += collections.Counter(sentence)

    self.vocabulary = list(self.counter.keys())
    self.vocabulary_size = len(self.vocabulary)
    self.total_vocab = sum(self.counter.values())

    print(f'Total vocabulary size: {self.vocabulary_size}')
    print(f'Total vocabs: {self.total_vocab}')

  def _calculate_word_idx(self):
    self.word2idx = {word: idx for idx, word in enumerate(self.vocabulary)}
    self.idx2word = {idx: word for idx, word in enumerate(self.vocabulary)}


  def preprocess(self, corpus):
    self.corpus = self._cleaning(corpus)
    self.tokenized_corpus = self._tokenize(self.corpus)
    self._make_vocabulary_list()
    self._calculate_word_idx()

class RNN(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(RNN, self).__init__()
    self.hidden_size = hidden_size

    self.W_ih = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.W_hh = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    self.W_ho = torch.nn.Linear(hidden_size, output_size, bias=True)

    self.tanh = torch.nn.Tanh()

  def forward(self, input_vec, hidden_vec):
    v_x = self.W_ih(input_vec)
    v_h = self.W_hh(hidden_vec)
    new_hidden_vec = self.tanh(v_x + v_h)
    
    output_vec = self.W_ho(new_hidden_vec)
    return hidden_vec, output_vec


class Trainer:
  def __init__(self, corpus):
    self.learning_rate = 0.03
    self.epochs = 100
    self.criterion = torch.nn.CrossEntropyLoss()

    self.preprocessing = Preprocessing()
    self.preprocessing.preprocess(corpus)

    self.rnn = RNN(self.preprocessing.vocabulary_size, 30, self.preprocessing.vocabulary_size)
    self.optimizer = torch.optim.SGD(self.rnn.parameters(), lr=self.learning_rate)

  def train(self):
    total_loss = 0
    total_run = 0
    for epoch in range(self.epochs):
      for sentence in self.preprocessing.tokenized_corpus:
        hidden_vec = torch.zeros(self.rnn.hidden_size)

        for idx, token in enumerate(sentence[:-1]):
          input_idx = self.preprocessing.word2idx[token]
          input_vec = torch.zeros(self.preprocessing.vocabulary_size)
          input_vec[input_idx] = 1

          target_idx = self.preprocessing.word2idx[sentence[idx+1]]
          target_vec = torch.from_numpy(np.array([target_idx]))

          self.optimizer.zero_grad()
          hidden_vec, output_vec = self.rnn.forward(input_vec, hidden_vec)

          loss = self.criterion(output_vec.view(1, -1), target_vec)
          total_loss += loss.item()
          total_run += 1

          loss.backward(retain_graph=True)
          
          self.optimizer.step()

          output_word = self.preprocessing.idx2word[np.argmax(output_vec.detach().numpy())]
          # print(f'{" ".join(sentence[:idx+1])} -> {output_word}')

      print(f'[{datetime.datetime.now()}] Epoch {epoch+1} finished')
      print(f'Loss: {total_loss/total_run}')

  def eval(self, corpus):
    for sentence in self.preprocessing.tokenized_corpus:
      hidden_vec = torch.zeros(self.rnn.hidden_size)

      for idx, token in enumerate(sentence[:-1]):
        input_idx = self.preprocessing.word2idx[token]
        input_vec = torch.zeros(self.preprocessing.vocabulary_size)
        input_vec[input_idx] = 1

        with torch.no_grad():
          hidden_vec, output_vec = self.rnn.forward(input_vec, hidden_vec)

          output_word = self.preprocessing.idx2word[np.argmax(output_vec.detach().numpy())]
          print(f'{" ".join(sentence[:idx+1])} -> {output_word}')
      

corpus_file = open('../training-monolingual/news.2011.en.shuffled', 'r')
corpus = corpus_file.readlines()

trainer = Trainer(corpus[:100])
trainer.train()
trainer.eval(corpus[0:5])