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

class LSTM(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(LSTM, self).__init__()
    self.hidden_size = hidden_size

    # forget gate
    self.W_xh_f = torch.nn.Linear(input_size, hidden_size, bias=True).cuda()
    self.W_hh_f = torch.nn.Linear(hidden_size, hidden_size, bias=False).cuda()
    
    # input gate (to remember current input vector)
    self.W_xh_i = torch.nn.Linear(input_size, hidden_size, bias=True).cuda()
    self.W_hh_i = torch.nn.Linear(hidden_size, hidden_size, bias=False).cuda()

    # output gate
    self.W_xh_o = torch.nn.Linear(input_size, hidden_size, bias=True).cuda()
    self.W_hh_o = torch.nn.Linear(hidden_size, hidden_size, bias=False).cuda()

    self.W_xh_g = torch.nn.Linear(input_size, hidden_size, bias=True).cuda()
    self.W_hh_g = torch.nn.Linear(hidden_size, hidden_size, bias=False).cuda()

    self.W_out = torch.nn.Linear(hidden_size, output_size, bias=True).cuda()

    self.sigmoid = torch.nn.Sigmoid().cuda()
    self.tanh = torch.nn.Tanh().cuda()

  def forward(self, input_vec, hidden_vec, cell_vec):
    f = self.W_xh_f(input_vec) + self.W_hh_f(hidden_vec)
    tan_f = self.sigmoid(f)

    i = self.W_xh_i(input_vec) + self.W_hh_i(hidden_vec)
    tan_i = self.sigmoid(i)

    o = self.W_xh_o(input_vec) + self.W_hh_o(hidden_vec)
    tan_o = self.sigmoid(o)

    g = self.W_xh_g(input_vec) + self.W_hh_g(hidden_vec)
    tan_g = self.tanh(g)

    new_cell_vec = tan_f * cell_vec + tan_i * tan_g
    new_hidden_vec = tan_o * self.tanh(new_cell_vec)

    output_vec = self.W_out(new_hidden_vec)

    return new_hidden_vec, new_cell_vec, output_vec

class Trainer:
  def __init__(self, corpus):
    self.learning_rate = 0.03
    self.epochs = 100
    self.criterion = torch.nn.CrossEntropyLoss().cuda()

    self.preprocessing = Preprocessing()
    self.preprocessing.preprocess(corpus)

    self.lstm = LSTM(self.preprocessing.vocabulary_size, 200, self.preprocessing.vocabulary_size)
    self.optimizer = torch.optim.SGD(self.lstm.parameters(), lr=self.learning_rate)

  def train(self):
    total_loss = 0
    total_run = 0
    for epoch in range(self.epochs):
      for idx, sentence in enumerate(self.preprocessing.tokenized_corpus):
        hidden_vec = torch.zeros(self.lstm.hidden_size).cuda()
        cell_vec = torch.zeros(self.lstm.hidden_size).cuda()

        for idx, token in enumerate(sentence[:-1]):
          input_idx = self.preprocessing.word2idx[token]
          input_vec = torch.zeros(self.preprocessing.vocabulary_size).cuda()
          input_vec[input_idx] = 1

          target_idx = self.preprocessing.word2idx[sentence[idx+1]]
          target_vec = torch.from_numpy(np.array([target_idx])).cuda()

          self.optimizer.zero_grad()
          hidden_vec, cell_vec, output_vec = self.lstm.forward(input_vec, hidden_vec, cell_vec)

          loss = self.criterion(output_vec.view(1, -1), target_vec)
          total_loss += loss.item()
          total_run += 1

          loss.backward()
          hidden_vec.detach_()
          hidden_vec = hidden_vec.detach()
          cell_vec.detach_()
          cell_vec = cell_vec.detach()

          self.optimizer.step()

          output_word = self.preprocessing.idx2word[np.argmax(output_vec.cpu().detach().numpy())]
          # print(f'{" ".join(sentence[:idx+1])} -> {output_word}')
        if (idx+1) % 1000 == 0:
          print(f'[{datetime.datetime.now()}] Epoch {epoch+1} - {idx}, Loss: {total_loss/total_run}')
      print(f'[{datetime.datetime.now()}] Epoch {epoch+1} finished')
      print(f'Loss: {total_loss/total_run}')

  def eval(self, corpus):
    for sentence in self.preprocessing.tokenized_corpus:
      hidden_vec = torch.zeros(self.lstm.hidden_size).cuda()
      cell_vec = torch.zeros(self.lstm.hidden_size).cuda()

      for idx, token in enumerate(sentence[:-1]):
        input_idx = self.preprocessing.word2idx[token]
        input_vec = torch.zeros(self.preprocessing.vocabulary_size).cuda()
        input_vec[input_idx] = 1

        with torch.no_grad():
          hidden_vec, cell_vec, output_vec = self.lstm.forward(input_vec, hidden_vec, cell_vec)

          output_word = self.preprocessing.idx2word[np.argmax(output_vec.cpu().detach().numpy())]
          print(f'{" ".join(sentence[:idx+1])} -> {output_word}')
      

corpus_file = open('../training-monolingual/news.2011.en.shuffled', 'r')
corpus = corpus_file.readlines()

trainer = Trainer(corpus[:1000])
trainer.train()
trainer.eval(corpus[:5])