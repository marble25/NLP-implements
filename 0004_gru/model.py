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

class GRUEncoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size):
    super(GRUEncoder, self).__init__()
    self.hidden_size = hidden_size

    self.W_z = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.U_z = torch.nn.Linear(hidden_size, hidden_size, bias=True)

    self.W_r = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.U_r = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    
    self.W = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.U = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    
    self.logistic_sigmoid = torch.nn.LogSigmoid()
    self.tanh = torch.nn.Tanh()


  def forward(self, input_vec, hidden_vec):
    z = self.logistic_sigmoid(self.W_z(input_vec) + self.U_z(hidden_vec))
    r = self.logistic_sigmoid(self.W_r(input_vec) + self.U_r(hidden_vec))
    hidden_vec_after_reset = r * hidden_vec
    new_hidden_vec = self.tanh(self.W(input_vec) + self.U(hidden_vec_after_reset))

    final_hidden_vec = z * hidden_vec + (1-z) * new_hidden_vec

    return final_hidden_vec

class Encoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size):
    super(Encoder, self).__init__()
    self.gru_encoder = GRUEncoder(input_size, hidden_size)

    self.V = torch.nn.Linear(hidden_size, output_size, bias=True)
    self.tanh = torch.nn.Tanh()

  def forward(self, input_vecs, hidden_vec):
    for input_vec in input_vecs:
      hidden_vec = self.gru_encoder.forward(input_vec, hidden_vec)
    
    context = self.tanh(self.V(hidden_vec))
    return context

class GRUDecoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size, context_size, vocab_size):
    super(GRUDecoder, self).__init__()
    self.hidden_size = hidden_size
    self.g_size = 500

    self.W_z = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.U_z = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    self.C_z = torch.nn.Linear(context_size, hidden_size, bias=True)

    self.W_r = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.U_r = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    self.C_r = torch.nn.Linear(context_size, hidden_size, bias=True)
    
    self.W = torch.nn.Linear(input_size, hidden_size, bias=True)
    self.U = torch.nn.Linear(hidden_size, hidden_size, bias=True)
    self.C = torch.nn.Linear(context_size, hidden_size, bias=True)

    self.O_h = torch.nn.Linear(hidden_size, output_size*2, bias=True)
    self.O_y = torch.nn.Linear(input_size, output_size*2, bias=True)
    self.O_c = torch.nn.Linear(context_size, output_size*2, bias=True)

    self.G_l = torch.nn.Linear(vocab_size, self.g_size, bias=False)
    self.G_r = torch.nn.Linear(output_size, vocab_size, bias=False)
    self.G = self.G_l(self.G_r)
    
    self.logistic_sigmoid = torch.nn.LogSigmoid()
    self.tanh = torch.nn.Tanh()
    self.softmax = torch.nn.Softmax()


  def forward(self, input_vec, hidden_vec, context_vec):
    z = self.logistic_sigmoid(self.W_z(input_vec) + self.U_z(hidden_vec) + self.C_z(context_vec))
    r = self.logistic_sigmoid(self.W_r(input_vec) + self.U_r(hidden_vec)+ self.C_r(context_vec))
    hidden_vec_after_reset = r * (self.U(hidden_vec) + self.C(context_vec))
    new_hidden_vec = self.tanh(self.W(input_vec) + self.U(hidden_vec_after_reset))

    final_hidden_vec = z * hidden_vec + (1-z) * new_hidden_vec

    s = self.O_h(hidden_vec) + self.O_y(input_vec) + self.O_c(context_vec)
    final_s = max(s[::2], s[1::2])

    output_vec = self.softmax(self.G(final_s))

    return final_hidden_vec, output_vec

class Decoder(torch.nn.Module):
  def __init__(self, input_size, hidden_size, output_size, context_size, vocab_size):
    super(Decoder, self).__init__()
    self.gru_decoder = GRUDecoder(input_size, hidden_size, output_size, context_size, vocab_size)

    self.V = torch.nn.Linear(hidden_size, output_size, bias=True)
    self.tanh = torch.nn.Tanh()

  def forward(self, input_vecs, context_vec):
    hidden_vec = self.tanh(self.V(context_vec))

    output_vecs = []
    for input_vec in input_vecs:
      hidden_vec, output_vec = self.gru_decoder.forward(input_vec, hidden_vec, context_vec)
      output_vecs.append(output_vec)

    return output_vec
