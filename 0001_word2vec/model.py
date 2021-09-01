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

word2idx = {word: idx for idx, word in enumerate(vocabulary)}
idx2word = {idx: word for idx, word in enumerate(vocabulary)}
