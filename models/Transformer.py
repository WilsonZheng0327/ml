import torch
import torch.nn as nn
from torch.nn import functional as F

'''
Following Andrej Karpathy's YouTube video:
"Let's build GPT: from scratch, in code, spelled out."

3Blue1Brown's video on attention:
"Attention in transformers, visually explained | Chapter 6, Deep Learning"
'''

batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2   # drop 20% intermediate neurons

vocab_size = 1 # changed based on what corpus the model is trained on

class Head(nn.Module):
    """
    Fundamental building block of a transformer: an attention head

    query   - represents the qualities a word might look for
    key     - represents the qualities a word holds
    value   - how it alters meanings of other words

    q, k, v are 3 trainable matrices; 

    (what's mentioned below refers to these 3 matrices multiplied by the 
    embeddings of the relevant word, so each word gets its own q, k, v matrices)

    q @ k   - to what extent two words are relevant to each other (attention pattern)
              so each word in the context will have an attention value 
              with all words before it (not considering future words)
    
    the meaning of words are changed based on the context, attention pattern
    decides to what extend other words change some word, and v decides how 
    the words change some word.
    """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))    # PyTorch convention for non-parameter variables
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),      # projection layer
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd  // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

print("Creating actual model...")

class GPTLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()

        # matrix representing:
        #       row - context chars
        #       col - embedding values (hidden layer)
        # how each char affects prediction
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # matrix representing:
        #       row - position in the block
        #       col - embedding values (hidden layer)
        # how position of each char affects prediction
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])

        self.ln_f = nn.LayerNorm(n_embd)

        self.lm_head = nn.Linear(n_embd, vocab_size)    # basically hiddle linear layer

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx dim.: (batch_size * block_size)
        # logits dim.: (batch_size * block_size * embedding_dim)
        #       there are intermediate steps (hidden layers)
        #       here final embedding_dim is vocab_size, repr. output prob. distribution
        #       video refers this embedding_dim as "channel", indicated with C
        # BASICALLY returns for each batch, for each char in block, what's the prediction prob. distribution looking like

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))    # pass locations
        x = tok_emb + pos_emb
        x = self.blocks(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)     # neg. log likelihood

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # pick the last block_size tokens to fit in position_embedding_table
            idx_cropped = idx[:, -block_size:]
            # get predictions
            logits, loss = self(idx_cropped)
            # only look at predictions based on last char since bigram
            logits = logits[:, -1, :]
            # convert to probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # add to running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            # print(idx)

        return idx