import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm


CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
EMBEDDING_DIM = 64
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()

vocab = set(raw_text)
vocab_size = len(vocab)

word_to_ix = {word: i for i, word in enumerate(vocab)}
data = []
for i in range(2, len(raw_text) - 2):
    context = [raw_text[i - 2], raw_text[i - 1],
               raw_text[i + 1], raw_text[i + 2]]
    target = raw_text[i]
    data.append((context, target))

class CBOW(nn.Module):
    def __init__(self, vocab_size, ebd_size, cont_size):
        super(CBOW, self).__init__()

        self.ebd = nn.Embedding(vocab_size, ebd_size)
        self.lr1 = nn.Linear(ebd_size*cont_size*2, 128)
        self.lr2 = nn.Linear(128, vocab_size)

        self._init_weight()

    def forward(self, inputs):
        out = self.ebd(inputs).view(1, -1)
        out = F.relu(self.lr1(out))
        out = self.lr2(out)
        out = F.log_softmax(out, dim=-1)
        return out

    def _init_weight(self, scope=0.1):
        self.ebd.weight.data.uniform_(-scope, scope)
        self.lr1.weight.data.uniform_(0, scope)
        self.lr1.bias.data.fill_(0)
        self.lr2.weight.data.uniform_(0, scope)
        self.lr2.bias.data.fill_(0)

def make_context_vector(context, word_to_ix):
    idxs = [word_to_ix[w] for w in context]
    tensor = torch.LongTensor(idxs)
    return tensor

loss_function = nn.NLLLoss()
model = CBOW(vocab_size, EMBEDDING_DIM, CONTEXT_SIZE)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(1, 41):
    total_loss = 0.0
    with tqdm(data, total=len(data), leave=False) as pbar:
        for context, target in pbar:
            v_ctx = make_context_vector(context, word_to_ix)
            v_tar = torch.LongTensor([word_to_ix[target]])
            model.zero_grad()
            out = model(v_ctx)
            loss = loss_function(out, v_tar)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            pbar.set_description(
                f'end of epoch {epoch} | loss {total_loss:2.3f}'
            )

