<img src="./electra.png"></img>

## Electra - Pytorch

A simple wrapper for fast pretraining of language models as detailed in <a href="https://arxiv.org/abs/2003.10555">this paper</a>. The paper claims to be able to reach performances comparable to Roberta in a quarter of the compute.

## Install

```bash
$ pip install electra-pytorch
```

## Usage

The following example uses `reformer-pytorch`, which is available to be pip installed.

```python
import torch
from torch import nn
from reformer_pytorch import ReformerLM

from electra_pytorch.electra_pytorch import Electra

# instantiate the generator and discriminator

generator = ReformerLM(
    num_tokens = 20000,
    dim = 512,
    depth = 1,
    max_seq_len = 1024
)

discriminator = ReformerLM(
    num_tokens = 20000,
    dim = 512,
    depth = 2,
    max_seq_len = 1024,
    return_embeddings = True
)

# weight tie the token and positional embeddings of generator and discriminator

generator.token_emb = discriminator.token_emb
generator.pos_emb = discriminator.pos_emb

# make sure discriminator outputs to a sigmoid (predicting replaced or original)

discriminator_with_sigmoid = nn.Sequential(
    discriminator,
    nn.Linear(512, 1),
    nn.Sigmoid()
)

# instantiate electra

trainer = Electra(
    generator,
    discriminator_with_sigmoid,
    mask_token_id = 2,
    mask_prob = 0.15
)

# train

x = torch.randint(0, 20000, (1, 1024))
loss = trainer(x)
loss.backward()

# after much training, the discriminator should have improved

torch.save(discriminator, f'./pretrained-model.pt')
```

## Citations

```bibtex
@misc{clark2020electra,
    title={ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators},
    author={Kevin Clark and Minh-Thang Luong and Quoc V. Le and Christopher D. Manning},
    year={2020},
    eprint={2003.10555},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
