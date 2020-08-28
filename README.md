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

from electra_pytorch import Electra

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
    max_seq_len = 1024
)

# weight tie the token and positional embeddings of generator and discriminator

generator.token_emb = discriminator.token_emb
generator.pos_emb = discriminator.pos_emb

# instantiate electra

trainer = Electra(
    generator,
    discriminator,
    num_tokens = 20000,         # number of tokens, needed for masked language pretraining
    discr_dim = 512,            # the embedding dimension of the discriminator
    discr_layer = 'reformer',   # the layer name in the discriminator, whose output would be used for predicting token is still the same or replaced
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15            # masking probability for masked language modeling
)

# train

data = torch.randint(0, 20000, (1, 1024))

results = trainer(data)
results.loss.backward()

# after much training, the discriminator should have improved

torch.save(discriminator, f'./pretrained-model.pt')
```

If you would rather not have the framework auto-magically intercept the hidden output of the discriminator, you can pass in the discriminator (with the extra linear [dim x 1]) by yourself with the following.

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

# instantiate electra

discriminator_with_adapter = nn.Sequential(discriminator, nn.Linear(512, 1))

trainer = Electra(
    generator,
    discriminator_with_adapter,
    num_tokens = 20000,         # number of tokens, needed for masked language pretraining
    mask_token_id = 2,          # the token id reserved for masking
    pad_token_id = 0,           # the token id for padding
    mask_prob = 0.15            # masking probability for masked language modeling
)

# train

data = torch.randint(0, 20000, (1, 1024))

results = trainer(data)
results.loss.backward()

# after much training, the discriminator should have improved

torch.save(discriminator, f'./pretrained-model.pt')
```

## Testing

```bash
$ python setup.py test
```

## Pre-Training

1. Download the [OpenWebText](https://github.com/jcpeterson/openwebtext) dataset.

```bash
$ mkdir data
$ cd data
$ pip3 install gdown
$ gdown --id 1EA5V0oetDCOke7afsktL_JDQ-ETtNOvx
$ tar -xf openwebtext.tar.xz
$ wget https://storage.googleapis.com/electra-data/vocab.txt
$ cd ..
```

2. Tokenize dataset.

```bash
$ python3 pretraining/openwebtext/preprocess.py
```

3. Pre-train.

```bash
$ python3 pretraining/openwebtext/pretrain.py
```

4. Download GLUE dataset.

```bash
$ python3 examples/glue/download.py 
```

5. Fine-tune on the MRPC sub-task of the GLUE benchmark.

```bash
$ python3 examples/glue/run.py --model_name_or_path output/yyyy-mm-dd-hh-mm-ss/ckpt/200000
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
