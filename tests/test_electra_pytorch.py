import torch
from torch import nn
from reformer_pytorch import ReformerLM

from electra_pytorch import Electra

def test_electra():
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

    generator.token_emb = discriminator.token_emb
    generator.pos_emb = discriminator.pos_emb

    trainer = Electra(
        generator,
        discriminator,
        discr_dim = 512,
        discr_layer = 'reformer'
    )

    data = torch.randint(0, 20000, (1, 1024))
    loss = trainer(data)
    loss.backward()

def test_electra_without_magic():
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

    generator.token_emb = discriminator.token_emb
    generator.pos_emb = discriminator.pos_emb


    discriminator_with_adapter = nn.Sequential(
        discriminator,
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    trainer = Electra(
        generator,
        discriminator_with_adapter
    )

    data = torch.randint(0, 20000, (1, 1024))
    loss = trainer(data)
    loss.backward()
