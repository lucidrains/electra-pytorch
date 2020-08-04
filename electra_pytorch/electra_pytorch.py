import torch
from torch import nn
import torch.nn.functional as F

def batched_index_select(values, indices):
    last_dim = values.shape[-1]
    return values.gather(1, indices[:, :, None].expand(-1, -1, last_dim))

class Electra(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        pad_token_id = 0,
        mask_token_id = 2,
        mask_prob = 0.15):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id

    def forward(self, input):
        b, t = input.shape

        mask_prob = torch.zeros_like(input).float().uniform_(0, 1)
        mask = (mask_prob < self.mask_prob) & (input != self.pad_token_id)

        masked_input = input.masked_fill(mask, self.mask_token_id)
        gen_labels = input.masked_fill(~mask, self.pad_token_id)

        print(masked_input.shape)
        logits = self.generator(masked_input)

        mlm_loss = F.cross_entropy(
            logits.reshape(b * t, -1),
            gen_labels.view(-1),
            ignore_index = self.pad_token_id
        )

        mask_indices = torch.nonzero(mask, as_tuple=True)
        sample_logits = logits[mask_indices].softmax(dim=-1)
        sampled = torch.multinomial(sample_logits, 1)

        disc_input = input.clone()
        disc_input[mask_indices] = sampled.squeeze(-1)

        disc_logits = self.discriminator(disc_input)
        disc_labels = (input == disc_input).float()

        disc_loss = F.binary_cross_entropy(
            disc_logits.squeeze(-1),
            disc_labels
        )

        return mlm_loss + disc_loss
