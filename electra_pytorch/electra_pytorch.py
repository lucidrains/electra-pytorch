from functools import reduce
import torch
from torch import nn
import torch.nn.functional as F

# hidden layer extractor class, for magically adding adapter to language model to be pretrained

class HiddenLayerExtractor(nn.Module):
    def __init__(self, net, layer = -2):
        super().__init__()
        self.net = net
        self.layer = layer

        self.hidden = None
        self.hook_registered = False

    def _find_layer(self):
        if type(self.layer) == str:
            modules = dict([*self.net.named_modules()])
            return modules.get(self.layer, None)
        elif type(self.layer) == int:
            children = [*self.net.children()]
            return children[self.layer]
        return None

    def _hook(self, _, __, output):
        self.hidden = output

    def _register_hook(self):
        layer = self._find_layer()
        assert layer is not None, f'hidden layer ({self.layer}) not found'
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def forward(self, x):
        if self.layer == -1:
            return self.net(x)

        if not self.hook_registered:
            self._register_hook()

        _ = self.net(x)
        hidden = self.hidden
        self.hidden = None
        assert hidden is not None, f'hidden layer {self.layer} never emitted an output'
        return hidden

# main electra class

class Electra(nn.Module):
    def __init__(
        self,
        generator,
        discriminator,
        discr_dim = -1,
        discr_layer = -1,
        mask_prob = 0.15,
        mask_token_id = 2,
        pad_token_id = 0,
        mask_ignore_token_ids = []):
        super().__init__()

        self.generator = generator
        self.discriminator = discriminator

        if discr_dim > 0:
            self.discriminator = nn.Sequential(
                HiddenLayerExtractor(discriminator, layer = discr_layer),
                nn.Linear(discr_dim, 1),
                nn.Sigmoid()
            )

        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])

    def forward(self, input):
        b, t = input.shape

        # generate mask for mlm pre-training of generator
        mask_prob = torch.zeros_like(input).float().uniform_(0, 1)
        mask = (mask_prob < self.mask_prob)

        # do not mask [pad] tokens, or any other tokens in the tokens designated to be excluded ([cls], [sep])
        init_no_mask = torch.full_like(input, False, dtype=torch.bool)
        no_mask = reduce(lambda acc, el: acc | (input == el), self.mask_ignore_token_ids, init_no_mask)
        mask &= ~no_mask

        # mask input with mask tokens, set inverse of mask to padding tokens for labels
        masked_input = input.masked_fill(mask, self.mask_token_id)
        gen_labels = input.masked_fill(~mask, self.pad_token_id)

        # get generator output and get mlm loss
        logits = self.generator(masked_input)

        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            gen_labels,
            ignore_index = self.pad_token_id
        )

        # use mask from before to select logits that need sampling
        mask_indices = torch.nonzero(mask, as_tuple=True)
        sample_logits = logits[mask_indices].softmax(dim=-1)

        # sample
        sampled = torch.multinomial(sample_logits, 1)

        # scatter the sampled values back to the input
        disc_input = input.clone()
        disc_input[mask_indices] = sampled.squeeze(-1)

        # generate discriminator labels, with replaced as True and original as False
        disc_labels = (input != disc_input).float()

        # get discriminator predictions of replaced / original
        disc_logits = self.discriminator(disc_input)

        disc_loss = F.binary_cross_entropy(
            disc_logits.squeeze(-1),
            disc_labels
        )

        # return losses summed
        return mlm_loss + disc_loss
