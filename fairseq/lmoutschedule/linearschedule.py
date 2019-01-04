from torch import nn
import numpy as np
from fairseq import utils
from . import register_lmoutschedule

@register_lmoutschedule('linear')
class LinearLmOutSchedule(object):

    def __init__(self, args, encoder, decoder):
        warmup_init_tradeoff = args.tradeoff
        warmup_end_tradeoff = args.tradeoff
        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_tradeoff - warmup_init_tradeoff) / args.tradeoff_step
        self.warmup_init_tradeoff = warmup_init_tradeoff
        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_tradeoff * args.tradeoff_step
        self.tradeoff_step = args.tradeoff_step

        # initial learning rate
        self.tradeoff = warmup_init_tradeoff
        self.encoder = encoder
        self.decoder = decoder
        self.set_tradeoff()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--tradeoff', type=float, default=1.)
        parser.add_argument('--tradeoff-step', type=int, default=4000)

    def set_tradeoff(self):
        self.encoder.tradeoff = self.tradeoff
        self.decoder.tradeoff = self.tradeoff

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates <= self.tradeoff_step:
            self.tradeoff = self.warmup_init_tradeoff + self.lr_step * num_updates
        else:
            self.tradeoff = self.decay_factor / num_updates
        self.tradeoff = float(np.clip(self.tradeoff, 0.1, 1.))
        self.set_tradeoff()
        return self.tradeoff