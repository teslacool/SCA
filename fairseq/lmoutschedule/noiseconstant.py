from . import register_lmoutschedule
import numpy as np
@register_lmoutschedule('noiseconstant')
class NoiseConstantLmOutSchedule(object):

    def __init__(self, args, encoder):
        warmup_init_tradeoff = args.tradeoff
        self.warmup_init_tradeoff = warmup_init_tradeoff
        self.decay_factor = warmup_init_tradeoff
        self.tradeoff = warmup_init_tradeoff
        self.encoder = encoder
        self.sigma = args.sigma
        self.tradeoff = float(np.clip(np.random.normal(self.tradeoff, self.sigma), 0., 1.))

        self.set_tradeoff()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--tradeoff', type=float, default=1.)
        parser.add_argument('--tradeoff-step', type=int, default=4000)
        parser.add_argument('--sigma', type=float, default=0.3)

    def set_tradeoff(self):
        self.encoder.tradeoff = self.tradeoff

    def step_update(self, num_updates):
        self.tradeoff = self.warmup_init_tradeoff
        self.tradeoff = float(np.clip(np.random.normal(self.tradeoff, self.sigma), 0., 1.))
        self.set_tradeoff()
        return self.tradeoff