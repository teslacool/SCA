from . import register_lmoutschedule

@register_lmoutschedule('constant')
class ConstantLmOutSchedule(object):

    def __init__(self, args, encoder):
        self.tradeoff = args.tradeoff
        self.encoder = encoder
        self.set_tradeoff()

    @staticmethod
    def add_args(parser):
        parser.add_argument('--tradeoff', type=float, default=1.)
        parser.add_argument('--tradeoff-step', type=int, default=4000)

    def set_tradeoff(self):
        self.encoder.tradeoff = self.tradeoff

    def step_update(self, num_updates):
        self.set_tradeoff()
        return self.tradeoff