# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import importlib
import os


LMOUTSCHEDULE_REGISTRY = {}
LMOUTSCHEDULE_CLASS_NAMES = set()


def build_lmoutschedule(args, encoder):
    return LMOUTSCHEDULE_REGISTRY[args.lmoutschedule](args, encoder)


def register_lmoutschedule(name):
    """Decorator to register a new criterion."""

    def register_lmoutschedule_cls(cls):
        if name in LMOUTSCHEDULE_REGISTRY:
            raise ValueError('Cannot register duplicate criterion ({})'.format(name))

        LMOUTSCHEDULE_REGISTRY[name] = cls
        LMOUTSCHEDULE_CLASS_NAMES.add(cls.__name__)
        return cls

    return register_lmoutschedule_cls


# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('fairseq.lmoutschedule.' + module)
