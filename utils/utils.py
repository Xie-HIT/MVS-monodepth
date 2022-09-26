import sys
import os
import yaml


class Option:
    def __init__(self):
        self.model = {}
        self.loss = {}
        self.training = {}
        self.evaluation = {}

        self.num_dataset = {
            "Training": 0,
            "Evaluation": 0,
        }

    def read(self, path):
        with open(path, 'r') as f:
            configs = yaml.safe_load_all(f.read())

            for config in configs:
                for k, v in config.items():
                    if k == 'Model':
                        self.model = v
                    elif k == 'Loss':
                        self.loss = v
                    elif k == 'Training':
                        self.training = v
                        self.num_dataset['Training'] = len(v)
                    elif k == 'Evaluation':
                        self.evaluation = v
                        self.num_dataset['Evaluation'] = len(v)
                    else:
                        raise NotImplementedError('Invalid config: {}'.format(k))
