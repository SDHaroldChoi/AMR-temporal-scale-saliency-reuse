"""
Copyright $today.year LY Corporation

LY Corporation licenses this file to you under the Apache License,
version 2.0 (the "License"); you may not use this file except in compliance
with the License. You may obtain a copy of the License at:

  https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations
under the License.
"""

import os

import yaml
from easydict import EasyDict

from basic_utils import mkdirp


class BaseOptions(object):
    """Loads a YAML config into an EasyDict and derives the output file paths."""

    def __init__(self, config_path):
        self.config_path = config_path
        self.opt = {}

    @property
    def option(self):
        if len(self.opt) == 0:
            raise RuntimeError('option is empty. Did you run parse()?')
        return self.opt

    def parse(self):
        with open(self.config_path, 'r') as f:
            self.opt.update(yaml.load(f, Loader=yaml.FullLoader))

        self.opt = EasyDict(self.opt)
        mkdirp(self.opt.results_dir)
        self.opt.ckpt_filepath = os.path.join(self.opt.results_dir, self.opt.ckpt_filename)
        self.opt.train_log_filepath = os.path.join(self.opt.results_dir, self.opt.train_log_filename)
        self.opt.eval_log_filepath = os.path.join(self.opt.results_dir, self.opt.eval_log_filename)
