# coding=utf-8
# Copyright 2019 The Tensor2Tensor Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data generators for En-Vi translation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.data_generators import translate
from tensor2tensor.utils import registry

# End-of-sentence marker.
EOS = text_encoder.EOS_ID

# For English-Vietnamese the IWSLT'15 corpus
# from https://nlp.stanford.edu/projects/nmt/ is used.
# The original dataset has 133K parallel sentences.
_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.en", "train.vi")
]]

_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.vi", "train.en")
]]

# For development 1,553 parallel sentences are used.
_ENVI_TEST_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz",  # pylint: disable=line-too-long
    ("tst2012.en", "tst2012.vi")
]]


# See this PR on github for some results with Transformer on this Problem.
# https://github.com/tensorflow/tensor2tensor/pull/611


@registry.register_problem
class TranslateEnviIwslt32k(translate.TranslateProblem):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


@registry.register_problem
class TranslateVienIwslt32k(translate.TranslateProblem):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


# _PSEUDO_LABEL_MULTICC_ENVI_TRAIN_DATASETS = [
#     ['', ('train.en', 'train.vi')],  # original.
#     ['', ('MultiCCAligned.en.fixed.filter.filtertest.subset', 'MultiCCAligned.vi.fixed.filter.filtertest.subset')]
# ]

# @registry.register_problem
# class PseudoLabelMulticcTranslateEnviIwslt32k(translate.TranslateProblem):
#   """Problem spec for IWSLT'15 En-Vi translation."""

#   @property
#   def approx_vocab_size(self):
#     return 2**15  # 32768

#   def source_data_files(self, dataset_split):
#     train = dataset_split == problem.DatasetSplit.TRAIN
#     return _PSEUDO_LABEL_MULTICC_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

# _PSEUDO_LABEL_MULTICC_VIEN_TRAIN_DATASETS = [
#     ['', ('train.vi', 'train.en')],  # original.
#     ['', ('MultiCCAligned.vi.fixed.filter.filtertest.subset', 'MultiCCAligned.en.fixed.filter.filtertest.subset')]
# ]


# @registry.register_problem
# class PseudoLabelMulticcTranslateViEnIwslt32k(translate.TranslateProblem):
#   """Problem spec for IWSLT'15 En-Vi translation."""

#   @property
#   def approx_vocab_size(self):
#     return 2**15  # 32768

#   def source_data_files(self, dataset_split):
#     train = dataset_split == problem.DatasetSplit.TRAIN
#     return _PSEUDO_LABEL_MULTICC_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

OPENSUBTITLES_VIEN = [
     ['', ('train.vi', 'train.en')],  # original.
     ['', ('OpenSubtitles.vi.subset', 'OpenSubtitles.en.subset')]
]


@registry.register_problem
class OpensubtitlesViEn(translate.TranslateProblem):

    @property
    def approx_vocab_size(self):
        return 2**15  # 32768

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return OPENSUBTITLES_VIEN if train else _ENVI_TEST_DATASETS
    
OPENSUBTITLES_ENVI = [
     ['', ('train.en', 'train.vi')],  # original.
     ['', ('OpenSubtitles.en.subset', 'OpenSubtitles.vi.subset')]
]


@registry.register_problem
class OpensubtitlesEnVi(translate.TranslateProblem):

    @property
    def approx_vocab_size(self):
        return 2**15  # 32768

    def source_data_files(self, dataset_split):
        train = dataset_split == problem.DatasetSplit.TRAIN
        return OPENSUBTITLES_ENVI if train else _ENVI_TEST_DATASETS


