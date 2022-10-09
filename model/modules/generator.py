#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


"""
Generic PyTorch-based Generator agent.

Implements quite a bit of boilerplate, including forced-decoding loss and a tree search.

Contains the following utilities:

* `ref:TorchGeneratorAgent` class, which serves as a useful parent for generative torch
  agents.
* Beam class which provides some generic beam functionality for classes to use
"""

from abc import ABC, abstractmethod
from typing import TypeVar, List, Dict, Optional, Tuple, Set, Iterable
import math
from operator import attrgetter

import torch
import torch.nn as nn
import torch.nn.functional as F



try:
    from nltk.translate import bleu_score as nltkbleu

except ImportError:
    nltkbleu = None

try:
    from fairseq import bleu as fairseq_bleu

except ImportError:
    fairseq_bleu = None

NEAR_INF = 1e20
NEAR_INF_FP16 = 65504

def neginf(dtype: torch.dtype) -> float:
    """
    Return a representable finite number near -inf for a dtype.
    """
    if dtype is torch.float16:
        return -NEAR_INF_FP16
    else:
        return -NEAR_INF

TSType = TypeVar('TSType', bound='TreeSearch')

class _HypothesisTail(object):
    """
    Hold some bookkeeping about a hypothesis.
    """

    # use slots because we don't want dynamic attributes here
    __slots__ = ['timestep', 'hypid', 'score', 'tokenid']

    def __init__(self, timestep, hypid, score, tokenid):
        self.timestep = timestep
        self.hypid = hypid
        self.score = score
        self.tokenid = tokenid


class TreeSearch(object):
    """
    Abstract Tree Search class.

    It keeps information about beam_size concurrent, developing hypotheses. Concrete
    implementations make choices about which token to explore next at each point in the
    tree. Different choices result in different generation algorithms.
    """

    def __init__(
        self,
        beam_size,
        block_ngram=-1,
        context_block_ngram=-1,
        padding_token=0,
        bos_token=1,
        eos_token=2,
        min_length=3,
        device='cpu',
        length_penalty=0.65,
    ):
        """
        Instantiate Beam object.

        :param beam_size:
            number of hypothesis in the beam
        :param block_ngram:
            size of ngrams to block.
        :param context_block_ngram:
            size of context ngrams to block
        :param padding_token:
            padding token ID
        :param bos_token:
            beginning of sentence token ID
        :param eos_token:
            end of sentence token ID
        :param min_length:
            minimum length of the predicted sequence
        :param device:
            What device to use for computations
        """
        self.beam_size = beam_size
        self.length_penalty = length_penalty
        self.block_ngram = block_ngram
        self.min_length = min_length
        self.eos = eos_token
        self.bos = bos_token
        self.pad = padding_token
        self.context = None
        self.context_block_ngram = context_block_ngram
        # self.block_list: Optional[SearchBlocklist] = None
        self.device = device
        # recent score for each hypo in the beam
        self.scores = None
        # self.scores values per each time step
        self.all_scores = [torch.Tensor([0.0] * beam_size).to(self.device)]
        # backtracking id to hypothesis at previous time step
        self.bookkeep = []
        # output tokens at each time step
        self.outputs = [
            torch.Tensor(self.beam_size).long().fill_(self.bos).to(self.device)
        ]
        # keeps tuples (score, time_step, hyp_id)
        self.finished = []
        self.eos_top = False
        self.eos_top_ts = None
        self.n_best_counter = 0
        self.partial_hyps = [[self.bos] for i in range(beam_size)]

    def set_context(self: TSType, context: torch.LongTensor) -> TSType:
        """
        Set the internal context representation and return self.

        :param context:
            a LongTensor representing the input context; used for context
            ngram blocking, if supplied
        """
        self.context = context.tolist()
        return self

    # def set_block_list(self: TSType, block_list: Optional[SearchBlocklist]) -> TSType:
    #     self.block_list = block_list
    #     return self

    def get_output_from_current_step(self):
        """
        Get the outputput at the current step.
        """
        return self.outputs[-1]

    def get_backtrack_from_current_step(self):
        """
        Get the backtrack at the current step.
        """
        return self.bookkeep[-1]

    @abstractmethod
    def select_paths(self, logprobs, prior_scores, current_length):
        """
        Select the next vocabulary item in these beams.

        :param logprobs:
            a (beamsize x vocab) tensor of log probabilities. If this is the first
            turn in the dialogue, it will be a (1 x vocab) tensor.
        :param prior_scores:
            a (beamsize) tensor of weights with the cumulative running
            log-probability of each beam. If the first turn, it will be a (1) tensor.
        :param current_length:
            the current length in tokens
        :return:
            a (hypothesis_ids, token_id, scores) tuple, where:

            - hypothesis_ids is a LongTensor of hypotheses we're extending. May have
              repeats, but should always be (beamsize) long.
            - token_ids is a (beamsize) LongTensor of next-token choices for
              each of the hypotheses.
            - scores is a (beamsize) Tensor with the updated cumulative log-probs
              of each beam.
        """
        pass

    def _block_ngrams(
        self, ngram_size: int, logprobs: torch.Tensor, source: torch.LongTensor = None
    ):
        """
        Hard block ngrams from the logprobs, based on the source.

        :param ngram_size:
            The length of ngrams to block. Must be > 0.
        :param logprobs:
            Float or HalfTensor, representing the log-probabilities. This is
            modified in place.
        :param source:
            Source text to grab ngrams from. If None, it uses the current
            hypothesis (i.e. self-blocking).
        """
        for beam_id, hyp in enumerate(self.partial_hyps):
            if len(hyp) < ngram_size - 1:
                continue
            source_ = hyp if source is None else source
            ngrams = self._find_ngrams(source_, ngram_size)
            prefix = hyp[-(ngram_size - 1) :]
            for ngram in ngrams:
                if ngram_size == 1 or prefix == list(ngram[:-1]):
                    logprobs[beam_id][ngram[-1]] = neginf(logprobs.dtype)
        return logprobs

    def _block_block_list(self, logprobs: torch.Tensor) -> torch.Tensor:
        if self.block_list is None:
            return logprobs

    def advance(self, logprobs):
        """
        Advance the beam one step.
        """
        current_length = len(self.all_scores) - 1
        if current_length < self.min_length:
            # penalize all eos probs to make it decode longer
            for hyp_id in range(logprobs.size(0)):
                logprobs[hyp_id][self.eos] = neginf(logprobs.dtype)

        if self.scores is None:
            self.scores = torch.zeros(1).type_as(logprobs).to(logprobs.device)

        # penalize hypotheses ending in EOS on the prior scores (self.scores) level
        # this is related to search which uses prior scores (self.scores) (e.g. beam)
        for hyp_id, token in enumerate(self.outputs[-1]):
            if token == self.eos:
                self.scores[hyp_id] = neginf(self.scores.dtype)

        # beam blocking
        if self.block_ngram > 0:
            logprobs = self._block_ngrams(self.block_ngram, logprobs, None)

        logprobs = self._block_block_list(logprobs)

        if self.context_block_ngram > 0:
            if self.context is None:
                raise ValueError(
                    "Must use TreeSearch.set_context to use context blocking."
                )
            logprobs = self._block_ngrams(
                self.context_block_ngram, logprobs, self.context
            )

        hyp_ids, tok_ids, self.scores = self.select_paths(
            logprobs, self.scores, current_length
        )
        # use clone() here to ensure that self.all_scores will not be changed
        # later due to any penalties to self.scores
        self.all_scores.append(self.scores.clone())

        self.outputs.append(tok_ids)
        self.bookkeep.append(hyp_ids)
        self.partial_hyps = [
            self.partial_hyps[hyp_ids[i]] + [tok_ids[i].item()]
            for i in range(self.beam_size)
        ]

        #  check new hypos for eos label, if we have some, add to finished
        for hypid in range(self.beam_size):
            if self.outputs[-1][hypid] == self.eos:
                if self.scores[hypid] <= neginf(self.scores.dtype):
                    continue
                #  this is finished hypo, adding to finished
                eostail = _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=hypid,
                    score=self.all_scores[-1][hypid],
                    tokenid=self.eos,
                )
                self.finished.append(eostail)
                self.n_best_counter += 1

        if self.outputs[-1][0] == self.eos:
            self.eos_top = True
            if self.eos_top_ts is None:
                self.eos_top_ts = len(self.outputs) - 1

    def is_done(self):
        """
        Return whether beam search is complete.
        """
        return self.eos_top and self.n_best_counter >= self.beam_size

    def _find_ngrams(self, input_list, n):
        """
        Find ngrams of size n in input list.
        """
        return list(zip(*[input_list[i:] for i in range(n)]))

    def _get_hyp_from_finished(self, hypothesis_tail):
        """
        Extract hypothesis ending with EOS at timestep with hyp_id.

        :param timestep:
            timestep with range up to len(self.outputs) - 1

        :param hyp_id:
            id with range up to beam_size - 1

        :return:
            hypothesis sequence
        """
        hyp_idx = []
        endback = hypothesis_tail.hypid
        for i in range(hypothesis_tail.timestep, -1, -1):
            hyp_idx.append(
                _HypothesisTail(
                    timestep=i,
                    hypid=endback,
                    score=self.all_scores[i][endback],
                    tokenid=self.outputs[i][endback],
                )
            )
            endback = self.bookkeep[i - 1][endback]

        return hyp_idx

    def _get_pretty_hypothesis(self, list_of_hypotails):
        """
        Return hypothesis as a tensor of token ids.
        """
        return torch.stack([ht.tokenid for ht in reversed(list_of_hypotails)])

    def get_rescored_finished(self, n_best=None):
        """
        Return finished hypotheses according to adjusted scores.

        Score adjustment is done according to the Google NMT paper, which
        penalizes long utterances.

        :param n_best:
            number of finalized hypotheses to return

        :return:
            list of (tokens, score) pairs, in sorted order, where:
              - tokens is a tensor of token ids
              - score is the adjusted log probability of the entire utterance
        """
        # if we never actually finished, force one
        if not self.finished:
            self.outputs[-1][0] = self.eos
            self.finished.append(
                _HypothesisTail(
                    timestep=len(self.outputs) - 1,
                    hypid=0,
                    score=self.all_scores[-1][0],
                    tokenid=self.outputs[-1][0],
                )
            )

        rescored_finished = []
        for finished_item in self.finished:
            current_length = finished_item.timestep + 1
            # these weights are from Google NMT paper
            length_penalty = math.pow((1 + current_length) / 6, self.length_penalty)
            rescored_finished.append(
                _HypothesisTail(
                    timestep=finished_item.timestep,
                    hypid=finished_item.hypid,
                    score=finished_item.score / length_penalty,
                    tokenid=finished_item.tokenid,
                )
            )

        # Note: beam size is almost always pretty small, so sorting is cheap enough
        srted = sorted(rescored_finished, key=attrgetter('score'), reverse=True)

        if n_best is not None:
            srted = srted[:n_best]

        n_best_list = [
            (self._get_pretty_hypothesis(self._get_hyp_from_finished(hyp)), hyp.score)
            for hyp in srted
        ]

        # check that there is at least one finished candidate
        # and assert that each of them contains only one EOS
        assert (
            len(n_best_list) >= 1
        ), f'TreeSearch returned {len(n_best_list)} candidates, must be >= 1'
        for (pred, score) in n_best_list:
            assert (
                pred == self.eos
            ).sum() == 1, f'TreeSearch returned a finalized hypo with multiple end tokens \
            with score {score.item():.2f}'

        return n_best_list


class GreedySearch(TreeSearch):
    """
    Greedy search.

    Picks the highest probability utterance at each step.  Only works with
    --beam-size 1.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.beam_size != 1:
            raise ValueError('Greedy search can only be run with beam size 1.')

    def select_paths(self, logprobs, prior_scores, current_length):
        tok_scores, tok_ids = logprobs.max(1)
        best_scores = tok_scores + prior_scores
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        return (hyp_ids, tok_ids, best_scores)


class BeamSearch(TreeSearch):
    """
    Beam search.
    """

    def select_paths(self, logprobs, prior_scores, current_length):
        """
        Select the next vocabulary item in these beams.
        """
        # if numel is 1, then this is the first time step, only one hyp is expanded
        if prior_scores.numel() == 1:
            logprobs = logprobs[0:1]

        # beam search actually looks over all hypotheses together so we flatten
        beam_scores = logprobs + prior_scores.unsqueeze(1).expand_as(logprobs)
        flat_beam_scores = beam_scores.view(-1)
        best_scores, best_idxs = torch.topk(flat_beam_scores, self.beam_size, dim=-1)
        voc_size = logprobs.size(-1)

        # get the backtracking hypothesis id as a multiple of full voc_sizes
        hyp_ids = best_idxs // voc_size
        # get the actual word id from residual of the same division
        tok_ids = best_idxs % voc_size

        return (hyp_ids, tok_ids, best_scores)


class DelayedBeamSearch(TreeSearch):
    """
    DelayedBeam: Top-K sampling followed by beam search (Massarelli et al., 2019).

    Samples from a truncated distribution where only the most probable K words
    are considered at each time for the first N tokens, then switches to beam
    after N steps.

    See https://arxiv.org/abs/1911.03587 for details.
    """

    def __init__(self, k, delay, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k
        self.delay = delay

    def select_paths(self, logprobs, prior_scores, current_length):
        if current_length < self.delay:
            return TopKSampling.select_paths(
                self, logprobs, prior_scores, current_length
            )
        else:
            return BeamSearch.select_paths(self, logprobs, prior_scores, current_length)


class TopKSampling(TreeSearch):
    """
    Top-K sampling (Fan et al., 2018).

    Samples from a truncated distribution where only the most probable K words
    are considered at each time.

    Typical values of k are 2, 10, 50.

    See https://arxiv.org/abs/1805.04833 for details.
    """

    def __init__(self, k, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.k = k

    def select_paths(self, logprobs, prior_scores, current_length):
        values, indices = logprobs.topk(self.k, dim=-1)
        probs = torch.softmax(values, dim=-1)
        choices = torch.multinomial(probs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = indices[hyp_ids, choices]
        scores = values[hyp_ids, choices]
        best_scores = prior_scores.expand_as(scores) + scores
        return (hyp_ids, tok_ids, best_scores)


class NucleusSampling(TreeSearch):
    """
    Nucelus, aka top-p sampling (Holtzman et al., 2019).

    Samples from a truncated distribution which covers a fixed CDF proportion
    of the original distribution.

    Typical values of p are 0.3 and 0.9.

    See https://arxiv.org/abs/1904.09751 for details.
    """

    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p

    def select_paths(self, logprobs, prior_scores, current_length):
        # Unlike the other treesearch methods, we have to switch to linspace
        # for the probabilities in order to compute the CDF.
        probs = torch.softmax(logprobs, dim=-1)
        sprobs, sinds = probs.sort(dim=-1, descending=True)
        # The subtraction here is to get the exclusive prefix sum,
        # to guarantee the first element is not masked
        mask = (sprobs.cumsum(dim=-1) - sprobs) >= self.p
        sprobs[mask] = 0
        sprobs.div_(sprobs.sum(dim=-1).unsqueeze(1))
        choices = torch.multinomial(sprobs, 1)[:, 0]
        hyp_ids = torch.arange(logprobs.size(0)).to(logprobs.device)
        tok_ids = sinds[hyp_ids, choices]
        # Convert back to logspace.
        scores = sprobs[hyp_ids, choices].log()
        best_scores = prior_scores.expand_as(scores) + scores
        return (hyp_ids, tok_ids, best_scores)

if __name__ == "__main__":
    batch_size = 4
    beam_size = 5
    g = DelayedBeamSearch(beam_size=beam_size, k=3, delay=6)
    print(g)
    max_len = 50
    prior_scores = torch.zeros(batch_size*beam_size)
    for i in range(max_len):
        logprobs = torch.rand(batch_size, beam_size, 10)
        logprobs = logprobs.view(batch_size*beam_size, 10)
        hyp_ids, tok_ids, best_scores = g.select_paths(logprobs, prior_scores, i+1)
        prior_scores += best_scores
        print(f'logprobs: {logprobs}')
        print(f'hyp_ids: {hyp_ids}, '
              f'tok_ids: {tok_ids}, '
              f'best_scores: {best_scores}')
    print(g)