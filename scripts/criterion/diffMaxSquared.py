# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from unittest.main import MAIN_EXAMPLES
from .custom import extract_from_sample

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import torch

@dataclass
class MultiRefDiffMaxSquaredLossConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("multi_ref_diff_max_squared_loss", dataclass=MultiRefDiffMaxSquaredLossConfig)
class MultiRefDiffMaxSquaredLoss(FairseqCriterion):
    def __init__(self, task, sentence_avg):
        super().__init__(task)
        self.sentence_avg = sentence_avg

        if hasattr(task, "target_dictionary"):
            tgt_dict = task.target_dictionary
            self.padding_idx = tgt_dict.pad() if tgt_dict is not None else -100
            self.eos_idx = tgt_dict.eos() if tgt_dict is not None else -100
            self.sep_idx = tgt_dict.sep() if tgt_dict is not None else -100
        
        # taskからargを受け取る
        if hasattr(task, "criterion_alpha"):
            self.criterion_alpha = task.criterion_alpha

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """

        ###
        # 1. sampleを分解する
        ###
        additional = None

        # 学習時のみ分解する
        if model.training == True:
            additional, sample =\
                extract_from_sample(
                    sample, 
                    padding_idx=self.padding_idx, 
                    eos_idx=self.eos_idx,
                    sep_idx=self.sep_idx
                    )

        ###
        # 2. modelにデータを流す
        ###
        net_output = model(**sample["net_input"])

        ###
        # 3. lossの計算をする
        ###
        loss, _ = self.compute_loss(
            model, 
            net_output, 
            sample, 
            additional, 
            reduce=reduce,
            alpha=self.criterion_alpha
            )

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, additional, reduce=True, alpha=0.1):
        
        ###
        # 1. (既存) 出力確率の取得
        ###
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))

        ###
        # 2. (既存) 参照文の取得
        ###
        target = model.get_targets(sample, net_output).view(-1)

        ###
        # 3. (変更) loss計算
        ###

        # 学習時は提案損失関数を用いる
        if model.training == True:
            loss = self.my_loss_loop_01(
                    lprobs, 
                    target, 
                    additional, 
                    padding_idx=self.padding_idx, 
                    eos_idx=self.eos_idx,
                    sep_idx=self.sep_idx,
                    alpha=alpha
            )
        
        # 推論時は通常のcross entropyを用いる
        else:
            loss = F.nll_loss(
                lprobs,
                target,
                ignore_index=self.padding_idx,
                reduction="sum" if reduce else "none",
            )


        return loss, loss

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        if sample_size != ntokens:
            metrics.log_scalar(
                "nll_loss", loss_sum / ntokens / math.log(2), ntokens, round=3
            )
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
            )
        else:
            metrics.log_derived(
                "ppl", lambda meters: utils.get_perplexity(meters["loss"].avg)
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True


    def my_loss_loop_01(
        self,
        lprobs,
        target,
        additional,
        padding_idx=1,
        eos_idx=2,
        sep_idx=4,
        alpha=0.1
        ):
        "分割した各データを利用し提案損失の計算を行う"

        max_length      = additional["max_length"]

        # デコーダの出力確率の取得
        lprobs_set = torch.split(lprobs, max_length)

        # 目的参照文の取得
        main_targets = torch.split(target, max_length)

        sub_targets_set = additional["sub_targets_set"]
        sub_LoDs_set    = additional["sub_LoDs_set"]
        main_LoDs       = additional["main_LoDs"]

        loss = None
        for (main_LoD, main_target, lprobs, sub_LoDs, sub_targets) in \
            zip(main_LoDs, main_targets, lprobs_set, sub_LoDs_set, sub_targets_set):

            # deviceを一致させる
            lprobs      = lprobs.cuda()
            main_target = main_target.to(lprobs.device)

            # 目的参照文のloss計算
            L_main = F.nll_loss(
                        lprobs,
                        main_target,
                        ignore_index=padding_idx,
                        reduction="sum",
                    )

            sum_diff    = None
            sub_length  = len(sub_LoDs_set)

            # 副参照文と難易度のloop
            for (sub_LoD, sub_target) in zip(sub_LoDs, sub_targets):

                # deviceを一致させる
                sub_target = sub_target.to(lprobs.device)

                # 副参照文のloss計算
                L_sub = F.nll_loss(
                        lprobs,
                        sub_target,
                        ignore_index=padding_idx,
                        reduction="sum",
                    )

                # 差の計算
                ###
                # d_i * max((L_main - L_sub), 0)
                ###
                if (L_main - L_sub) > 0:
                    tmp_diff = (torch.square(main_LoD - sub_LoD) * (L_main - L_sub))
                else:
                    tmp_diff = (torch.square(main_LoD - sub_LoD) * 0.0)

                tmp_diff = tmp_diff.to(lprobs.device)

                # 比の加算
                if sum_diff == None:
                    sum_diff = tmp_diff
                else:
                    sum_diff+= tmp_diff

            ###
            # L_main + alpha * 1/n * sum(abs(d_i * L_main - L_sub)))
            ###
            tmp_loss = L_main + (alpha * (1 / sub_length) * sum_diff)

            # 加算
            if loss == None:
                loss = tmp_loss
            else:
                loss+=tmp_loss
                        
        return loss
