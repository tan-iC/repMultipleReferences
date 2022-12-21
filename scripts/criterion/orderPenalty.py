# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from dataclasses import dataclass
from unittest.main import MAIN_EXAMPLES

import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

import torch
import  itertools               as it

@dataclass
class MultiRefOrderPenaltyLossConfig(FairseqDataclass):
    sentence_avg: bool = II("optimization.sentence_avg")

@register_criterion("multi_ref_order_penalty_loss", dataclass=MultiRefOrderPenaltyLossConfig)
class MultiRefOrderPenaltyLoss(FairseqCriterion):
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
                self.extract_from_sample(
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

    def eosSplit(self, target, padding_idx=1, eos_idx=2):
        "参照文のトークン列targetをeosで分割する"
        eos_flag = False
        tmp = []
        targets = []
        for i, tkn in enumerate(target):

            if (eos_flag == True) and (tkn != padding_idx):
                targets.append(tmp)
                tmp = []
                eos_flag = False

            tmp.append(tkn)

            if (tkn == eos_idx) :
                eos_flag = True

            if (i == len(target)-1):
                targets.append(tmp)
                tmp = []

        return torch.LongTensor(targets)

    def listPadding(self, ref, tgt_len, padding_idx=1, eos_idx=2, prev=False):
        "list::refをint::tgt_lenまでpadding"
        refPadded = ref

        while(len(refPadded) != tgt_len):
            refPadded.append(padding_idx)

        return refPadded

    def sepSplit(self, targets, tgt_len=None, eos_idx=2, sep_idx=4, prev=False):
        "参照文のトークン列のリストであるtargetsのそれぞれをsepで区切る"
        tmp     = []
        tgts    = []
        tgt_set = []

        if tgt_len == None:
            tgt_len = targets.size(1)

        # 難易度（LoD : Level_of_Difficulty）
        LoDs    = []
        LoDs_set= []

        for target in targets:

            for i, tkn in enumerate(target):
                if (prev==True) and (tkn==eos_idx):
                    continue

                if (tkn == sep_idx) :
                    if prev==False :
                        tmp.append(eos_idx)

                    # 難易度を取り出す
                    LoDs.append(tmp[0])

                    if prev==True :
                        tmp.insert(1, eos_idx)

                    tmp = self.listPadding(tmp[1:], tgt_len, prev=prev)
                    tgts.append(tmp)

                    tmp = []

                else :
                    tmp.append(tkn)


                if (i == len(target)-1):

                    # 難易度を取り出す
                    LoDs.append(tmp[0])


                    if prev==True :
                        tmp.insert(1, eos_idx)

                    tmp = self.listPadding(tmp[1:], tgt_len, prev=prev)
                    tgts.append(tmp)

                    tmp = []
            
            tgt_set.append(torch.LongTensor(tgts))
            tgts = []

            # 難易度
            LoDs_set.append(torch.LongTensor(LoDs))
            LoDs = []

        return tgt_set, LoDs_set

    def make_main_LoDs(self, net_inputs, padding_idx=1):
        "mainの難易度を取得する"
        main_LoDs = []

        for net_input in net_inputs:

            for tkn in net_input:
                if tkn == padding_idx:
                    continue

                main_LoDs.append(tkn)
                break

        return torch.LongTensor(main_LoDs)

    def get_max_length(self, tensors):
        "tensorsの最大文長を求める"
        tmp = []
        for tensor in tensors:
            for tokens in tensor:
                tokens = tokens[torch.where(tokens > 1)]
                tmp.append(len(tokens))

        return max(tmp)

    def output_padding(self, tensors, max_length, prev=False):
        "最大文長に合わせる"
        out = []
        for tensor in tensors:
            for tokens in tensor:
                tokens = tokens[torch.where(tokens > 1)]
                out.append(self.listPadding(tokens.tolist(), tgt_len=max_length, prev=prev))

        return torch.LongTensor(out)

    def extract_from_sample(self, sample, padding_idx=1, eos_idx=2, sep_idx=4):

        # データの用意
        src_tokens  = sample["net_input"]["src_tokens"]
        prev_output = sample["net_input"]["prev_output_tokens"]
        target      = sample["target"]

        # mainの参照文の難易度を取得
        main_LoDs = self.make_main_LoDs(src_tokens)

        # sepトークンによる分割
        targets, LoDs_set       = self.sepSplit(target)
        prev_outputs, _LoDs_set = self.sepSplit(prev_output, prev=True)

        # paddingを除く最大文長の取得
        max_length = self.get_max_length(targets)

        # 和集合の最大文長に合わせてpadding
        targets         = self.output_padding(targets, max_length)
        prev_outputs    = self.output_padding(prev_outputs, max_length, prev=True)

        # mainとsubの分類
        i = 0
        sub_targets_set = []
        sub_LoDs_set    = []
        main_targets    = []
        main_prev_output_tokens     = []
        sub_prev_output_tokens_set  = []

        for (LoDs, main_LoD) in zip(LoDs_set, main_LoDs):
            tmp_prev    = []
            tmp_targets = []
            tmp_LoDs    = []

            for LoD in LoDs:
                if LoD == main_LoD:
                    main_prev_output_tokens.append(prev_outputs[i].tolist())
                    main_targets.append(targets[i].tolist())
                
                else:
                    tmp_prev.append(prev_outputs[i].tolist())
                    tmp_targets.append(targets[i].tolist())
                    tmp_LoDs.append(LoD)
                i += 1

            sub_prev_output_tokens_set.append(torch.LongTensor(tmp_prev))
            sub_targets_set.append(torch.LongTensor(tmp_targets))
            sub_LoDs_set.append(torch.LongTensor(tmp_LoDs))

        main_targets = torch.LongTensor(main_targets)
        main_prev_output_tokens = torch.LongTensor(main_prev_output_tokens)

        # loss計算時に利用する情報のディクショナリを作成
        additional = {
            "max_length": max_length,
            "sub_targets_set": sub_targets_set,
            "sub_prev_output_tokens_set": sub_prev_output_tokens_set,
            "main_LoDs": main_LoDs,
            "sub_LoDs_set": sub_LoDs_set,
        }

        # sampleの変数の更新
        sample.update(target=main_targets)
        sample["net_input"].update(prev_output_tokens=main_prev_output_tokens)

        return additional, sample

    ###
    ###
    ###
    def make_dict(self, keys, vals):
        "2つのリストからkey,valueの型に依らずディクショナリを作成する"
        d = {k : v for (k, v) in zip(keys, vals)}
        return d

    ###
    ###
    ###
    def make_combination(self, LoDs, losses, main_LoD):
        "List[dict]の形式で条件を満たす難易度とlossの組合せを返す"

        # 難易度の数値をkey、lossをvalueとする辞書を作成
        LoD2loss = self.make_dict(LoDs, losses)

        # すべての難易度から仮の組合せを作成する
        tmp_combination = it.combinations(LoDs, 2)

        # 条件を満たす組合わせを残す
        tmp = []
        for comb in tmp_combination:
            i = comb[0]
            j = comb[1]

            if ((main_LoD - i)*(main_LoD - j) >= 0) and (i < j):
                d = {
                    "LoD" : [i, j],
                    "loss" : [LoD2loss[i], LoD2loss[j]]
                }
                tmp.append(d)
        return tmp


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

            # 計算に用いる要素を格納するリスト
            losses  = [L_main]
            LoDs    = [main_LoD]

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

                # 計算に用いる要素をそれぞれ追加
                losses.append(L_sub)
                LoDs.append(sub_LoD)
            
            # 計算が必要な組合せの辞書のリストを作成
            comb_dicts = self.make_combination(LoDs, losses, main_LoD)


            tmp = torch.tensor(0.0, device=lprobs.device)

            for comb_dict in comb_dicts:

                # 辞書から計算に利用する要素を取り出す
                L_sub_i = comb_dict["loss"][0]
                L_sub_j = comb_dict["loss"][1]
                sub_LoD_i = comb_dict["LoD"][0]

                # penaltyの計算
                penalty = torch.tensor(0.0, device=lprobs.device)

                if (L_sub_i - L_sub_j) * (main_LoD - sub_LoD_i) < 0:
                    penalty = F.huber_loss(L_sub_i, L_sub_j)
                
                    # 加算
                    tmp += penalty


            # 加算
            if loss == None:
                loss = L_main + alpha * tmp
            else:
                loss+= L_main + alpha * tmp

        return loss
