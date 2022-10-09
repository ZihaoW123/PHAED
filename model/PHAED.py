import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import context_cache as ctx
from model.modules.embeddings import Embedder
from model.modules.position_embedding import RelativeSegmentEmbeddings
from model.modules.sublayers import PositionwiseFeedForward, MultiHeadedAttention, MultiHeadedAttentionRelative
from model.modules.mem_transformer import MemTransformerLM
from model.modules.transformer_xl_utils.parameter_init import *
from utils import *



class EncoderBlock(nn.Module):
    attention_cls = {
        "normal": MultiHeadedAttention,
        "relative": MultiHeadedAttentionRelative
    }
    def __init__(self, d_model, d_inner_hid, n_head, d_head, dropout=0.1, attention_type="normal"):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)

        attn_cls = EncoderBlock.attention_cls[attention_type]
        self.slf_attn = attn_cls(d_model=d_model, n_head=n_head, d_head=d_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model=d_model, d_inner=d_inner_hid, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_input, slf_attn_mask=None, rel_attn_kv=(None, None)):
        input_norm = self.layer_norm(enc_input)
        context, attn_map, _ = self.slf_attn(input_norm, input_norm, input_norm, slf_attn_mask, rel_attn_kv=rel_attn_kv)
        out = self.dropout(context) + enc_input
        return self.pos_ffn(out), attn_map


class Encoder(nn.Module):
    def __init__(self,
                 embedder, n_layers, n_head,
                 d_model, d_inner_hid, dim_per_head,
                 num_pos_embeddings, num_turn_embeddings,
                 padding_idx, dropout, proj_share_weight=True, **kwargs):
        super(Encoder, self).__init__()
        self.pad = padding_idx
        self.embedder = embedder
        self.block_stack = nn.ModuleList(
            [EncoderBlock(d_model, d_inner_hid, n_head, dim_per_head, dropout) for _ in range(n_layers)])
        self.global_encoder_layer = EncoderBlock(d_model, d_inner_hid, n_head, dim_per_head, dropout, attention_type="relative")
        self.global_rel_seg_k = RelativeSegmentEmbeddings(
                                 num_turn_embeddings, d_model // n_head)
        self.global_rel_seg_v = RelativeSegmentEmbeddings(
                                 num_turn_embeddings, d_model // n_head)
        self.layer_norm = nn.LayerNorm(d_model)

    ###########  prepare mask
    def _prepare_local_mask(self, segment_ids, enc_mask):
        # segment_ids: [bsz, len]. [[0,0,0,1,1,1,2,2,2,2...], ...]
        # enc_mask: [bsz, qlen, mlen] (qlen==mlen)
        # return: [bsz, qlen, mlen] (qlen==mlen). [[[0,0,0,1,1,1,1,1,1,1...], [1,1,1,0,0,0,1,1,1,1...],...]]
        local_mask = []
        for i in range(segment_ids.size(-1)):
            # [bsz, mlen]
            local_mask.append(
                segment_ids.ne(
                    segment_ids[:, i:i + 1].expand_as(segment_ids)
                )
            )
        local_mask = torch.stack(local_mask, 1)
        return (local_mask + enc_mask).gt(0)

    def _prepare_global_lm_mask(self, segment_ids, enc_mask):
        # segment_ids: [bsz, len]. [[0,0,0,1,1,1,2,2,2,2...], ...]
        # enc_mask: [bsz, qlen, mlen] (qlen==mlen)
        # return: [bsz, qlen, mlen] (qlen==mlen). [[[0,0,0,1,1,1,1,1,1,1...], [1,1,1,0,0,0,1,1,1,1...],...]]
        local_mask = []
        for i in range(segment_ids.size(-1)):
            # [bsz, mlen]
            local_mask.append(
                segment_ids.gt(
                    segment_ids[:, i:i + 1].expand_as(segment_ids)
                )
            )
        local_mask = torch.stack(local_mask, 1)
        return (local_mask + enc_mask).gt(0)

    def _prepare_masks(self, enc_mask, segment_ids):
        # enc_mask: padding mask [bsz, src_len]
        global_mask = enc_mask[:, None, :].expand(*enc_mask.size(), enc_mask.size(1))

        local_mask = self._prepare_local_mask(segment_ids, global_mask)
        global_lm_mask = self._prepare_global_lm_mask(segment_ids, global_mask)
        encoder_self_attention_mask = local_mask
        return encoder_self_attention_mask, global_mask, global_lm_mask

    def _prepare_segment_relative(self, segment_ids):
        # segment-level relative attention related
        return self.global_rel_seg_k(segment_ids), self.global_rel_seg_v(segment_ids)

    def forward(self, src, position, segment_ids, turn_inp, attn_map=False):
        # src: [batch, all_lengths],
        # position: [batch, all_lengths],
        # segment_ids: [batch, all_lengths]
        # turn_inp: [batch, all_lengths]
        enc_mask = src.detach().eq(self.pad)  # [batch, turns_src*lengths]
        emb = self.embedder(token_inp=src, turn_inp=turn_inp, pos_inp=position)
        encoder_self_attention_mask, global_attention_mask, global_lm_mask = \
            self._prepare_masks(
                enc_mask=enc_mask,
                segment_ids=segment_ids)
        out = emb
        for layer in self.block_stack:
            out, _ = layer(out, encoder_self_attention_mask)
        if ctx.GLOBAL_ENCODING:
            seg_rel_k, seg_rel_v = self._prepare_segment_relative(segment_ids)
            out, attn_maps= self.global_encoder_layer(out, global_lm_mask, rel_attn_kv=[seg_rel_k, seg_rel_v])
        out_enc = out
        if attn_map:
            return self.layer_norm(out_enc), enc_mask, attn_maps
        return self.layer_norm(out_enc), enc_mask 


class Generator(nn.Module):

    def __init__(self, n_words, hidden_size, shared_weight=None, padding_idx=0, eos=EOS_ID):
        super(Generator, self).__init__()

        self.n_words = n_words
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.eos = eos

        self.proj = nn.Linear(self.hidden_size, self.n_words, bias=False)

        if shared_weight is not None:
            self.proj.weight = shared_weight

    def _pad_2d(self, x, no_eos=False):

        if self.padding_idx == -1:
            return x
        else:
            x_size = x.size()
            x_2d = x.view(-1, x.size(-1))

            mask = x_2d.new(1, x_2d.size(-1)).zero_()
            mask[0][self.padding_idx] = float('-inf')
            if no_eos:
                mask[0][self.eos] = float('-inf')
            x_2d = x_2d + mask

            return x_2d.view(x_size)

    def forward(self, input, log_probs=True, no_eos=False):
        """
        input == > Linear == > LogSoftmax
        """
        logits = self.proj(input)
        # no_eos=input.size(1) <= 3
        logits = self._pad_2d(logits, no_eos=no_eos)

        if log_probs:
            return F.log_softmax(logits, dim=-1)
        else:
            return F.softmax(logits, dim=-1)



class PHAED(nn.Module):
    def __init__(self,
                 input_size, output_size,
                 num_encoder_layers=6, num_decoder_layers=6, n_head=8, d_word_vec=512, d_model=512, d_inner_hid=1024, dim_per_head=None,
                 num_pos_embeddings=52, num_turn_embeddings=20,
                 padding_idx=0, sos_idx=1, dropout=0.1, proj_share_weight=True, teach_force=1.0, **kwargs):
        super(PHAED, self).__init__()
        self.teach_force = teach_force
        self.output_size = output_size
        self.pad, self.sos = padding_idx, sos_idx
        self.embedder = Embedder(d_word_vec, input_size,
                                 num_pos_embeddings, num_turn_embeddings,
                                 padding_idx, dropout)

        self.encoder = Encoder(
            embedder=self.embedder, n_layers=num_encoder_layers, n_head=n_head,
            d_model=d_model, d_inner_hid=d_inner_hid, dim_per_head=dim_per_head,
            num_pos_embeddings=num_pos_embeddings, num_turn_embeddings=num_turn_embeddings,
            padding_idx=padding_idx, dropout=dropout, **kwargs)

        tgt_len, mem_len, ext_len = 49, 100, 0
        self.decoder = MemTransformerLM(embedder=self.embedder,
                                        n_token=output_size, n_layer=num_decoder_layers, n_head=n_head,
                                        d_model=d_model, d_head=dim_per_head,
                                        d_embed=d_word_vec,
                                        dropout=dropout, dropatt=dropout,
                                        d_inner=d_inner_hid, attn_type=0,
                                        tgt_len=tgt_len, mem_len=mem_len, ext_len=ext_len,
                                        pre_lnorm=True)
        self.decoder.apply(weights_init)
        self.decoder.word_emb.apply(weights_init)

        weight_ = None
        self.proj_share_weight = proj_share_weight
        if self.proj_share_weight:
            weight_ = self.embedder.token_embeddings.weight
        self.generator = Generator(output_size, d_model, shared_weight=weight_, padding_idx=self.pad)
        assert input_size == output_size
        

    def forward(self, src, tgt, lengths, log_probs=True):
        

        return 0


    def encode(self, src, src_turn_inp):
        ctx, ctx_mask = self.encoder(src, src_turn_inp)
        return {"ctx": ctx, "ctx_mask": ctx_mask}

    def decode_train(self, tgt_seq, tgt_turn_id, enc_out, enc_mask, log_probs=True):
        # tgt_seq [length_tgt, batch]
        # enc_out [length, batch, d_model], enc_mask [length, batch]
        # kw_seq [length_kw, batch]
        dec_inp = tgt_seq
        dec_inp_T = dec_inp.transpose(0, 1)
        enc_out_T = enc_out.transpose(0, 1)
        enc_mask_T = enc_mask.transpose(0, 1)
        # ctx.memory_cache
        if not ctx.ENABLE_CONTEXT:
            ctx.memory_cache = tuple()
            ctx.memory_mask = None
        dec_pred_T, ctx.memory_cache = self.decoder(dec_inp_T, tgt_turn_id, enc_out_T, enc_mask_T, *ctx.memory_cache)
        dec_pred = dec_pred_T.transpose(0, 1).contiguous()
        return self.generator(dec_pred, log_probs=log_probs)

    def finish_decoder(self):
        for layer in self.decoder.layers:
            layer.ctx_attn.attn_cache = None
    

    def predict_beam(self, src, sbatch_turn_id, tbatch, tbatch_turn_id,
                     maxlen=30, beam_size=5, alpha=0.65, loss=True):
        # src: [enc_out, batch]
        sbatch, sents_mapping, sent_rank = src
        with torch.no_grad():
            enc_out, enc_mask = self.encoder(sbatch, position=sent_rank,
                                              segment_ids=sents_mapping, turn_inp=sbatch_turn_id)
            ctx.memory_cache = tuple()
            ctx.memory_mask = None
            y_batch = torch.einsum('btl->tbl', tbatch)
            n_sents = y_batch.size(0)
            for sents_no, y_sents in enumerate(y_batch):
                y_inp = y_sents[:, :-1].contiguous()
                y_label = y_sents[:, 1:].contiguous()
                is_not_current_sents = sents_mapping.detach().gt(sents_no)
                current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents.long(),
                                                enc_mask.long()).bool()
                y_turn_id = tbatch_turn_id[:, sents_no]
                log_probs = self.decode_train(y_inp, y_turn_id, enc_out, current_sent_mask, log_probs=True)
                new_mask = y_label.eq(PAD_ID).view(-1, y_label.size(-1)).transpose(0, 1)
                ctx.memory_mask = new_mask if ctx.memory_mask is None else torch.cat([ctx.memory_mask, new_mask], dim=0)

            batch_size = sbatch.shape[0]
            beam_scores = torch.zeros((batch_size, beam_size))
            beam_scores[:, 1:] = -1e9
            beam_scores = beam_scores.view(-1)
            done = [False for _ in range(batch_size)]  # 标记每个输入句子的beam search是否完成
            generated_hyps = [
                BeamHypotheses(beam_size, maxlen, length_penalty=alpha)
                for _ in range(batch_size)
            ]

            input_ids = torch.full((batch_size * beam_size, 1), self.sos, dtype=torch.long)
            cur_len = 1
            if torch.cuda.is_available():
                input_ids = input_ids.cuda()
                beam_scores = beam_scores.cuda()
            ################## decoder state
            tgt_turn_id = tbatch_turn_id.new_zeros(batch_size)
            enc_out_T = enc_out.transpose(0, 1)
            is_not_current_sents = sents_mapping.detach().gt(n_sents)
            current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents.long(),
                                            enc_mask.long()).bool()
            enc_mask_T = current_sent_mask.transpose(0, 1)

            while cur_len < maxlen:
                # outputs: (batch_size*num_beams, cur_len, vocab_size)
                input_ids = input_ids.view(batch_size, beam_size, cur_len)
                dec_pred = []
                for i in range(beam_size):
                    input_ids_tmp = input_ids[:, i]
                    dec_inp_T = input_ids_tmp.transpose(0, 1)
                    dec_pred_T, _ = self.decoder(dec_inp_T, tgt_turn_id, enc_out_T, enc_mask_T, *ctx.memory_cache)
                    dec_pred_tmp = dec_pred_T.transpose(0, 1).contiguous() ##[batch_size, length, vocab_size]
                    dec_pred.append(dec_pred_tmp)
                input_ids = input_ids.view(batch_size*beam_size, cur_len)
                dec_pred = torch.stack(dec_pred, dim=1)
                dec_pred = dec_pred.view(batch_size*beam_size, cur_len, -1)
                # 取最后一个timestep的输出 (batch_size*num_beams, vocab_size)
                next_token_logits = dec_pred[:, -1, :]
                scores = self.generator(next_token_logits, log_probs=True)# log_softmax (batch_size*num_beams, vocab_size)
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # 累加上以前的scores
                next_scores = next_scores.view(
                    batch_size, beam_size * self.output_size
                )  # 转成(batch_size, num_beams * vocab_size), 如上图所示
                # 取topk
                # next_scores: (batch_size, num_beams) next_tokens: (batch_size, num_beams)
                next_scores, next_tokens = torch.topk(next_scores, beam_size, dim=1, largest=True, sorted=True)
                next_batch_beam = []
                for batch_idx in range(batch_size):
                    # if done[batch_idx]:
                    #    # 当前batch的句子都解码完了，那么对应的num_beams个句子都继续pad
                    #    next_batch_beam.extend([(0, self.pad, 0)] * beam_size)  # pad the batch
                    #    continue
                    next_sent_beam = []  # 保存三元组(beam_token_score, token_id, effective_beam_id)
                    for beam_token_rank, (beam_token_id, beam_token_score) in enumerate(zip(next_tokens[batch_idx], next_scores[batch_idx])):
                        beam_id = beam_token_id // self.output_size  # 1
                        token_id = beam_token_id % self.output_size  # 1
                        # 上面的公式计算beam_id只能输出0和num_beams-1, 无法输出在(batch_size, num_beams)中的真实id
                        # 如上图, batch_idx=0时，真实beam_id = 0或1; batch_idx=1时，真实beam_id如下式计算为2或3
                        # batch_idx=1时，真实beam_id如下式计算为4或5
                        effective_beam_id = batch_idx * beam_size + beam_id
                        # 如果遇到了eos, 则讲当前beam的句子(不含当前的eos)存入generated_hyp
                        # 保存第beam_id个句子累加到当前的log_prob以及当前的token_id
                        next_sent_beam.append((beam_token_score, token_id, effective_beam_id))

                        if len(next_sent_beam) == beam_size:
                            break
                        # 当前batch是否解码完所有句子
                        done[batch_idx] = done[batch_idx] or generated_hyps[batch_idx].is_done(next_scores[batch_idx].max().item(), cur_len)  # 注意这里取当前batch的所有log_prob的最大值
                        # 每个batch_idx, next_sent_beam中有num_beams个三元组(假设都不遇到eos)
                        # batch_idx循环后，extend后的结果为num_beams * batch_size个三元组
                    next_batch_beam.extend(next_sent_beam)
                    # 如果batch中每个句子的beam search都完成了，则停止
                if all(done):
                    break
                # 准备下一次循环(下一层的解码)
                # beam_scores: (num_beams * batch_size)
                # beam_tokens: (num_beams * batch_size)
                # beam_idx: (num_beams * batch_size)
                # 这里beam idx shape不一定为num_beams * batch_size，一般是小于等于
                # 因为有些beam id对应的句子已经解码完了 (下面假设都没解码完)
                beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
                beam_tokens = input_ids.new([x[1] for x in next_batch_beam])
                beam_idx = input_ids.new([x[2] for x in next_batch_beam])
                # 取出有效的input_ids, 因为有些beam_id不在beam_idx里面,
                # 因为有些beam id对应的句子已经解码完了
                input_ids = input_ids[beam_idx, :]  # (num_beams * batch_size, seq_len)
                # (num_beams * batch_size, seq_len) ==> (num_beams * batch_size, seq_len + 1)
                input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
                cur_len = cur_len + 1
            # 注意有可能到达最大长度后，仍然有些句子没有遇到eos token，这时done[batch_idx]是false
            for batch_idx in range(batch_size):
                if done[batch_idx]:
                    continue
                for beam_id in range(beam_size):
                    # 对于每个batch_idx的每句beam，都执行加入add
                    # 注意这里已经解码到max_length长度了，但是并没有遇到eos，故这里全部要尝试加入
                    effective_beam_id = batch_idx * beam_size + beam_id
                    final_score = beam_scores[effective_beam_id].item()
                    final_tokens = input_ids[effective_beam_id]
                    generated_hyps[batch_idx].add(final_tokens, final_score)
                # 经过上述步骤后，每个输入句子的类中保存着num_beams个最优序列
                # 下面选择若干最好的序列输出
                # 每个样本返回几个句子
            output_num_return_sequences_per_batch = 1
            output_batch_size = output_num_return_sequences_per_batch * batch_size
            # 记录每个返回句子的长度，用于后面pad
            sent_lengths = input_ids.new(output_batch_size)
            best = []

            # retrieve best hypotheses
            for i, hypotheses in enumerate(generated_hyps):
                # x: (score, hyp), x[0]: score
                sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
                for j in range(output_num_return_sequences_per_batch):
                    effective_batch_idx = output_num_return_sequences_per_batch * i + j
                    best_hyp = sorted_hyps.pop()[1]
                    sent_lengths[effective_batch_idx] = len(best_hyp)
                    best.append(best_hyp)
            if sent_lengths.min().item() != sent_lengths.max().item():
                sent_max_len = min(sent_lengths.max().item() + 1, maxlen)
                # fill pad
                decoded = input_ids.new(output_batch_size, sent_max_len).fill_(self.pad)
                # 填充内容
                for i, hypo in enumerate(best):
                    decoded[i, : sent_lengths[i]] = hypo
                    if sent_lengths[i] < maxlen:
                        decoded[i, sent_lengths[i]] = EOS_ID
            else:
                # 否则直接堆叠起来
                decoded = torch.stack(best).type(torch.long)
            return decoded.transpose(0, 1), None
                # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)

    def predict_nucelus(self, src, sbatch_turn_id, tbatch, tbatch_turn_id,
                     max_len=50, temp=None, k=None, p=0.95, greedy=None, m=0.5, loss=True):
        # src: [enc_out, batch]
        sbatch, sents_mapping, sent_rank = src
        with torch.no_grad():
            enc_out, enc_mask = self.encoder(sbatch, position=sent_rank,
                                              segment_ids=sents_mapping, turn_inp=sbatch_turn_id)
            ctx.memory_cache = tuple()
            ctx.memory_mask = None
            y_batch = torch.einsum('btl->tbl', tbatch)
            n_sents = y_batch.size(0)
            for sents_no, y_sents in enumerate(y_batch):
                y_inp = y_sents[:, :-1].contiguous()
                y_label = y_sents[:, 1:].contiguous()
                is_not_current_sents = sents_mapping.detach().gt(sents_no)
                current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents.long(),
                                                enc_mask.long()).bool()
                y_turn_id = tbatch_turn_id[:, sents_no]
                log_probs = self.decode_train(y_inp, y_turn_id, enc_out, current_sent_mask, log_probs=True)
                ctx.memory_mask = y_label.eq(PAD_ID).view(-1, y_label.size(-1)).transpose(0, 1)

            batch_size = sbatch.shape[0]
            context = torch.full((batch_size, 1), self.sos, dtype=torch.long)
            if torch.cuda.is_available():
                context = context.cuda()
            output = [
                    {
                    'ended': False,
                    'tokens': [],
                    'len': 0,
                    'nll4tok': [],
                    'ppl4tok': [],
                    'ppl': 0,
                    'nll': 0
                    }
                    for _ in range(batch_size)
                    ]
            ################## decoder state
            tgt_turn_id = tbatch_turn_id.new_zeros(batch_size)
            enc_out_T = enc_out.transpose(0, 1)
            is_not_current_sents = sents_mapping.detach().gt(n_sents)
            current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents.long(),
                                            enc_mask.long()).bool()
            enc_mask_T = current_sent_mask.transpose(0, 1)

            for i in range(max_len):
                # outputs: (batch_size, cur_len, vocab_size)
                input_ids = context
                dec_inp_T = input_ids.transpose(0, 1)
                dec_pred_T, _ = self.decoder(dec_inp_T, tgt_turn_id, enc_out_T, enc_mask_T, *ctx.memory_cache)
                dec_pred = dec_pred_T.transpose(0, 1).contiguous() ##[batch_size, length, vocab_size]
                # 取最后一个timestep的输出 (batch_size*num_beams, vocab_size)
                logits = dec_pred[:, -1, :]
                probs = self.generator(logits, log_probs=False)# log_softmax (batch_size*num_beams, vocab_size)
                logprobs = self.generator(logits, log_probs=True)
                if temp is not None:
                    samp_probs = F.softmax(logits.div_(temp), dim=-1)
                else:
                    samp_probs = probs.clone()
                if greedy:
                    next_probs, next_tokens = probs.topk(1)
                    next_logprobs = logprobs.gather(1, next_tokens.view(-1, 1))
                elif k is not None:
                    indices_to_remove = samp_probs < torch.topk(samp_probs, k)[0][..., -1, None]
                    samp_probs[indices_to_remove] = 0
                    if m is not None:
                        samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
                        samp_probs.mul_(1 - m)
                        samp_probs.add_(probs.mul(m))
                    next_tokens = samp_probs.multinomial(1)
                    next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
                elif p is not None:
                    sorted_probs, sorted_indices = torch.sort(samp_probs, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    sorted_indices_to_remove = cumulative_probs > p
                    sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                    sorted_indices_to_remove[:, 0] = 0
                    sorted_samp_probs = sorted_probs.clone()
                    sorted_samp_probs[sorted_indices_to_remove] = 0
                    if m is not None:
                        sorted_samp_probs.div_(sorted_samp_probs.sum(1).unsqueeze(1))
                        sorted_samp_probs.mul_(1 - m)
                        sorted_samp_probs.add_(sorted_probs.mul(m))
                    sorted_next_indices = sorted_samp_probs.multinomial(1).view(-1, 1)
                    next_tokens = sorted_indices.gather(1, sorted_next_indices)
                    next_logprobs = sorted_samp_probs.gather(1, sorted_next_indices).log()
                else:
                    if m is not None:
                        samp_probs.div_(samp_probs.sum(1).unsqueeze(1))
                        samp_probs.mul_(1 - m)
                        samp_probs.add_(probs.mul(m))
                    next_tokens = samp_probs.multinomial(1)
                    next_logprobs = samp_probs.gather(1, next_tokens.view(-1, 1)).log()
                next_cat = next_tokens
                next_tokens, next_logprobs = next_tokens.cpu(), next_logprobs.cpu()
                for b in range(batch_size):
                    out = output[b]
                    if out['ended']:
                        out['tokens'] += [PAD_ID] * (max_len - len(out['tokens']))
                        continue
                    v = next_tokens[b].item()
                    logprob = next_logprobs[b].item()
                    out['ended'] = v == EOS_ID
                    out['tokens'].append(v)
                    out['len'] += 1
                    out['nll4tok'].append(-logprob)
                    out['ppl4tok'].append(np.exp(-logprob))
                context = torch.cat([context, next_cat], dim=1)
            outputs = []
            nlls = []
            for b in range(batch_size):
                out = output[b]
                out['ppl'] = np.exp(sum(out['nll4tok']) / out['len'])
                out['nll'] = sum(out['nll4tok']) / out['len']
                outputs.append(out['tokens'])
                nlls.append(out['nll'])
            if loss:
                return torch.tensor(outputs).transpose(0, 1), sum(nlls)
            else:
                return torch.tensor(outputs).transpose(0, 1)

                # (output_batch_size, sent_max_len) ==> (batch_size, sent_max_len)
    def dynamaic_length_predict(self, src, sbatch_turn_id, tbatch, tbatch_turn_id, minlen=3, maxlen=50, tgt_w2idx=None, loss=False, norml_score_by_words=False):
        outputs = []
        scores = []
        bsz = src[0].shape[0]
        
        for i in range(minlen, maxlen):
            expect_len = i 
            # outputs [maxlen, bsz]
            # score [bsz]

            output, score = self.predict(src, sbatch_turn_id, tbatch, tbatch_turn_id, maxlen=maxlen, loss=True, expect_len=expect_len)
            length = score.new_zeros(bsz) + float('inf')
            for j in range(bsz):
                sent = output[:, j]
                for idx, id in enumerate(sent):
                    if id == EOS_ID:
                        length[j] = idx+1
                        if sent[idx-1] != tgt_w2idx['.'] and sent[idx-1] != tgt_w2idx['?'] and sent[idx-1] != tgt_w2idx['!']:
                            score[j] = float('-inf')
                        break
            if norml_score_by_words:
                score = score / length
            outputs.append(output)
            scores.append(score)

        outputs = torch.stack(outputs, dim=0) # [L, maxlen, bsz]
        print('outputs', outputs.size())
        scores = torch.stack(scores, dim=0) # [L, bsz]
        print('scores', scores.size())
        print('scores', scores[:, bsz//2])
        values, indices = scores.max(dim=0)
        print('indices', indices.size())
        print('indices', indices)
        assert bsz == indices.size(0)
        final_output = []
        for i in range(bsz):
            idx = indices[i]
            out_i = outputs[idx, :, i]
            print('idx', idx, 'out_i', out_i)
            final_output.append(out_i)
        final_output = torch.stack(final_output, dim=0).transpose(0, 1)

        return final_output, values


    def predict(self, src, sbatch_turn_id, tbatch, tbatch_turn_id, maxlen=50, loss=False, expect_len=None, attn_map=False):
        # src: [seq, batch]
        sbatch, sents_mapping, sent_rank = src
        if expect_len is not None:
            self.decoder.reset_length(expect_length=expect_len)
        with torch.no_grad():
            enc_out, enc_mask = self.encoder(sbatch, position=sent_rank,
                                              segment_ids=sents_mapping, turn_inp=sbatch_turn_id)
            ctx.memory_cache = tuple()
            ctx.memory_mask = None
            ctx.memory_pad_num = None

            y_batch = tbatch # torch.einsum('btl->tbl', tbatch)
            n_sents = len(y_batch)
            ctx.IS_INFERRING = False
            for sents_no, y_sents in enumerate(y_batch):
                y_inp = y_sents[:, :-1].contiguous()
                y_label = y_sents[:, 1:].contiguous()
                is_not_current_sents = sents_mapping.detach().ne(sents_no)
                current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents.long(),
                                                enc_mask.long()).bool()
                y_turn_id = tbatch_turn_id[:, sents_no]
                log_probs = self.decode_train(y_inp, y_turn_id, enc_out, current_sent_mask, log_probs=True)
                new_mask = y_label.eq(PAD_ID).view(-1, y_label.size(-1)).transpose(0, 1)
                ctx.memory_mask = new_mask if ctx.memory_mask is None else torch.cat([ctx.memory_mask, new_mask], dim=0)
                memory_pad_num = new_mask.new_zeros(new_mask.size()) + new_mask.sum(0)[None, :]
                ctx.memory_pad_num = memory_pad_num if ctx.memory_pad_num is None else torch.cat([ctx.memory_pad_num+new_mask.sum(0)[None, :], memory_pad_num], dim=0)
            
            ctx.IS_INFERRING = True
            batch_size = sbatch.shape[0]
            outputs = torch.zeros(maxlen, batch_size)
            floss = torch.zeros(maxlen, batch_size, self.output_size)
            score = torch.zeros(batch_size)
            if torch.cuda.is_available():
                outputs = outputs.cuda()
                floss = floss.cuda()
                score = score.cuda()

            if not ctx.ENABLE_CONTEXT:
                ctx.memory_cache = tuple()
                ctx.memory_mask = None
            input = sbatch.new_zeros(batch_size, 1).fill_(self.sos)
            stop_flag = torch.zeros(batch_size)
            #### Generate last utterence
            tgt_turn_id = tbatch_turn_id.new_zeros(batch_size).fill_(n_sents)
            enc_out_T = enc_out.transpose(0, 1)
            is_not_current_sents = sents_mapping.detach().ne(n_sents)
            current_sent_mask = torch.where(is_not_current_sents, is_not_current_sents, enc_mask)
            enc_mask_T = current_sent_mask.transpose(0, 1)
            attn_maps_list = []
            for t in range(1, maxlen):
                # tgt: [seq, batch, vocab_size]
                # this part is slow druing inference
                dec_inp_T = input.transpose(0, 1)
                if attn_map:
                    dec_pred_T, _, attn_maps= self.decoder(dec_inp_T, tgt_turn_id, enc_out_T, enc_mask_T, *ctx.memory_cache, attn_map=attn_map)
                    attn_maps_list.append(attn_maps)
                else:
                    dec_pred_T, _ = self.decoder(dec_inp_T, tgt_turn_id, enc_out_T, enc_mask_T, *ctx.memory_cache)
                dec_pred = dec_pred_T.transpose(0, 1).contiguous()
                # no_eos=dec_pred.size(1)<=3
                next_scores = self.generator(dec_pred[:, -1], log_probs=True, no_eos=dec_pred.size(1)<=2)
                floss[t] = next_scores
                next_token_score, next_token = next_scores.topk(1)
                outputs[t] = next_token.squeeze()
                input = torch.cat((input, next_token), dim=-1)

                assert len(next_token.squeeze(1)) == batch_size
                next_token_score = next_token_score.squeeze(1)
                for idx, i in enumerate(next_token.squeeze(1)):
                    if stop_flag[idx] == 0:
                        score[idx] += next_token_score[idx]
                    if i == EOS_ID:
                        score[idx] += next_token_score[idx]
                        stop_flag[idx] = 1
                if stop_flag.sum() == batch_size:
                    break
            outputs = outputs[1:]
            floss = floss[1:]
            self.finish_decoder()
            ctx.IS_INFERRING = False

        if loss:
            if attn_map:
                return outputs, score, attn_maps_list
            return outputs, score
        else:
            return outputs


from data_loader import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train script')
    parser.add_argument('--src_train', type=str, default=None, help='src train file')
    parser.add_argument('--tgt_train', type=str, default=None, help='src train file')
    parser.add_argument('--src_test', type=str, default=None, help='src test file')
    parser.add_argument('--tgt_test', type=str, default=None, help='tgt test file')
    parser.add_argument('--src_dev', type=str, default=None, help='src dev file')
    parser.add_argument('--tgt_dev', type=str, default=None, help='tgt dev file')
    parser.add_argument('--min_threshold', type=int, default=0,
                        help='epoch threshold for loading best model')
    parser.add_argument('--max_threshold', type=int, default=20,
                        help='epoch threshold for loading best model')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--model', type=str, default='HRED', help='model to be trained')
    parser.add_argument('--utter_hidden', type=int, default=150,
                        help='utterance encoder hidden size')
    parser.add_argument('--teach_force', type=float, default=0.5,
                        help='teach force ratio')
    parser.add_argument('--context_hidden', type=int, default=150,
                        help='context encoder hidden size')
    parser.add_argument('--decoder_hidden', type=int, default=150,
                        help='decoder hidden size')
    parser.add_argument('--seed', type=int, default=30,
                        help='random seed')
    parser.add_argument('--embed_size', type=int, default=200,
                        help='embedding layer size')
    parser.add_argument('--patience', type=int, default=5,
                        help='patience for early stop')
    parser.add_argument('--dataset', type=str, default='dailydialog',
                        help='dataset for training')
    parser.add_argument('--grad_clip', type=float, default=10.0, help='grad clip')
    parser.add_argument('--epochs', type=int, default=20, help='epochs for training')
    parser.add_argument('--src_vocab', type=str, default=None, help='src vocabulary')
    parser.add_argument('--tgt_vocab', type=str, default=None, help='tgt vocabulary')
    parser.add_argument('--maxlen', type=int, default=50,
                        help='the maxlen of the utterance')
    parser.add_argument('--tgt_maxlen', type=int, default=50,
                        help='the maxlen of the responses')
    parser.add_argument('--utter_n_layer', type=int, default=1,
                        help='layers of the utterance encoder')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--no-debug', dest='debug', action='store_false', default=False)
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout ratio')
    parser.add_argument('--hierarchical', type=int, default=1,
                        help='Whether hierarchical architecture')
    parser.add_argument('--transformer_decode', type=int, default=0,
                        help='transformer decoder need a different training process')
    parser.add_argument('--d_model', type=int, default=512,
                        help='d_model for transformer')
    parser.add_argument('--nhead', type=int, default=8,
                        help='head number for transformer')
    parser.add_argument('--num_encoder_layers', type=int, default=6)
    parser.add_argument('--num_decoder_layers', type=int, default=6)
    parser.add_argument('--num_turn_embeddings', type=int, default=50)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--warmup_step', type=int, default=4000, help='warm up steps')
    parser.add_argument('--contextrnn', dest='contextrnn', action='store_true')
    parser.add_argument('--no-contextrnn', dest='contextrnn', action='store_false', default=False)
    parser.add_argument('--position_embed_size', type=int, default=30)
    parser.add_argument('--graph', type=int, default=0)
    parser.add_argument('--train_graph', type=str, default=None,
                        help='train graph data path')
    parser.add_argument('--test_graph', type=str, default=None,
                        help='test graph data path')
    parser.add_argument('--dev_graph', type=str, default=None,
                        help='dev graph data path')
    parser.add_argument('--context_threshold', type=int, default=3,
                        help='low turns filter')
    parser.add_argument('--pred', type=str, default=None,
                        help='the file save the output')
    parser.add_argument('--dynamic_tfr', type=int, default=20,
                        help='begin to use the dynamic teacher forcing ratio, each ratio minus the tfr_weight')
    parser.add_argument('--dynamic_tfr_weight', type=float, default=0.05)
    parser.add_argument('--dynamic_tfr_counter', type=int, default=5)
    parser.add_argument('--dynamic_tfr_threshold', type=float, default=0.3)
    parser.add_argument('--bleu', type=str, default='nltk', help='nltk or perl')
    parser.add_argument('--lr_mini', type=float, default=1e-6, help='minial lr (threshold)')
    parser.add_argument('--lr_gamma', type=float, default=0.8, help='lr schedule gamma factor')
    parser.add_argument('--gat_heads', type=int, default=5, help='heads of GAT layer')
    parser.add_argument('--z_hidden', type=int, default=100, help='z_hidden for VHRED')
    parser.add_argument('--kl_annealing_iter', type=int, default=20000, help='KL annealing for VHRED')

    args = parser.parse_args()
    #################################################################

    dataset = 'DailyDialog'  # DailyDialog
    model = 'DHAED'

    if model == 'Transformer':
        transformer_decode = 1
    else:
        transformer_decode = 0
    hierarchical_list = ['HRED', 'HRAN', 'HRAN-ablation', 'VHRED', 'KgCVAE', 'WSeq', 'WSeq_RA', 'MReCoSa', \
                         'MReCoSa_RA', 'DSHRED', 'DSHRED_RA', 'MTGCN', 'MTGAT', 'GatedGCN', 'DHAED']
    non_hierarchical_list = ['Seq2Seq', 'Seq2Seq_MHA', 'Transformer']
    graph_list = ['MTGAT', 'MTGAT', 'GatedGCN']
    if model in hierarchical_list:
        hierarchical = 1
    elif model in non_hierarchical_list:
        hierarchical = 0
    else:
        print(f'not find model: {model}')
        hierarchical = 0
    if model in graph_list:
        graph = 1
    else:
        graph = 0

    if os.path.exists(f'./ckpt/{dataset}/{model}'):
        os.removedirs(f'./ckpt/{dataset}/{model}')
    os.makedirs(os.getcwd() + f'/ckpt/{dataset}/{model}')
    if os.path.exists(f"./processed/{dataset}/{model}/trainlog.txt"):
        os.remove(f'./processed/{dataset}/{model}/trainlog.txt')
    if os.path.exists(f"./processed/{dataset}/{model}/metadata.txt"):
        os.remove(f'./processed/{dataset}/{model}/metadata.txt')
    os.makedirs(os.getcwd() + f'/processed/{dataset}/{model}', exist_ok=True)
    if os.path.exists(f'./tblogs/{dataset}/{model}'):
        for i in os.listdir(f'./tblogs/{dataset}/{model}'):
            path_file = os.path.join(f'./tblogs/{dataset}/{model}', i)
            if os.path.isfile(path_file):
                os.remove(path_file)
    else:
        os.makedirs(os.getcwd() + f'/tblogs/{dataset}/{model}')

    if model == 'VHRED' or model == 'KgCVAE' or model == 'DHAED':
        src_vocab = f'../processed/{dataset}/vocab.pkl'
        tgt_vocab = f'../processed/{dataset}/vocab.pkl'
    else:
        src_vocab = f'../processed/{dataset}/iptvocab.pkl'
        tgt_vocab = f'../processed/{dataset}/optvocab.pkl'
    dropout = 0.3
    lr = 1e-4
    lr_mini = 1e-6
    if hierarchical == 1:
        maxlen = 50
        tgtmaxlen = 50
        batch_size = 2  # 128
    elif transformer_decode == 1:
        maxlen = 200
        tgtmaxlen = 25
        batch_size = 2  # 64
    else:
        maxlen = 150
        tgtmaxlen = 25
        batch_size = 2  # 64

    args.src_train = f'../data/{dataset}/src-train.txt'
    args.tgt_train = f'../data/{dataset}/tgt-train.txt'
    args.src_test = f'../data/{dataset}/src-test.txt'
    args.tgt_test = f'../data/{dataset}/tgt-test.txt'
    args.src_dev = f'../data/{dataset}/src-dev.txt'
    args.tgt_dev = f'../data/{dataset}/tgt-dev.txt'
    args.src_vocab = src_vocab
    args.tgt_vocab = tgt_vocab
    args.train_graph = f'../processed/{dataset}/train-graph.pkl'
    args.test_graph = f'../processed/{dataset}/test-graph.pkl'
    args.dev_graph = f'../processed/{dataset}/dev-graph.pkl'
    args.pred = f'../processed/{dataset}/{model}/pure-pred.txt'
    args.min_threshold = 0
    args.max_threshold = 100
    args.seed = 30
    args.epochs = 100
    args.lr = lr
    args.batch_size = batch_size
    args.model = model
    args.utter_n_layer = 2
    args.utter_hidden = 512
    args.teach_force = 1
    args.context_hidden = 512
    args.decoder_hidden = 512
    args.embed_size = 256
    args.patience = 10
    args.dataset = dataset
    args.grad_clip = 3.0
    args.dropout = dropout
    args.d_model = 512
    args.nhead = 8
    args.num_encoder_layers = 6
    args.num_decoder_layers = 8
    args.dim_feedforward = 1024
    args.hierarchical = hierarchical
    args.transformer_decode = transformer_decode
    args.graph = graph
    args.maxlen = maxlen
    args.tgt_maxlen = tgtmaxlen
    args.position_embed_size = 51
    args.context_threshold = 2
    args.dynamic_tfr = 15
    args.dynamic_tfr_weight = 0.0
    args.dynamic_tfr_counter = 10
    args.dynamic_tfr_threshold = 1.0
    args.bleu = nltk
    args.lr_mini = lr_mini
    args.lr_gamma = 0.5
    args.warmup_step = 4000
    args.gat_heads = 8

    ##################################################################
    # show the parameters and write into file
    print('[!] Parameters:')
    print(args)
    with open(f'./processed/{args.dataset}/{args.model}/metadata.txt', 'w') as f:
        print(vars(args), file=f)
    # set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # main function
    kwargs = vars(args)
    args_dict = vars(args)
    src_vocab, tgt_vocab = load_pickle(kwargs['src_vocab']), load_pickle(kwargs['tgt_vocab'])
    src_w2idx, src_idx2w = src_vocab
    tgt_w2idx, tgt_idx2w = tgt_vocab
    net = PHAED(len(src_w2idx), len(tgt_w2idx),
                padding_idx=tgt_w2idx['<pad>'], sos_idx=tgt_w2idx['<sos>'])

    train_iter = get_batch_data_hier_tf(kwargs['src_train'], kwargs['tgt_train'], kwargs['src_vocab'],
                                        kwargs['tgt_vocab'], kwargs['batch_size'], kwargs['maxlen'],
                                        kwargs['tgt_maxlen'], ld=True)
    for idx, batch in enumerate(train_iter):
        sbatch, tbatch, others = batch
        ctx.memory_mask = None
        ctx.memory_cache = tuple()
        output = net(sbatch, tbatch, others)

