from typing import Dict, Any, Tuple, Sequence

import torch
import transformers as tfm
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.utils.checkpoint import checkpoint
from transformers.file_utils import (
    add_start_docstrings,
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithPastAndCrossAttentions,
    ModelOutput
)
from transformers.models.gpt2.modeling_gpt2 import Block, GPT2PreTrainedModel
from transformers.utils.model_parallel_utils import get_device_map, assert_device_map

from libs import get_between, pad_tensor, loggers, log_info, del_key
from .doc_strings import *

tok = tfm.GPT2Tokenizer.from_pretrained('gpt2')


def get_device(*tensors):
    for i, tensor in enumerate(tensors):
        try:
            if isinstance(tensor, Sequence):
                tensor = tensor[0]
                yield [x.device for x in tensor]
            else:
                yield 'None' if tensor is None else tensor.device
        except:
            print(i)
            # exit()


@add_start_docstrings(
    "The bare GPT2 REModel transformer outputting raw hidden-states without any specific head on top.",
    GPT2_START_DOCSTRING
)
class GPT2REModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2REModel, self).__init__(config)
        # self.output_hidden_states = config.output_hidden_states
        # self.output_attentions = config.output_attentions
        # self.output_past = config.output_past
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h: nn.ModuleList = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.ent = nn.Linear(config.n_embd, config.n_embd)
        self.pos = nn.Linear(config.n_embd, config.n_embd)

        self.device_map = None
        self.model_parallel = False
        self.first_device = None
        self.last_device = None

        self.between = hasattr(config, 'between') and config.between
        if not (hasattr(config, 'random') and config.random):
            self.init_weights()

    def parallelize(self, device_map=None):
        self.h: nn.ModuleList
        # Check validity of device_map
        self.device_map = (
            get_device_map(len(self.h), range(torch.cuda.device_count())) if device_map is None else device_map
        )
        assert_device_map(self.device_map, len(self.h))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_map.keys() else "cuda:" + str(min(self.device_map.keys()))
        self.last_device = "cuda:" + str(max(self.device_map.keys()))
        self.wte = self.wte.to(self.first_device)
        self.wpe = self.wpe.to(self.first_device)
        # Load onto devices
        for k, v in self.device_map.items():
            for block in v:
                cuda_device = "cuda:" + str(k)
                self.h[block] = self.h[block].to(cuda_device)
        # ln_f to last
        self.ln_f = self.ln_f.to(self.last_device)

    def deparallelize(self):
        self.h: nn.ModuleList
        self.model_parallel = False
        self.device_map = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        self.wte = self.wte.to("cpu")
        self.wpe = self.wpe.to("cpu")
        for index in range(len(self.h)):
            self.h[index] = self.h[index].to("cpu")
        self.ln_f = self.ln_f.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        self.h: nn.ModuleList
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(GPT2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        tokenizer_class=TOKENIZER_FOR_DOC,
        checkpoint="gpt2",
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=CONFIG_FOR_DOC,
    )
    def forward(self,
                input_ids,
                attention_mask,
                position_ids,
                token_type_ids,
                past=None,
                output_hidden_states=None,
                output_attentions=None,
                use_cache=None,
                return_dict=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None):

        self.h: nn.ModuleList
        from global_constants import DEBUG, main_device
        config = self.config
        output_hidden_states = output_hidden_states or config.output_hidden_states
        output_attentions = output_attentions or config.output_attentions
        use_cache = use_cache or config.use_cache
        return_dict = return_dict or config.return_dict

        device = main_device

        inputs_embeds = torch.zeros(*input_ids.shape, self.wte.embedding_dim, dtype=self.wte.weight.dtype,
                                    device=device)
        position_embeds = torch.zeros(*inputs_embeds.shape, dtype=self.wpe.weight.dtype, device=device)
        position_ids = position_ids.to(self.device)
        input_ids = input_ids.to(self.device)
        token_type_ids = token_type_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        if past is not None:
            past = (x.to(self.device) for x in past)
        for i in range(input_ids.shape[0]):
            inp = input_ids[i]
            pos = position_ids[i]
            type_id = token_type_ids[i]
            # print(type_id, inp)
            e1, e2 = inp[type_id == 1], inp[type_id == 2]
            ep1, ep2 = pos[type_id == 1], pos[type_id == 2]
            e1e, e2e = self.wte(e1), self.wte(e2)
            e1p, e2p = self.wpe(ep1), self.wpe(ep2)
            embd = self.ent(torch.cat((e1e, e2e)))
            pos_embd = self.pos(torch.cat((e1p, e2p)))
            if 0 in type_id:
                s = inp[type_id == 0]
                sp = pos[type_id == 0]
                sl = s.shape[0]
                if self.between:
                    a, b = get_between(e1, e2, s, inclusive=True)
                    se = self.wte(s)[a:b]
                    attention_mask[i][len(e1) + len(e2) + (b - a):] = 0
                    se = pad_tensor(se, sl, 0)
                else:
                    se = self.wte(s)
                embd = torch.cat((embd, se))
                pos_embd = torch.cat((pos_embd, self.wpe(sp)))
            inputs_embeds[i] = embd
            position_embeds[i] = pos_embd

        if DEBUG:
            rstr = ''
            for i in range(len(input_ids)):
                inps = input_ids[i][input_ids[i] != 50256]
                # print(attention_mask)
                atts = attention_mask[i][attention_mask[i] == 1.0]
                rstr += 'inp: {}, {}\n'.format(len(inps), inps)
                rstr += 'att: {}, {}\n'.format(len(atts), atts)
                rstr += 'pos: {}\n'.format(position_ids[i])
                rstr += 'tok: {}\n'.format(token_type_ids[i])
                rstr += 'sent: {}'.format(tok.decode(inps))
            log_info(loggers.sample_logger, rstr)

        token_type_ids[token_type_ids == 2] = 1
        token_type_embeds = self.wte(token_type_ids)

        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        if attention_mask is not None:
            attention_mask = attention_mask.view(input_ids.shape[0], -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)
            attention_mask: torch.Tensor = (attention_mask - 1.0) * 10000.0
        # print(inputs_embeds.shape, position_embeds.shape, token_type_embeds.shape)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)
        head_mask = self.get_head_mask(None, self.config.n_layer)

        # input_shape = inputs_embeds.size()[:-1]
        input_shape = input_ids.size()
        output_shape = input_shape + (hidden_states.size(-1),)

        if past is None:
            past = [None] * len(self.h)

        presents = () if use_cache else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, (block, layer_past) in enumerate(zip(self.h, past)):

            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)

                if layer_past is not None:
                    layer_past = layer_past.to(hidden_states.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)

            if output_hidden_states:
                # all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)
                all_hidden_states = all_hidden_states + (hidden_states,)

            if getattr(config, "gradient_checkpointing", False):
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # checkpointing only works with tuple returns, not with lists
                        return tuple(output for output in module(*inputs, use_cache, output_attentions))

                    return custom_forward

                outputs = checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    layer_past,
                    attention_mask,
                    head_mask[i],
                    encoder_hidden_states,
                    encoder_attention_mask,
                )
            else:
                # print('{}, {}, {}, {}, {}, {}'.format(hidden_states.shape, attention_mask.shape, head_mask[i],
                #                                       encoder_hidden_states,
                #                                       encoder_attention_mask,
                #                                       layer_past.shape if layer_past is not None else None))
                outputs = block(
                    hidden_states,
                    layer_past=layer_past,
                    attention_mask=attention_mask,
                    head_mask=head_mask[i],
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    use_cache=use_cache,
                    output_attentions=output_attentions
                )

            hidden_states, present = outputs[:2]

            if use_cache:
                presents = presents + (present,)
            # print(present.shape)

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[2],)
                if config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[3],)

            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return (v for v in [hidden_states, presents, all_hidden_states, all_self_attentions] if v is not None)

        # temp = list(get_device(hidden_states, presents, all_hidden_states, all_self_attentions, all_cross_attentions))
        # temp = [input_ids.size(1)] + temp
        # print(len(temp))
        # print('last: {}, {}, {}, {}, {}, {}'.format(*temp))
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=presents,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions
        )


class GPT2LMREModel(GPT2PreTrainedModel):

    def __init__(self, config):
        # print(config)
        super(GPT2LMREModel, self).__init__(config)
        self.transformer = GPT2REModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.device_map = None
        self.model_parallel = False

        if not (hasattr(config, 'random') and config.random):
            self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.transformer.h), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.transformer.h))
        self.transformer.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.transformer.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.transformer.deparallelize()
        self.transformer = self.transformer.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        torch.cuda.empty_cache()

    def forward(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None, labels=None, past=None,
                return_dict=None):
        from global_constants import ignore_index

        transformer_outputs: BaseModelOutputWithPastAndCrossAttentions = \
            self.transformer(input_ids=input_ids,
                             attention_mask=attention_mask,
                             position_ids=position_ids,
                             token_type_ids=token_type_ids,
                             past=past,
                             return_dict=return_dict)
        hidden_states = transformer_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            outputs = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + outputs) if loss is not None else outputs
            # (loss), lm_logits, presents, (all hidden_states), (attentions)

        # if past is not None:
        #     lm_logits = lm_logits.cpu()

        return CausalLMOutputWithPastAndCrossAttentions(
            loss=loss,
            logits=lm_logits.cpu(),
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
            past_key_values=transformer_outputs.past_key_values
        )

    @classmethod
    def _update_model_kwargs_for_generation(cls,
                                            outputs: ModelOutput,
                                            model_kwargs: Dict[str, Any],
                                            is_encoder_decoder: bool = False
                                            ) -> Dict[str, Any]:
        model_kwargs['past'] = getattr(
            outputs, 'past_key_values',
            getattr(
                outputs, 'mems',
                getattr(
                    outputs, 'past_buckets_states', None
                )
            )
        )

        use_cache = model_kwargs['use_cache']

        if 'position_ids' in model_kwargs:
            position_ids: torch.Tensor = model_kwargs['position_ids']
            plus = 0 if 'first' in model_kwargs else position_ids[:, None, -1] + 1
            new_pos = torch.zeros(position_ids.size(0), 1, dtype=position_ids.dtype) + plus
            if use_cache:
                model_kwargs['position_ids'] = new_pos
            else:
                model_kwargs['position_ids'] = torch.cat([position_ids, new_pos], dim=-1)

        # update token_type_ids with last value
        if "token_type_ids" in model_kwargs:
            token_type_ids: torch.Tensor = model_kwargs["token_type_ids"]
            new_token = torch.zeros(token_type_ids.size(0), 1, dtype=token_type_ids.dtype)
            if use_cache:
                model_kwargs['token_type_ids'] = new_token
            else:
                model_kwargs["token_type_ids"] = torch.cat([token_type_ids, new_token], dim=-1)

        # if 'input_ids' in model_kwargs:
        #     input_ids: torch.Tensor = model_kwargs['input_ids']
        #     if use_cache:
        #         model_kwargs['input_ids'] = input_ids[:, None, -1]

        # update attention mask
        if not is_encoder_decoder:
            if "attention_mask" in model_kwargs:
                attention_mask = model_kwargs["attention_mask"]
                new_att = attention_mask.new_ones((attention_mask.shape[0], 1))
                if use_cache:
                    model_kwargs["attention_mask"] = new_att
                else:
                    model_kwargs["attention_mask"] = torch.cat(
                        [attention_mask, new_att], dim=-1
                    )
        del_key(model_kwargs, 'first')
        # print('po:{}, to:{}, at:{}'.format(
        #     model_kwargs['position_ids'].shape,
        #     model_kwargs['token_type_ids'].shape,
        #     model_kwargs['attention_mask'].shape
        # ))
        # print(list(model_kwargs.keys()))
        # print('np:{}, nt:{}, na:{}'.format(
        #     new_pos.shape,
        #     new_token.shape,
        #     new_att.shape
        # ))
        return model_kwargs

    @staticmethod
    def _expand_inputs_for_generation(
            input_ids: torch.LongTensor,
            expand_size: int = 1,
            is_encoder_decoder: bool = False,
            attention_mask: torch.LongTensor = None,
            encoder_outputs=None,
            **model_kwargs
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        expanded_return_idx = (
            torch.arange(input_ids.shape[0]).view(-1, 1).repeat(1, expand_size).view(-1).to(input_ids.device)
        )
        input_ids: torch.LongTensor = input_ids.index_select(0, expanded_return_idx)

        # print('expanding...')
        if "token_type_ids" in model_kwargs:
            token_type_ids = model_kwargs["token_type_ids"]
            model_kwargs["token_type_ids"] = token_type_ids.index_select(0, expanded_return_idx)

        if "position_ids" in model_kwargs:
            position_ids = model_kwargs["position_ids"]
            model_kwargs["position_ids"] = position_ids.index_select(0, expanded_return_idx)

        if attention_mask is not None:
            model_kwargs["attention_mask"] = attention_mask.index_select(0, expanded_return_idx)

        if is_encoder_decoder:
            assert encoder_outputs is not None
            encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
                0, expanded_return_idx
            )
            model_kwargs["encoder_outputs"] = encoder_outputs
        # print('model{}'.format(model_kwargs))
        return input_ids, model_kwargs

    def prepare_inputs_for_generation(self, input_ids: torch.LongTensor, **kwargs) -> Dict[str, Any]:
        device = self.device

        def to_device(dc, par):
            res = dc.get(par)
            return res if res is None else res

        use_cache = kwargs['use_cache'] and 'first' not in kwargs
        if use_cache:
            input_ids = input_ids[:, None, -1]

        return {'input_ids': input_ids.to(device),
                'token_type_ids': kwargs['token_type_ids'],
                'position_ids': kwargs['position_ids'],
                'attention_mask': kwargs['attention_mask'],
                'past': to_device(kwargs, 'past'),
                'labels': to_device(kwargs, 'labels')
                }
