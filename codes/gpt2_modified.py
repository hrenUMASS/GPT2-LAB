import math

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel
from transformers.modeling_gpt2 import Block


# import transformers as tfm
class GELU(nn.Module):

    def __init__(self, inplace=False):
        super(GELU, self).__init__()
        self.inplace = inplace

    def forward(self, x: torch.Tensor):
        data = 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        if self.inplace:
            x.data = data.data
            return x
        return data

    def extra_repr(self):
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class GPT2REModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2REModel, self).__init__(config)
        self.output_hidden_states = config.output_hidden_states
        self.output_attentions = config.output_attentions
        self.output_past = config.output_past

        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([Block(config.n_ctx, config, scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

        self.ent = nn.Linear(config.n_embd, config.n_embd)
        self.pos = nn.Linear(config.n_embd, config.n_embd)

        self.init_weights()

    def get_input_embeddings(self):
        return self.wte

    def set_input_embeddings(self, new_embeddings):
        self.wte = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(self, e1_ids, e2_ids, e1_mask, e2_mask, input_ids=None,
                attn_mask=None):

        device = e1_ids.device
        e1_shape = e1_ids.size()
        e2_shape = e2_ids.size()
        input_shape = torch.Size([e1_shape[0], 0])
        # print(e1_shape, e2_shape, input_ids.shape)
        e1_type = torch.zeros(e1_shape, dtype=torch.long, device=device)
        e2_type = torch.ones(e2_shape, dtype=torch.long, device=device)
        token_type_ids = torch.cat((e1_type, e2_type), dim=-1)

        if input_ids is not None:
            input_shape = input_ids.size()
            input_type = 2 * torch.ones(input_shape, dtype=torch.long, device=device)
            token_type_ids = torch.cat((token_type_ids, input_type), dim=-1)

        token_type_ids = token_type_ids.view(-1, input_shape[-1] + e1_shape[-1] + e2_shape[-1])

        if input_ids is not None:
            input_ids = input_ids.view(-1, input_shape[-1])

        e1_ids = e1_ids.view(-1, e1_shape[-1])
        e2_ids = e2_ids.view(-1, e2_shape[-1])
        e1_embeds = self.wte(e1_ids)
        e2_embeds = self.wte(e2_ids)

        e1_pos_ids = torch.arange(0, e1_shape[-1], dtype=torch.long, device=device).unsqueeze(0).view(-1, e1_shape[-1])
        e2_pos_ids = torch.arange(0, e2_shape[-1], dtype=torch.long, device=device).unsqueeze(0).view(-1, e2_shape[-1])
        e1_pos_embeds = self.wpe(e1_pos_ids)
        e2_pos_embeds = self.wpe(e2_pos_ids)

        inputs_embeds = torch.cat((e1_embeds, e2_embeds), dim=-2)
        inputs_embeds = self.ent(inputs_embeds)
        position_embeds = torch.cat((e1_pos_embeds, e2_pos_embeds), dim=-2)
        position_embeds = self.pos(position_embeds)
        attention_mask = torch.cat((e1_mask, e2_mask), dim=-1)
        token_type_embeds = self.wte(token_type_ids)
        # token_type_embeds = 0
        if input_ids is not None:
            position_ids = torch.arange(0, input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
            pos_embeds = self.wpe(position_ids)
            inputs_embeds = torch.cat((inputs_embeds, self.wte(input_ids)), dim=-2)
            position_embeds = torch.cat((position_embeds, pos_embeds), dim=-2)
            attention_mask = torch.cat((attention_mask, attn_mask), dim=-1)

        attention_mask = attention_mask.view(-1, input_shape[-1] + e1_shape[-1] + e2_shape[-1])
        attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (attention_mask - 1.0) * 10000.0

        # print(e1_mask.shape, e2_mask.shape, attention_mask.shape)

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        input_shape = inputs_embeds.size()[:-1]

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, attention_mask=attention_mask)

            hidden_states, present = outputs[:2]
            if self.output_past:
                presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_past:
            outputs = outputs + (presents,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (presents), (all hidden_states), (attentions)


class GPT2LMREModel(GPT2PreTrainedModel):

    def __init__(self, config):
        # print(config)
        super(GPT2LMREModel, self).__init__(config)
        self.transformer = GPT2REModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        # only last token for inputs_ids if past is defined in kwargs
        if "past" in kwargs and kwargs["past"]:
            input_ids = input_ids[:, -1].unsqueeze(-1)

        inputs = {"input_ids": input_ids}
        inputs.update(kwargs)
        return inputs

    def forward(self, e1_ids, e2_ids, e1_mask, e2_mask, e1_labels=None, e2_labels=None, input_ids=None,
                attention_mask=None, labels=None):
        transformer_outputs = self.transformer(e1_ids=e1_ids, e2_ids=e2_ids, e1_mask=e1_mask, e2_mask=e2_mask,
                                               input_ids=input_ids, attn_mask=attention_mask)
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None and e1_labels is not None and e2_labels is not None:
            # Shift so that tokens < n predict n
            # print(e1_labels.shape, e2_labels.shape, labels.shape)
            labels = torch.cat((e1_labels, e2_labels, labels), dim=-1)
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

# a = nn.Embedding(20, 5)
# b = torch.randint(0, 20, (10, 20))
# c = torch.randint(0, 20, (10, 25))
# b, c = a(b), a(c)
# print(torch.cat((b, c), dim=-2))
# e1l, e2l, sent_len = 2, 3, 10
# e1p, e2p, pos = torch.arange(0, e1l, dtype=torch.long), \
#                 torch.arange(0, e2l, dtype=torch.long), \
#                 torch.arange(0, sent_len, dtype=torch.long)
#
# print(e1p, '\n', e2p, '\n', pos)
# temp = torch.cat((e1p, e2p, pos))
# print(temp)
# print(temp.unsqueeze(0).view(-1, sent_len + e1l + e2l))
