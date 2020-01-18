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

        self.e1e = nn.Sequential(
            nn.Embedding(config.vocab_size, config.n_embd),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )

        self.e2e = nn.Sequential(
            nn.Embedding(config.vocab_size, config.n_embd),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )

        self.e1p = nn.Sequential(
            nn.Embedding(config.n_positions, config.n_embd),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )

        self.e2p = nn.Sequential(
            nn.Embedding(config.n_positions, config.n_embd),
            nn.Linear(config.n_embd, config.n_embd, bias=False),
        )

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

    def forward(self, e1_ids, e2_ids, e1_mask, e2_mask, input_ids=None, past=None,
                attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        e1_shape = e1_ids.size()
        e2_shape = e2_ids.size()

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(past_length, input_shape[-1] + past_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        attention_mask = torch.cat((e1_mask, e2_mask, attention_mask), dim=-1)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1] + e1_shape[-1] + e2_shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0

        device = input_ids.device if input_ids is not None else inputs_embeds.device
        e1_ids = e1_ids.view(-1, e1_shape[-1])
        e2_ids = e2_ids.view(-1, e2_shape[-1])
        e1_embeds = self.e1e(e1_ids)
        e2_embeds = self.e2e(e2_ids)
        e1_pos_ids = torch.arange(0, e1_shape[-1], dtype=torch.long, device=device)
        e1_pos_ids = e1_pos_ids.unsqueeze(0).view(-1, e1_shape[-1])
        e2_pos_ids = torch.arange(0, e2_shape[-1], dtype=torch.long, device=device)
        e2_pos_ids = e2_pos_ids.unsqueeze(0).view(-1, e2_shape[-1])
        e1_pos_embeds = self.e1p(e1_pos_ids)
        e2_pos_embeds = self.e2p(e2_pos_ids)

        inputs_embeds = torch.cat((e1_embeds, e2_embeds, inputs_embeds), dim=-2)
        position_embeds = torch.cat((e1_pos_embeds, e2_pos_embeds, position_embeds), dim=-2)
        # print(e1_mask.shape, e2_mask.shape, attention_mask.shape)

        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        input_shape = inputs_embeds.size()[:-1]

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states, layer_past=layer_past, attention_mask=attention_mask, head_mask=head_mask[i]
            )

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
                past=None, attention_mask=None, token_type_ids=None, position_ids=None,
                head_mask=None, inputs_embeds=None, labels=None):
        transformer_outputs = self.transformer(e1_ids=e1_ids, e2_ids=e2_ids, e1_mask=e1_mask, e2_mask=e2_mask,
                                               past=past, input_ids=input_ids, attention_mask=attention_mask,
                                               token_type_ids=token_type_ids, position_ids=position_ids,
                                               head_mask=head_mask, inputs_embeds=inputs_embeds)
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
