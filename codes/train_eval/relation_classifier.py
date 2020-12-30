from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2PreTrainedModel


class GPT2REClsModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2REClsModel, self).__init__(config)
        self.transformer = GPT2Model(config)
        self.cls_head = nn.Linear(config.n_embd, config.n_classes, bias=False)
        # print(self.lm_head)
        self.init_weights()
        # print(1, self.cls_head)

    def get_output_embeddings(self):
        return None

    def forward(self, input_ids, attention_mask=None, position_ids=None, token_type_ids=None, labels=None, past=None):
        from global_constants import ignore_index
        # print(2, self.cls_head)
        transformer_outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask,
                                               position_ids=position_ids, token_type_ids=token_type_ids,
                                               past_key_values=past)
        hidden_states = transformer_outputs[0]
        # print(hidden_states.shape)
        hid = hidden_states.mean(axis=1)
        # print(self.lm_head)
        cls_logits = self.cls_head(hid)

        outputs = (cls_logits,) + transformer_outputs[1:]
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            slo = cls_logits.contiguous()
            sla = labels.contiguous()
            loss = loss_fct(slo.view(-1, slo.size(-1)), sla.view(-1))
            outputs = (loss,) + outputs
            # print(loss, loss.dtype, loss.device)

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
