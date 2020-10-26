from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import GPT2PreTrainedModel

from .gpt2_modified import GPT2REModel


class GPT2REClsModel(GPT2PreTrainedModel):

    def __init__(self, config):
        super(GPT2REClsModel, self).__init__(config)
        self.transformer = GPT2REModel(config)
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
                                               position_ids=position_ids, token_type_ids=token_type_ids, past=past)
        hidden_states = transformer_outputs[0]
        hid = hidden_states.mean(axis=1)
        # print(self.lm_head)
        cls_logits = self.cls_head(hid).softmax(dim=1)

        outputs = (cls_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            # shift_logits = cls_logits[..., :-1, :].contiguous()
            # shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=ignore_index)
            slo = cls_logits.contiguous()
            sla = labels.contiguous()
            # for_print = {
            #     'lab': (labels, labels.shape),
            #     'log': (cls_logits, cls_logits.shape),
            #     'hid': (hid, hid.shape)
            # }
            # print(for_print)
            # loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss_fct(slo.view(-1, slo.size(-1)), sla.view(-1))
            outputs = (loss,) + outputs
            # print(loss, loss.dtype, loss.device)

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)
