import torch
import lightning as L
from torch import optim
import torch.utils
from transformers import (
    Conversation,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)


class LitLlamaChat(L.LightningModule):
    def __init__(
        self, llm: LlamaForCausalLM, tokenizer: LlamaTokenizerFast, max_new_tokens=10
    ):
        super().__init__()
        self.llm = llm
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens

    def training_step(self, batch, batch_idx):
        outputs = self.llm(**batch)
        loss = outputs.loss
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return optim.AdamW(self.parameters(), lr=5e-5)

    def validation_step(self, batch, batch_idx):
        outputs = self.llm(**batch)
        self.log('val_loss', outputs.loss)

    def generate(self, batch, *args, **kwargs):
        # argmax (greedy search) by default
        return self.llm.generate(batch, *args, **kwargs)

    def forward(self, batch):
        self.eval()
        with torch.no_grad():
            out = self.llm.forward(input_ids=torch.tensor(batch, dtype=torch.long))
        next_token_logits = out.logits[:, -1, :]  # B,1,V
        return next_token_logits

    def predict_step(self, batch):
        toks_out = self.generate(batch, max_new_tokens=self.max_new_tokens)
        out = self.tokenizer.batch_decode(toks_out)
        self.logger.experiment.log_dict(
            dictionary={'out': out},
            artifact_file='predictions.json',
            run_id=self.logger.run_id,
        )
        return out

    def conversations_to_tensors(self, conversations: list[Conversation]):
        tokens = [self.tokenizer.apply_chat_template(conv) for conv in conversations]
        return torch.tensor([tokens], dtype=torch.long)
