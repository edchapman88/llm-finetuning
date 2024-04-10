import lightning as L
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Conversation,
    LlamaForCausalLM,
    LlamaTokenizerFast,
)
import torch
from torch import optim
import torch.utils
from lightning.pytorch.loggers import MLFlowLogger
from azureml.core.run import Run


HF_TOKEN = ''
MODEL_ID = 'meta-llama/Llama-2-7b-chat-hf'


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


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
    model = LitLlamaChat(
        llm=AutoModelForCausalLM.from_pretrained(
            MODEL_ID, token=HF_TOKEN, cache_dir='model_cache'
        ),
        tokenizer=tokenizer,
    )

    batch = [
        Conversation(
            [
                {'role': 'system', 'content': 'Answer the following questions:'},
                {'role': 'user', 'content': 'What is the capital of England?'},
            ]
        ),
        Conversation(
            [
                {'role': 'system', 'content': 'Answer the following questions:'},
                {'role': 'user', 'content': 'What is the capital of France?'},
            ]
        ),
    ]

    # MLFlow Logger
    run = Run.get_context()
    mlflow_url = run.experiment.workspace.get_mlflow_tracking_uri()
    mlf_logger = MLFlowLogger(
        experiment_name=run.experiment.name, tracking_uri=mlflow_url
    )
    mlf_logger._run_id = run.id

    trainer = L.Trainer(logger=mlf_logger, devices=2)
    trainer.predict(model=model, dataloaders=model.conversations_to_tensors(batch))

    torch.save(model, './outputs/model.pt')


if __name__ == '__main__':
    main()
