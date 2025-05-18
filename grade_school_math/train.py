import torch as th
from dataset import get_examples, GSMDataset
from transformers import (
    GPT2Tokenizer, GPT2LMHeadModel, GPT2Config,
    AdamW, get_scheduler
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

# === CLASS 1: Tokenizer Loader ===
class TokenizerLoader:
    @staticmethod
    def load(name="gpt2"):
        return GPT2Tokenizer.from_pretrained(name)

# === CLASS 2: Dataset + Dataloader ===
class DataLoaderBuilder:
    def __init__(self, tokenizer, batch_size=16):
        examples = get_examples("train")
        dataset = GSMDataset(tokenizer, examples)
        self.loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def get_loader(self):
        return self.loader

# === CLASS 3: Model + Optimizer ===
class ModelBuilder:
    def __init__(self, model_name="gpt2", lr=1e-5, total_steps=1000):
        config = GPT2Config.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name, config=config)
        self.device = th.device("cuda" if th.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.train()
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.scheduler = get_scheduler(
            "linear", optimizer=self.optimizer,
            num_warmup_steps=0, num_training_steps=total_steps
        )

# === CLASS 4: Trainer ===
class Trainer:
    def __init__(self, model, optimizer, scheduler, device):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train(self, dataloader, num_epochs):
        pbar = tqdm(range(num_epochs * len(dataloader)))
        for epoch in range(num_epochs):
            for batch in dataloader:
                self.optimizer.zero_grad()
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch, labels=batch["input_ids"])
                loss = outputs[0]
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                pbar.update(1)
                pbar.set_description(f"train_loss: {loss.item():.5f}")

# === MAIN FUNCTION ===
def main():
    tokenizer = TokenizerLoader.load("gpt2")
    data_builder = DataLoaderBuilder(tokenizer)
    train_loader = data_builder.get_loader()

    total_steps = len(train_loader) * 20
    model_builder = ModelBuilder("gpt2", lr=1e-5, total_steps=total_steps)

    trainer = Trainer(
        model_builder.model,
        model_builder.optimizer,
        model_builder.scheduler,
        model_builder.device
    )
    trainer.train(train_loader, num_epochs=20)

    model_builder.model.save_pretrained("model_ckpts/")

if __name__ == "__main__":
    main()
