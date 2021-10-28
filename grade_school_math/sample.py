import torch as th
from dataset import get_examples, GSMDataset
from calculator import sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def main():
    device = th.device("cuda")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("model_ckpts")
    model.to(device)
    print("Model Loaded")

    test_examples = get_examples("test")
    qn = test_examples[1]["question"]
    sample_len = 100
    print(qn.strip())
    print(sample(model, qn, tokenizer, device, sample_len))


if __name__ == "__main__":
    main()
