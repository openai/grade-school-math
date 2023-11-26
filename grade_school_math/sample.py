import torch as th
from dataset import get_examples, GSMDataset
from calculator import sample
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import logging

def main():
    try:
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger(__name__)

        device = th.device("cuda")
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained("model_ckpts")
        model.to(device)
        logger.info("Model Loaded")

        test_examples = get_examples("test")
        qn = test_examples[1]["question"]
        sample_len = 100
        logger.info(qn.strip())
        logger.info(sample(model, qn, tokenizer, device, sample_len))

    except Exception as e:
        logging.exception("An error occurred: %s", str(e))

if __name__ == "__main__":
    main()
