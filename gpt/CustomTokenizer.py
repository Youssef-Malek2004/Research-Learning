import string
from collections import Counter

class CustomBPETokenizer:
    def __init__(self):
        self.vocab_size = 0
        self.vocab = {}
        self.translator = str.maketrans("", "", string.punctuation)

    def clean_data(self, batch):
        batch["text"] = [
            t.lower()
             .translate(self.translator)
             .replace("\n", "")
            for t in batch["text"]
        ]
        return batch

    def build_vocab_from_dataset(self, ds, split="train"):
        counter = Counter()

        for text in ds[split]["text"]:
            counter.update(text)

        self.vocab = {ch: i for i, ch in enumerate(counter.keys())}
        self.vocab_size = len(self.vocab)


def main():
    from datasets import load_dataset

    ds = load_dataset("roneneldan/TinyStories")

    tokenizer = CustomBPETokenizer()

    ds = ds.map(
        tokenizer.clean_data,
        batched=True,
        num_proc=4,
        batch_size=1000,
    )

    tokenizer.build_vocab_from_dataset(ds, split="train")

    print("Vocab size:", tokenizer.vocab_size)
    print(list(tokenizer.vocab.items())[:20])


if __name__ == "__main__":
    main()
