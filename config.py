from model import GPTConfig  # Adjust 'model' to the correct module name where GPTConfig is defined

config = GPTConfig(
    vocab_size=50257,     # use the tokenizer's vocab size
    block_size=128,       # or whatever context size you're training with
    n_layer=6,            # no. of transformers
    n_head=6,         # no. of attention heads 
    n_embd=384,
    dropout=0.1,
    bias=True
)
