import numpy as np

class CharTokenizer:
    def __init__(self, corpus: str):
        chars = sorted(list(set(corpus)))
        self.vocab_size = len(chars)

        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list:
        return [self.char_to_idx[ch] for ch in text]
    
    def decode(self, tokens: list) -> str:
        """ Converts a list of integers to text """
        if isinstance(tokens, int) or isinstance(tokens, np.integer):
            tokens = [tokens]
        return "".join([self.idx_to_char[i] for i in tokens])
    
import tiktoken
class FilteredBPETokenizer:
    def __init__(self, corpus: str, encoding_name = "gpt2"):
        """
        BPE Tokenizer based on gpt2's vocabulary though only filtered to use the tokens that appear in the corpus
        Uses tiktotken. This is my first implementation which is not from scratch (except for numpy ofc).
        This is because using a BPE tokenizer should be a huge leap in performance, and implementing an optimized
        version of the algorithm would take too long
        """

        self.base_tokenizer = tiktoken.get_encoding(encoding_name)

        full_encoded = self.base_tokenizer.encode(corpus)

        unique_tokens = sorted(list(set(full_encoded)))
        self.vocab_size = len(unique_tokens)

        print(f"Original GPT-2 vocab size : {self.base_tokenizer.n_vocab} versus used vocab size : {self.vocab_size}")

        # Encoding
        self.old_to_new = {old_id: new_id for new_id, old_id in enumerate(unique_tokens)}

        # Decoding
        self.new_to_old = {new_id: old_id for new_id, old_id in enumerate(unique_tokens)}

    def encode(self, text: str) -> list:
        raw_tokens = self.base_tokenizer.encode(text)
        # Tokens outside of vocab are ignored without raising anything
        safe_tokens = [self.old_to_new[t] for t in raw_tokens if t in self.old_to_new]
        return safe_tokens
    
    def decode(self, tokens: list) -> list:
        raw_tokens = [self.new_to_old[t] for t in tokens]
        return self.base_tokenizer.decode(raw_tokens)
