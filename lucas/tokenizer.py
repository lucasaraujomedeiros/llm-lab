from collections import Counter, deque
from functools import lru_cache
import re
import json


class BPETokenizerSimple:
    def __init__(self):
        # Maps token_id to token_str (e.g., {11246: "some"})
        self.vocab = {}
        # Maps token_str to token_id (e.g., {"some": 11246})
        self.inverse_vocab = {}
        # Dictionary of BPE merges: {(token_id1, token_id2): merged_token_id}
        self.bpe_merges = {}

        # For the official OpenAI GPT-2 merges, use a rank dict:
        #  of form {(string_A, string_B): rank}, where lower rank = higher priority
        self.bpe_ranks = {}

    def train(self, text, vocab_size, allowed_special={"<|endoftext|>"}):
        """
        Train the BPE tokenizer from scratch.

        Args:
            text (str): The training text.
            vocab_size (int): The desired vocabulary size.
            allowed_special (set): A set of special tokens to include.
        """

        # Pre-tokenize training text using the same boundary rules as encode()
        tokens = self.pretokenize_text(text)

        # Initialize vocab with unique characters, including "Ġ" if present
        # Start with the first 256 ASCII characters
        unique_chars = [chr(i) for i in range(256)]
        unique_chars.extend(
            char for char in sorted({char for token in tokens for char in token})
            if char not in unique_chars
        )
        if "Ġ" not in unique_chars:
            unique_chars.append("Ġ")

        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}

        # Add allowed special tokens
        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id

        # Tokenize each pre-token into character IDs
        token_id_sequences = [
            [self.inverse_vocab[char] for char in token]
            for token in tokens
        ]

        # BPE steps 1-3: Repeatedly find and replace frequent pairs
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_id_sequences, mode="most")
            if pair_id is None:
                break
            token_id_sequences = self.replace_pair(token_id_sequences, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id

        # Build the vocabulary with merged tokens
        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id

    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        """
        Load pre-trained vocabulary and BPE merges from OpenAI's GPT-2 files.

        Args:
            vocab_path (str): Path to the vocab file (GPT-2 calls it 'encoder.json').
            bpe_merges_path (str): Path to the bpe_merges file  (GPT-2 calls it 'vocab.bpe').
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            # encoder.json is {token_str: id}; we want id->str and str->id
            self.vocab = {int(v): k for k, v in loaded_vocab.items()}
            self.inverse_vocab = {k: int(v) for k, v in loaded_vocab.items()}
    
        # Must have GPT-2's printable newline character 'Ċ' (U+010A) at id 198
        if "Ċ" not in self.inverse_vocab or self.inverse_vocab["Ċ"] != 198:
            raise KeyError("Vocabulary missing GPT-2 newline glyph 'Ċ' at id 198.")
    
        # Must have <|endoftext|> at 50256
        if "<|endoftext|>" not in self.inverse_vocab or self.inverse_vocab["<|endoftext|>"] != 50256:
            raise KeyError("Vocabulary missing <|endoftext|> at id 50256.")
    
        # Provide a convenience alias for '\n' -> 198
        # Keep printable character 'Ċ' in vocab so BPE merges keep working
        if "\n" not in self.inverse_vocab:
            self.inverse_vocab["\n"] = self.inverse_vocab["Ċ"]

        if "\r" not in self.inverse_vocab:
            if 201 in self.vocab:
                self.inverse_vocab["\r"] = 201
            else:
                raise KeyError("Vocabulary missing carriage return token at id 201.")

        # Load GPT-2 merges and store ranks
        self.bpe_ranks = {}
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            lines = file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]
    
            rank = 0
            for line in lines:
                token1, *rest = line.strip().split()
                if len(rest) != 1:
                    continue
                token2 = rest[0]
                if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                    self.bpe_ranks[(token1, token2)] = rank
                    rank += 1
                else:
                    # Safe to skip pairs whose symbols are not in vocab
                    pass


    def encode(self, text, allowed_special=None):
        """
        Encode the input text into a list of token IDs, with tiktoken-style handling of special tokens.
    
        Args:
            text (str): The input text to encode.
            allowed_special (set or None): Special tokens to allow passthrough. If None, special handling is disabled.
    
        Returns:
            List of token IDs.
        """
    
        # ---- This section is to mimic tiktoken in terms of allowed special tokens ----
        specials_in_vocab = [
            tok for tok in self.inverse_vocab
            if tok.startswith("<|") and tok.endswith("|>")
        ]
        if allowed_special is None:
            # Nothing is allowed
            disallowed = [tok for tok in specials_in_vocab if tok in text]
            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        else:
            # Some spefic tokens are allowed (e.g., we use this for <|endoftext|>)
            disallowed = [tok for tok in specials_in_vocab if tok in text and tok not in allowed_special]
            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")
        # -----------------------------------------------------------------------------

        token_ids = []
        # If some specials are allowed, split around them and passthrough those ids
        if allowed_special is not None and len(allowed_special) > 0:
            special_pattern = "(" + "|".join(
                re.escape(tok) for tok in sorted(allowed_special, key=len, reverse=True)
            ) + ")"
    
            last_index = 0
            for match in re.finditer(special_pattern, text):
                prefix = text[last_index:match.start()]
                token_ids.extend(self.encode(prefix, allowed_special=None))  # encode prefix normally
    
                special_token = match.group(0)
                if special_token in self.inverse_vocab:
                    token_ids.append(self.inverse_vocab[special_token])
                else:
                    raise ValueError(f"Special token {special_token} not found in vocabulary.")
                last_index = match.end()
    
            text = text[last_index:]  # remainder to process normally
    
            # Extra guard for any other special literals left over
            disallowed = [
                tok for tok in self.inverse_vocab
                if tok.startswith("<|") and tok.endswith("|>") and tok in text and tok not in allowed_special
            ]
            if disallowed:
                raise ValueError(f"Disallowed special tokens encountered in text: {disallowed}")

    
        # ---- Newline and carriage return handling ----
        tokens = self.pretokenize_text(text)
        # ---------------------------------------------------------------
    
        # Map tokens -> ids (BPE if needed)
        for tok in tokens:
            if tok in self.inverse_vocab:
                token_ids.append(self.inverse_vocab[tok])
            else:
                token_ids.extend(self.tokenize_with_bpe(tok))
    
        return token_ids

    def tokenize_with_bpe(self, token):
        """
        Tokenize a single token using BPE merges.

        Args:
            token (str): The token to tokenize.

        Returns:
            List[int]: The list of token IDs after applying BPE.
        """
        # Tokenize the token into individual characters (as initial token IDs)
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")

        # If we haven't loaded OpenAI's GPT-2 merges, use my approach
        if not self.bpe_ranks:
            can_merge = True
            while can_merge and len(token_ids) > 1:
                can_merge = False
                new_tokens = []
                i = 0
                while i < len(token_ids) - 1:
                    pair = (token_ids[i], token_ids[i + 1])
                    if pair in self.bpe_merges:
                        merged_token_id = self.bpe_merges[pair]
                        new_tokens.append(merged_token_id)
                        # Uncomment for educational purposes:
                        # print(f"Merged pair {pair} -> {merged_token_id} ('{self.vocab[merged_token_id]}')")
                        i += 2  # Skip the next token as it's merged
                        can_merge = True
                    else:
                        new_tokens.append(token_ids[i])
                        i += 1
                if i < len(token_ids):
                    new_tokens.append(token_ids[i])
                token_ids = new_tokens
            return token_ids

        # Otherwise, do GPT-2-style merging with the ranks:
        # 1) Convert token_ids back to string "symbols" for each ID
        symbols = [self.vocab[id_num] for id_num in token_ids]

        # Repeatedly merge all occurrences of the lowest-rank pair
        while True:
            # Collect all adjacent pairs
            pairs = set(zip(symbols, symbols[1:]))
            if not pairs:
                break

            # Find the pair with the best (lowest) rank
            min_rank = float("inf")
            bigram = None
            for p in pairs:
                r = self.bpe_ranks.get(p, float("inf"))
                if r < min_rank:
                    min_rank = r
                    bigram = p

            # If no valid ranked pair is present, we're done
            if bigram is None or bigram not in self.bpe_ranks:
                break

            # Merge all occurrences of that pair
            first, second = bigram
            new_symbols = []
            i = 0
            while i < len(symbols):
                # If we see (first, second) at position i, merge them
                if i < len(symbols) - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(first + second)  # merged symbol
                    i += 2
                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

            if len(symbols) == 1:
                break

        # Finally, convert merged symbols back to IDs
        merged_ids = [self.inverse_vocab[sym] for sym in symbols]
        return merged_ids

    def decode(self, token_ids):
        """
        Decode a list of token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        out = []
        for tid in token_ids:
            if tid not in self.vocab:
                raise ValueError(f"Token ID {tid} not found in vocab.")
            tok = self.vocab[tid]

            # Map GPT-2 special chars back to real chars
            if tid == 198 or tok == "\n":
                out.append("\n")
            elif tid == 201 or tok == "\r":
                out.append("\r")
            elif tok.startswith("Ġ"):
                out.append(" " + tok[1:])
            else:
                out.append(tok)
        return "".join(out)

    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Save the vocabulary and BPE merges to JSON files.

        Args:
            vocab_path (str): Path to save the vocabulary.
            bpe_merges_path (str): Path to save the BPE merges.
        """
        # Save vocabulary
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump(self.vocab, file, ensure_ascii=False, indent=2)

        # Save BPE merges as a list of dictionaries
        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id}
                           for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)

    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        """
        Load the vocabulary and BPE merges from JSON files.

        Args:
            vocab_path (str): Path to the vocabulary file.
            bpe_merges_path (str): Path to the BPE merges file.
        """
        # Load vocabulary
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}

        # Load BPE merges
        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge["new_id"]
                self.bpe_merges[pair] = new_id

    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)

    @staticmethod
    def pretokenize_text(text):
        tokens = []
        parts = re.split(r'(\r\n|\r|\n)', text)
        for part in parts:
            if part == "":
                continue
            if part == "\r\n":
                tokens.append("\r")
                tokens.append("\n")
                continue
            if part == "\r":
                tokens.append("\r")
                continue
            if part == "\n":
                tokens.append("\n")
                continue

            # Normal chunk without line breaks:
            # - If spaces precede a word, prefix the first word with 'Ġ' and
            #   add standalone 'Ġ' for additional spaces
            # - If spaces trail the chunk (e.g., before a newline) add
            #   standalone 'Ġ' tokens (tiktoken produces id 220 for 'Ġ')
            pending_spaces = 0
            for m in re.finditer(r'( +)|(\S+)', part):
                if m.group(1) is not None:
                    pending_spaces += len(m.group(1))
                else:
                    word = m.group(2)
                    if pending_spaces > 0:
                        for _ in range(pending_spaces - 1):
                            tokens.append("Ġ")  # remaining spaces as standalone
                        tokens.append("Ġ" + word) # one leading space
                        pending_spaces = 0
                    else:
                        tokens.append(word)
            # Trailing spaces (no following word): add standalone 'Ġ' tokens
            for _ in range(pending_spaces):
                tokens.append("Ġ")
        return tokens

    @staticmethod
    def find_freq_pair(token_id_sequences, mode="most"):
        pairs = Counter(
            pair
            for token_ids in token_id_sequences
            for pair in zip(token_ids, token_ids[1:])
        )

        if not pairs:
            return None

        if mode == "most":
            return max(pairs.items(), key=lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invalid mode. Choose 'most' or 'least'.")

    @staticmethod
    def replace_pair(token_id_sequences, pair_id, new_id):
        replaced_sequences = []

        for token_ids in token_id_sequences:
            dq = deque(token_ids)
            replaced = []

            while dq:
                current = dq.popleft()
                if dq and (current, dq[0]) == pair_id:
                    replaced.append(new_id)
                    # Remove the 2nd token of the pair, 1st was already removed
                    dq.popleft()
                else:
                    replaced.append(current)

            replaced_sequences.append(replaced)

        return replaced_sequences

