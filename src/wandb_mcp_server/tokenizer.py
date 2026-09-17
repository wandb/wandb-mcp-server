"""Pinned, offline token counting for response budgets."""

import base64
import hashlib
from functools import lru_cache
from importlib.resources import files

from tiktoken import Encoding

TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"

# cl100k_base metadata from tiktoken 0.11.0; attribution lives beside the data.
_PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"
_SPECIAL_TOKENS = {
    "<|endoftext|>": 100257,
    "<|fim_prefix|>": 100258,
    "<|fim_middle|>": 100259,
    "<|fim_suffix|>": 100260,
    "<|endofprompt|>": 100276,
}


@lru_cache(maxsize=1)
def load_tokenizer() -> Encoding:
    """Verify the packaged vocabulary before constructing the shared encoding."""
    try:
        vocabulary = files("wandb_mcp_server").joinpath("data", "cl100k_base.tiktoken").read_bytes()
    except OSError as exc:
        raise RuntimeError("packaged tokenizer is missing or unreadable") from exc
    if hashlib.sha256(vocabulary).hexdigest() != TOKENIZER_SHA256:
        raise RuntimeError("packaged tokenizer checksum does not match")
    ranks = {}
    for line in vocabulary.splitlines():
        token, rank = line.split()
        ranks[base64.b64decode(token, validate=True)] = int(rank)
    return Encoding(name="cl100k_base", pat_str=_PATTERN, mergeable_ranks=ranks, special_tokens=_SPECIAL_TOKENS)
