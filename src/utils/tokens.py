import tiktoken
enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(s: str) -> int:
    return len(enc.encode(s))