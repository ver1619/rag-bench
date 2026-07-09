import re


class SimpleTokenizer:
    """
    A lightweight tokenizer for lexical retrieval.

    Processing:
    - lowercase
    - remove punctuation
    - split on whitespace
    """

    def tokenize(
        self,
        text: str,
    ) -> list[str]:

        if not text:
            return []

        text = text.lower()

        text = re.sub(
            r"[^\w\s]",
            " ",
            text,
        )

        return text.split()