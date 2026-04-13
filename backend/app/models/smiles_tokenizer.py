"""
SMILES Tokenizer

Character-level tokenization for SMILES molecular strings.
Converts SMILES strings to integer sequences and back.
"""

from typing import List, Optional

from app.utils.constants import (
    SMILES_CHARSET,
    PAD_TOKEN,
    START_TOKEN,
    END_TOKEN,
    MAX_MOLECULE_LENGTH,
)


class SmilesTokenizer:
    """
    Character-level SMILES tokenizer.

    Encodes SMILES strings into integer sequences and decodes them back.
    Includes special tokens: <PAD>, <START>, <END>.
    """

    def __init__(self, max_length: int = MAX_MOLECULE_LENGTH):
        self.max_length = max_length

        # Build vocabulary: special tokens + SMILES characters
        self.special_tokens = [PAD_TOKEN, START_TOKEN, END_TOKEN]
        self.characters = list(SMILES_CHARSET)
        self.vocabulary = self.special_tokens + self.characters

        # Token ↔ Index mappings
        self.token_to_index = {
            token: index for index, token in enumerate(self.vocabulary)
        }
        self.index_to_token = {
            index: token for token, index in self.token_to_index.items()
        }

        # Quick access to special token indices
        self.pad_index = self.token_to_index[PAD_TOKEN]
        self.start_index = self.token_to_index[START_TOKEN]
        self.end_index = self.token_to_index[END_TOKEN]

    @property
    def vocabulary_size(self) -> int:
        """Total number of tokens in the vocabulary."""
        return len(self.vocabulary)

    def encode(self, smiles_string: str) -> List[int]:
        """
        Encode a SMILES string into a padded integer sequence.

        Format: [<START>, char1, char2, ..., <END>, <PAD>, <PAD>, ...]

        Args:
            smiles_string: A valid SMILES molecular string.

        Returns:
            List of integer token indices, length = max_length + 2.
        """
        token_indices = [self.start_index]

        for character in smiles_string[: self.max_length]:
            index = self.token_to_index.get(character, self.pad_index)
            token_indices.append(index)

        token_indices.append(self.end_index)

        # Pad to fixed length: max_length + 2 (for START and END)
        total_length = self.max_length + 2
        padding_needed = total_length - len(token_indices)
        token_indices.extend([self.pad_index] * padding_needed)

        return token_indices

    def decode(self, token_indices: List[int]) -> str:
        """
        Decode a sequence of integer indices back to a SMILES string.

        Strips <PAD>, <START>, and <END> tokens automatically.

        Args:
            token_indices: List of integer token indices.

        Returns:
            Decoded SMILES string.
        """
        characters = []

        for index in token_indices:
            token = self.index_to_token.get(index, "")

            if token == END_TOKEN:
                break
            if token in (PAD_TOKEN, START_TOKEN):
                continue

            characters.append(token)

        return "".join(characters)

    def encode_batch(self, smiles_list: List[str]) -> List[List[int]]:
        """Encode a batch of SMILES strings."""
        return [self.encode(smiles) for smiles in smiles_list]

    def decode_batch(self, token_batch: List[List[int]]) -> List[str]:
        """Decode a batch of token sequences back to SMILES strings."""
        return [self.decode(tokens) for tokens in token_batch]
