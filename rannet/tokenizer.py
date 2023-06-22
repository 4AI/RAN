# -*- coding: utf-8 -*-

from typing import Optional, List, Union, Dict, Iterator, Tuple

from tokenizers import Tokenizer, decoders, trainers
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing
from tokenizers.implementations.base_tokenizer import BaseTokenizer


class SpecialTokens:
    def __init__(self, unused_num: int = 1000) -> None:
        self.PAD = '[PAD]'
        self.UNK = '[UNK]'
        self.MASK = '[MASK]'
        self.SEP = '[SEP]'
        self.START = '[START]'
        self.END = '[END]'
        self.UNUSED_TOKENS = [f'[unused{i}]' for i in range(unused_num)]

    def __contains__(self, token: str) -> bool:
        """ Check if the input token exists in special tokens.
        Args:
          - token: str
        Return:
          bool
        """
        return token in self.tokens

    @property
    def tokens(self) -> List[str]:
        ret = []
        for field in self.__dict__.keys():
            if field.startswith('_'):
                continue
            if isinstance(getattr(self, field), str):
                ret.append(getattr(self, field))
            elif isinstance(getattr(self, field), list):
                ret.extend(getattr(self, field))
        return ret


class RanNetWordPieceTokenizer(BaseTokenizer):
    """ RanNet WordPiece Tokenizer """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        special_tokens: Optional[SpecialTokens] = None,
        clean_text: bool = True,
        handle_chinese_chars: bool = True,
        strip_accents: Optional[bool] = None,
        lowercase: bool = True,
        wordpieces_prefix: str = "##",
    ):
        if special_tokens is None:
            special_tokens = SpecialTokens()
        if vocab is not None:
            tokenizer = Tokenizer(WordPiece(vocab, unk_token=str(special_tokens.UNK)))
        else:
            tokenizer = Tokenizer(WordPiece(unk_token=str(special_tokens.UNK)))

        # Let the tokenizer know about special tokens if they are part of the vocab
        for special_token in special_tokens.tokens:
            if tokenizer.token_to_id(special_token) is not None:
                tokenizer.add_special_tokens([special_token])

        tokenizer.normalizer = BertNormalizer(
            clean_text=clean_text,
            handle_chinese_chars=handle_chinese_chars,
            strip_accents=strip_accents,
            lowercase=lowercase,
        )
        tokenizer.pre_tokenizer = BertPreTokenizer()
        if vocab is not None:
            sep_token_id = tokenizer.token_to_id(str(special_tokens.SEP))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            tokenizer.post_processor = TemplateProcessing(
                single="$0",
                pair="$A [SEP] $B:1 [SEP]:1",
                special_tokens=[(special_tokens.SEP, sep_token_id)]
            )

        tokenizer.decoder = decoders.WordPiece(prefix=wordpieces_prefix)

        parameters = {
            "model": "RanNetWordPiece",
            "special_tokens": special_tokens,
            "clean_text": clean_text,
            "handle_chinese_chars": handle_chinese_chars,
            "strip_accents": strip_accents,
            "lowercase": lowercase,
            "wordpieces_prefix": wordpieces_prefix,
        }

        super().__init__(tokenizer, parameters)
        self.special_tokens = special_tokens

    @staticmethod
    def from_file(vocab: str, **kwargs):
        vocab = WordPiece.read_file(vocab)
        return RanNetWordPieceTokenizer(vocab, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: Optional[SpecialTokens] = None,
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
    ):
        """ Train the model using the given files """

        if special_tokens is None:
            special_tokens = SpecialTokens()
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens.tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Union[Iterator[str], Iterator[Iterator[str]]],
        vocab_size: int = 30000,
        min_frequency: int = 2,
        limit_alphabet: int = 1000,
        initial_alphabet: List[str] = [],
        special_tokens: Optional[SpecialTokens] = None,
        show_progress: bool = True,
        wordpieces_prefix: str = "##",
        length: Optional[int] = None,
    ):
        """ Train the model using the given iterator """

        if special_tokens is None:
            special_tokens = SpecialTokens()
        trainer = trainers.WordPieceTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            limit_alphabet=limit_alphabet,
            initial_alphabet=initial_alphabet,
            special_tokens=special_tokens.tokens,
            show_progress=show_progress,
            continuing_subword_prefix=wordpieces_prefix,
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )

    def rematch_to_text(self, offsets: List[Tuple[int, int]]) -> List[List[int]]:
        """
        >>> text = 'hello [PAD] world'
        >>> t = tokenizer.encode(text)
        >>> mapping = tokenizer.rematch_to_text(t.offsets)
        >>> for ch_pos in mapping:
                print(text[ch_pos[0]: ch_pos[-1]+1])
        hello
        [PAD]
        world
        """
        mapping = []
        for offset in offsets:
            if offset[0] == 0 and offset[1] == 0:
                mapping.append([])
            else:
                mapping.append([i for i in range(offset[0], offset[1])])
        return mapping
