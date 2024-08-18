import os
from abc import ABC, abstractmethod
from typing import List

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteriaList,
    StoppingCriteria,
    TRANSFORMERS_CACHE,
)

from configs import logger


class LLM(ABC):
    @abstractmethod
    def init_model(self):
        """Initializes the llm"""

    @abstractmethod
    def is_model_available(self) -> bool:
        """Returns whether the model is available or it must be downloaded."""


class InFiller(ABC):
    @abstractmethod
    def infill(self, left_context: str, right_context: str) -> str:
        """Returns the missing code between left and right contexts."""


class FacebookInFiller(LLM, InFiller):
    """
    Implementation by using Facebook InCoder.

    References:
        - https://huggingface.co/facebook/incoder-6B
        - https://github.com/dpfried/incoder/blob/main/example_usage.py
        - https://github.com/code4me-me/code4me/blob/main/code4me-server/src/incoder.py
    """

    SMALL_MODEL_NAME = "facebook/incoder-1B"
    BIG_MODEL_NAME = "facebook/incoder-6B"
    EOF = "<|/ file |>"
    STOP_TOKENS = [
        205,
        284,
        353,
        536,
        994,
        3276,
        4746,
        15471,
        16027,
        28602,
        40289,
        43275,
        50517,
    ]

    def __init__(self, big_model: bool = False, cuda: bool = True):
        self.model = None
        self.tokenizer = None
        self.cuda = cuda and torch.cuda.is_available()
        self.device = "cuda" if self.cuda else "cpu"

        if big_model:
            self.model_name = self.BIG_MODEL_NAME
            if self.cuda:
                self.kwargs = dict(
                    revision="float16",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )
            else:
                self.kwargs = dict(
                    low_cpu_mem_usage=True,
                )
        else:
            self.model_name = self.SMALL_MODEL_NAME
            self.kwargs = {}

    def init_model(self):
        logger.info(
            f"Using model: {self.model_name}, cuda: {self.cuda}, device: {self.device}"
        )

        logger.info("Loading model")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, **self.kwargs
        )
        logger.info("Loading tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        logger.info("Loading complete")

        if self.cuda:
            self.model = self.model.half().cuda()

    def is_model_available(self) -> bool:
        cache_dir = TRANSFORMERS_CACHE
        logger.debug(f"{cache_dir = }")

        model_folder_name = self.model_name.replace("/", "--")

        if not os.path.exists(cache_dir):
            logger.debug("Cache directory does not exist.")
            return False

        logger.debug("Walking cache directory")
        for model_dir in os.listdir(cache_dir):
            logger.debug(f"{model_dir = }")
            if model_folder_name in model_dir:
                logger.info(f"Model directory found at: {model_dir}")
                return True

        return False

    @staticmethod
    def make_sentinel(i):
        return f"<|mask:{i}|>"

    def decode(self, tokens):
        return self.tokenizer.decode(
            tokens, clean_up_tokenization_spaces=False, skip_special_tokens=True
        )

    def infill(self, left_context: str, right_context: str) -> str:
        logger.info("Filling the gap...")

        left_input_ids = (
            self.tokenizer(
                left_context, return_tensors="pt", return_token_type_ids=False
            )
            .to(self.device)
            .input_ids[0]
        )
        right_input_ids = (
            self.tokenizer(
                right_context, return_tensors="pt", return_token_type_ids=False
            )
            .to(self.device)
            .input_ids[0]
        )

        left_context = self.decode(
            left_input_ids[-max(1000, 2000 - len(right_input_ids)) :]
        )

        right_context = self.decode(
            right_input_ids[: max(1000, 2000 - len(left_input_ids))]
        )

        prompt = (
            f"{left_context}"
            f"{self.make_sentinel(0)}"
            f"{right_context}"
            f"{self.EOF}"
            f"{self.make_sentinel(1)}"
            f"{self.make_sentinel(0)}"
        )

        tokens = self.tokenizer(
            prompt, return_tensors="pt", return_token_type_ids=False
        )
        tokens = tokens.to(self.device)

        token_count = len(tokens.input_ids[0])

        stopping_criteria = StoppingCriteriaList()
        stopping_criteria.append(
            StatementStoppingCriteria(token_count, self.STOP_TOKENS)
        )

        with torch.no_grad():
            completion = self.model.generate(
                **tokens,
                do_sample=True,
                top_p=0.95,
                temperature=0.2,
                max_length=min(2048, token_count + 48),
                stopping_criteria=stopping_criteria,
            )[0][token_count:]

        return self.decode(completion)


class StatementStoppingCriteria(StoppingCriteria):
    def __init__(self, init_length: int, stop_tokens: List[int]):
        self.init_length = init_length
        self.stop_tokens = set(stop_tokens)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        return any(
            token in self.stop_tokens for token in input_ids[0][self.init_length :]
        )
