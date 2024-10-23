import os
import re
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from uuid import uuid4

import boto3


class PPITypes(StrEnum):
    NAME = "NAME"
    PHONE = "PHONE"
    EMAIL = "EMAIL"
    ADDRESS = "ADDRESS"


@dataclass
class MaskedSet:
    text: str
    tokens: dict[str, str]


class BaseTokenizer(ABC):

    @abstractmethod
    def locate_ppi(self, text: str) -> dict[PPITypes, list]:
        """
        Locates PPI in the provided text.
        """
        pass

    def mask(self, text: str) -> MaskedSet:
        tokens = {}
        ppi = self.locate_ppi(text)
        for ppi_type, ppi_data in ppi.items():
            for value in ppi_data:
                token = gen_token(ppi_type)
                text = text.replace(value, token)
                tokens[token] = value
        return MaskedSet(text=text, tokens=tokens)

    def unmask(self, text: str, tokens: dict[str, str]) -> str:
        for token, value in tokens.items():
            text = text.replace(token, value)
        return text


def get_tokenizer_cls(name: str | None = None) -> type[BaseTokenizer]:
    """
    Reuturns a tokenizer class based on the provided name.
    """
    name = name or os.getenv("LLM_TOKENIZER_CLASS", "regex")

    if name == "regex":
        return RegexTokenizer
    elif name == "aws":
        return AWSComprehendTokenizer
    elif name == "gcp":
        return GCPNaturalLanguageTokenizer
    else:
        raise ValueError(f"Unknown tokenizer: {name}")


def gen_token(type: PPITypes | None = None) -> str:
    if not type:
        return f"<{uuid4()}>"
    return f"<{type.value}:{uuid4()}>"


class RegexTokenizer(BaseTokenizer):

    def locate_fname(self, text) -> list[str]:
        pattern = r"John|Greg|George|Michael"
        return re.findall(pattern, text)

    def locate_lname(self, text) -> list[str]:
        pattern = r"Doe|Smith|Johnson|Williams|Jones"
        return re.findall(pattern, text)

    def locate_phone_numbers(self, text) -> list[str]:
        pattern = r"\b\d{2,3}-?\d{3}-?\d{2}-?\d{2}(?!\d)\b|\b\d{9,10}\b"
        return re.findall(pattern, text)

    def locate_emails(self, text) -> list[str]:
        pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
        return re.findall(pattern, text)

    def locate_address(self, text) -> list[str]:
        pattern = r"He lives at ([^,]+)"
        return re.findall(pattern, text)

    def locate_ppi(self, text: str) -> dict[PPITypes, list]:
        return {
            PPITypes.NAME: self.locate_fname(text) + self.locate_lname(text),
            PPITypes.PHONE: self.locate_phone_numbers(text),
            PPITypes.EMAIL: self.locate_emails(text),
            PPITypes.ADDRESS: self.locate_address(text),
        }


class AWSComprehendTokenizer(BaseTokenizer):

    def locate_ppi(self, text) -> dict[PPITypes, list]:
        """
        Locate PPi in the provided text using AWS Comprehend.
        """
        client = boto3.client("comprehend")
        response = client.detect_pii_entities(Text=text, LanguageCode="en")
        ppis = defaultdict(list)

        for entity in response["Entities"]:
            typ = entity["Type"]
            ppi = text[entity["BeginOffset"] : entity["EndOffset"]]

            ppi_type = getattr(PPITypes, typ, None)
            if ppi_type:
                ppis[ppi_type].append(ppi)

        return dict(ppis)


class GCPNaturalLanguageTokenizer(BaseTokenizer):

    def locate_ppi(self, text):
        raise NotImplementedError
