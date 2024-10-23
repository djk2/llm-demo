import os
import uuid
from unittest.mock import patch

import pytest

from llm_demo.tokenizers import (
    AWSComprehendTokenizer,
    GCPNaturalLanguageTokenizer,
    PPITypes,
    RegexTokenizer,
    gen_token,
    get_tokenizer_cls,
)


def test_dummpy():
    assert True


def test_regex_tokenizer_locate_phone_numbers():
    tokenizer = RegexTokenizer()

    text = "My phone number is 123-456-78-90"
    assert tokenizer.locate_phone_numbers(text) == ["123-456-78-90"]

    text = "Mike phone number is 1234567890 and John's is 111-222-33-44"
    assert tokenizer.locate_phone_numbers(text) == ["1234567890", "111-222-33-44"]

    text = "Who has number 555-123-4567 or 987-654-3212?"
    assert tokenizer.locate_phone_numbers(text) == ["555-123-4567", "987-654-3212"]

    text = "I do not have any phone number"
    assert tokenizer.locate_phone_numbers(text) == []


def test_regex_tokenizer_locate_fname():
    tokenizer = RegexTokenizer()

    text = "John is a good guy"
    assert tokenizer.locate_fname(text) == ["John"]

    text = "John and George are good guys"
    assert tokenizer.locate_fname(text) == ["John", "George"]

    text = "John, George, and Michael are good guys"
    assert tokenizer.locate_fname(text) == ["John", "George", "Michael"]

    text = "I do not have any friends"
    assert tokenizer.locate_fname(text) == []


def test_regex_tokenizer_locate_lname():
    tokenizer = RegexTokenizer()

    text = "John Doe is a good guy"
    assert tokenizer.locate_lname(text) == ["Doe"]

    text = "John Doe and George Smith are good guys"
    assert tokenizer.locate_lname(text) == ["Doe", "Smith"]

    text = "John Doe, George Smith, and Michael Johnson are good guys"
    assert tokenizer.locate_lname(text) == ["Doe", "Smith", "Johnson"]

    text = "I do not have any friends"
    assert tokenizer.locate_lname(text) == []


def test_regex_tokenizer_locate_emails():
    tokenizer = RegexTokenizer()

    text = "My email is g.t@xyz.example.com"
    assert tokenizer.locate_emails(text) == ["g.t@xyz.example.com"]

    text = "Some text without @ . and .com and real emails: a@example.com, bob@a.xyz.com.pl"
    assert tokenizer.locate_emails(text) == ["a@example.com", "bob@a.xyz.com.pl"]


def test_regex_tokenizer_locate_address():
    tokenizer = RegexTokenizer()

    text = "He lives at 123 Main St."
    assert tokenizer.locate_address(text) == ["123 Main St."]

    text = "All we live in Poland, but He lives at 123 Maple Street in San Francisco, and his..."
    assert tokenizer.locate_address(text) == ["123 Maple Street in San Francisco"]

    text = "I do not have any address"
    assert tokenizer.locate_address(text) == []


def test_regex_tokenizer_locate_ppi():
    tokenizer = RegexTokenizer()
    text = (
        "John's phone number is 555-123-4567. "
        "He lives at 123 Maple Street in San Francisco, "
        "and his email is john.doe@example.com."
    )
    assert tokenizer.locate_ppi(text) == {
        PPITypes.NAME: ["John"],
        PPITypes.PHONE: ["555-123-4567"],
        PPITypes.EMAIL: ["john.doe@example.com"],
        PPITypes.ADDRESS: ["123 Maple Street in San Francisco"],
    }


def test_gen_token():
    with patch(
        "llm_demo.tokenizers.uuid4",
        return_value=uuid.UUID("fcb9824c-aa05-4463-a6e7-a4d15990c3a0"),
    ):

        assert gen_token(PPITypes.NAME) == "<NAME:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>"
        assert gen_token(PPITypes.PHONE) == "<PHONE:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>"
        assert gen_token(PPITypes.EMAIL) == "<EMAIL:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>"
        assert gen_token(PPITypes.ADDRESS) == "<ADDRESS:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>"
        assert gen_token() == "<fcb9824c-aa05-4463-a6e7-a4d15990c3a0>"


def test_regex_tokenizer_mask():
    text = (
        "John's phone number is 555-123-4567. "
        "He lives at 123 Maple Street in San Francisco, "
        "and his email is john.doe@example.com."
    )
    with patch(
        "llm_demo.tokenizers.uuid4",
        return_value=uuid.UUID("fcb9824c-aa05-4463-a6e7-a4d15990c3a0"),
    ):
        masked_set = RegexTokenizer().mask(text)

    assert masked_set.text == (
        "<NAME:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>'s phone number is "
        "<PHONE:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>. "
        "He lives at <ADDRESS:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>, "
        "and his email is <EMAIL:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>."
    )
    assert masked_set.tokens == {
        "<NAME:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "John",
        "<PHONE:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "555-123-4567",
        "<ADDRESS:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "123 Maple Street in San Francisco",
        "<EMAIL:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "john.doe@example.com",
    }


def test_regex_tokenizer_unmask():
    text = (
        "<NAME:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>'s phone number is "
        "<PHONE:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>. "
        "He lives at <ADDRESS:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>, "
        "and his email is <EMAIL:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>."
    )
    tokens = {
        "<NAME:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "John",
        "<PHONE:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "555-123-4567",
        "<ADDRESS:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "123 Maple Street in San Francisco",
        "<EMAIL:fcb9824c-aa05-4463-a6e7-a4d15990c3a0>": "john.doe@example.com",
    }
    assert RegexTokenizer().unmask(text, tokens) == (
        "John's phone number is 555-123-4567. "
        "He lives at 123 Maple Street in San Francisco, "
        "and his email is john.doe@example.com."
    )


def test_get_tokenizer_cls():
    assert get_tokenizer_cls("regex") == RegexTokenizer
    assert get_tokenizer_cls("aws") == AWSComprehendTokenizer
    assert get_tokenizer_cls("gcp") == GCPNaturalLanguageTokenizer
    with pytest.raises(ValueError):
        get_tokenizer_cls(name="unknown")

    # Check the env variable works
    os.environ["LLM_TOKENIZER_CLASS"] = "aws"
    assert get_tokenizer_cls() == AWSComprehendTokenizer
