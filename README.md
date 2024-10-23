# Run demo-app

```bash
cd llm_demo
fastapi dev --reload
```

# Run rests

    pytest -vvs --cov=llm_demo --cov-report term-missing

# Run checks

    ./run-checks.sh



# Tokenizer class
You can change the tokenizer class by setting the following environment variable `LLM_TOKENIZER_CLASS`.

```
export LLM_TOKENIZER_CLASS=aws|regex|gcp
```

* `regex` - **default** - `RegexTokenizer` - try to detect PII based on regex - default
* `aws` - `AWSComprehendTokenizer` - use Amazon Comprehend to detect PII
* `gcp` - **not implemented yet**  -`GCPNaturalLanguageTokenizer` - use Google Cloud Natural Language to detect PII


# use Amazon Comprehend to detect PII
Before running the app, you need to set up the AWS credentials in your environment.
You can do this by setting the following environment variables:

```bash
export AWS_ACCESS_KEY_ID=your_access_key_id
export AWS_SECRET_ACCESS_KEY=your_secret_access_key
export AWS_DEFAULT_REGION=your_default_region
export LLM_TOKENIZER_CLASS=aws
```

if you want to use the default tokenizer based on regex,
you can set the following environment variable:

```bash
export LLM_TOKENIZER_CLASS=regex
```


# Change the llm_service class:
You can change the llm_service class by setting the following environment variable:

```bash
export LLM_SERVICE_CLASS=mock
```

Possible values are:
- mock
- default

# TODO:

- CI/CD
- AWS
