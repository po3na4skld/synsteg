import json
import random

import openai
import backoff
import httpx

from models.pool_arguments import LLMConfig
from utils import prompts, constants

OPENAI = LLMConfig()


def _ollama_chat(llm: LLMConfig, messages: list[dict], temperature: float | None, json_schema: dict | None):
    # ollama's format="json" does not constrain qwen3; a JSON schema does (experiments/model_series/run_series.py)
    body = dict(model=llm.model or "qwen3:30b-a3b-instruct-2507-q4_K_M", messages=messages, stream=False, think=False,
                options=dict(num_predict=4096, **({} if temperature is None else dict(temperature=temperature))))
    if json_schema:
        body["format"] = json_schema
    r = httpx.post(llm.url.rstrip("/") + "/api/chat", json=body, timeout=1200)
    r.raise_for_status()
    r = r.json()
    pi, ci = r.get("prompt_eval_count", 0), r.get("eval_count", 0)
    return r["message"]["content"], dict(prompt_tokens=pi, completion_tokens=ci, total_tokens=pi + ci)


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIStatusError),
                      giveup=lambda e: isinstance(e, openai.BadRequestError))
def _openai_chat(model: str, messages: list[dict], temperature: float | None, json_mode: bool, api_key: str | None = None):
    kwargs = dict(model=model, messages=messages)
    if temperature is not None and not model.startswith("gpt-6"):  # gpt-6 models accept only the default temperature
        kwargs["temperature"] = temperature
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    response = (openai.OpenAI(api_key=api_key) if api_key else openai.OpenAI()).chat.completions.create(**kwargs)
    return response.choices[0].message.content if response.choices else None, response.usage.model_dump()


def chat(llm: LLMConfig, prompt: str, input_message: str, temperature: float | None = None,
         json_schema: dict | None = None, default_model: str | None = None):
    """The one LLM call of the method: (content, usage). json_schema != None asks for JSON output
    (OpenAI: JSON mode, ollama: the schema itself)."""
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": input_message}]
    if llm.provider == "ollama":
        return _ollama_chat(llm, messages, temperature, json_schema)
    if llm.provider != "openai":
        raise ValueError(f"unknown LLM provider {llm.provider!r}")
    return _openai_chat(llm.model or default_model or constants.OPENAI_MODEL_SYNONYMS, messages, temperature, json_schema is not None,
                        llm.api_key)


def words_schema(n_synonyms: int) -> dict:
    """Shape of the synonym-table answer; maxItems keeps a local model from listing synonyms until the token limit."""
    return {"type": "object", "required": ["words"], "properties": {"words": {"type": "array", "items": {
        "type": "object", "additionalProperties": {"type": "array", "maxItems": n_synonyms, "items": {"type": "string"}}}}}}


def get_openai_json_output(prompt: str, input_message: str, output_key: str = None, temperature: float = 1.0,
                           llm: LLMConfig = OPENAI, json_schema: dict | None = None):
    """JSON answer of the configured model (OpenAI by default); output_key selects one field. Returns (value, usage)."""
    content, usage = chat(llm, prompt, input_message, temperature, json_schema or {"type": "object"})
    data = json.loads(content)
    return (data[output_key] if output_key else data), usage


def get_openai_output(prompt: str, input_message: str, temperature: float = 1.0, llm: LLMConfig = OPENAI):
    """Plain-text answer of the configured model (OpenAI by default). Returns (text or None, usage)."""
    return chat(llm, prompt, input_message, temperature, default_model=constants.OPENAI_MODEL_CONTAINER)


def generate_container(words_number: int, llm: LLMConfig = OPENAI) -> str:
    """
    Generate a container with a specified number of words using OpenAI language model.

    Args:
        words_number (int): The desired number of words in the generated container.

    Returns:
        str: The generated container text.

    The function uses OpenAI's language model to generate a container text based on a predefined prompt.
    The prompt includes a random topic selected from a set of constants.TOPICS and an adjusted word count
    based on the given `words_number` multiplied by constants.CONTAINER_BUFFER.

    The generated container text is then processed to improve formatting by replacing multiple newlines
    with periods, removing extra spaces, and ensuring proper sentence endings.

    If the generated text contains fewer words than the specified `words_number`, the function recursively
    calls itself until the desired word count is achieved.
    """

    input_message = prompts.CONTAINER_GENERATION_INPUT.format(
        topic=random.choice(constants.TOPICS),
        words_number=int(words_number * constants.CONTAINER_BUFFER)
    )

    container, usage = get_openai_output(
        prompt=prompts.CONTAINER_GENERATION_PROMPT,
        input_message=input_message,
        temperature=constants.CONTAINER_TEMPERATURE,
        llm=llm,
    )

    container = container.replace('.\n\n', '. ').replace('\n\n', '. ').replace('\n', '').replace(' . ', ' ')
    if len(container.split()) < words_number:
        return generate_container(words_number, llm)
    return container, usage
