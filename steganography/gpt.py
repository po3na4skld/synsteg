import json
import random

import openai
import backoff

from utils import prompts, constants


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIStatusError))
def get_openai_json_output(prompt: str, input_message: str, output_key: str = None, temperature: float = 1.0):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=constants.OPENAI_MODEL_SYNONYMS,
        temperature=temperature,
        messages=[{"role": "system", "content": prompt}, {'role': 'user', 'content': input_message}],
        response_format={"type": "json_object"}
    )

    print('SECRET KEY', response.usage.completion_tokens, response.usage.prompt_tokens)

    if output_key:
        return json.loads(response.choices[0].message.content)[output_key]
    else:
        return json.loads(response.choices[0].message.content)


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIStatusError))
def get_openai_output(prompt: str, input_message: str, temperature: float = 1.0):
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=constants.OPENAI_MODEL_CONTAINER,
        temperature=temperature,
        messages=[{"role": "system", "content": prompt}, {'role': 'user', 'content': input_message}]
    )
    print('CONTAINER', response.usage.completion_tokens, response.usage.prompt_tokens)
    return response.choices[0].message.content if response.choices else None


def generate_container(words_number: int) -> str:
    input_message = prompts.CONTAINER_GENERATION_INPUT.format(
        topic=random.choice(constants.TOPICS),
        words_number=int(words_number * constants.CONTAINER_BUFFER)
    )

    container = get_openai_output(
        prompt=prompts.CONTAINER_GENERATION_PROMPT,
        input_message=input_message,
        temperature=constants.CONTAINER_TEMPERATURE
    ).replace('.\n\n', '. ').replace('\n\n', '. ').replace('\n', '').replace(' . ', ' ')

    if len(container.split()) < words_number:
        return generate_container(words_number)
    return container
