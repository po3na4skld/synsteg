import json
import random

import openai
import backoff

from utils import prompts, constants


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIStatusError))
def get_openai_json_output(prompt: str, input_message: str, output_key: str = None, temperature: float = 1.0):
    """
    Retrieves JSON output from the OpenAI GPT-3 model based on given prompt and input message.

    Args:
        prompt (str): The system-level instruction or prompt for the conversation.
        input_message (str): The user's input message in the conversation.
        output_key (str, optional): The key for extracting a specific value from the JSON output.
        temperature (float, optional): Controls the randomness of the model's output (default is 1.0).

    Returns:
        dict or specified data type: The JSON output from the OpenAI GPT-3 model. If output_key is provided,
        the value associated with that key will be returned.

    Raises:
        openai.RateLimitError: If the OpenAI API rate limit is exceeded.
        openai.APIStatusError: If there is an issue with the OpenAI API status.

    Note:
        This function uses exponential backoff for retries in case of RateLimitError or APIStatusError.
    """

    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=constants.OPENAI_MODEL_SYNONYMS,
        temperature=temperature,
        messages=[{"role": "system", "content": prompt}, {'role': 'user', 'content': input_message}],
        response_format={"type": "json_object"}
    )

    if output_key:
        return json.loads(response.choices[0].message.content)[output_key], response.usage.model_dump()
    else:
        return json.loads(response.choices[0].message.content), response.usage.model_dump()


@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.APIStatusError))
def get_openai_output(prompt: str, input_message: str, temperature: float = 1.0):
    """
    Retrieves OpenAI model output based on a given prompt and input message.

    Args:
        prompt (str): The system-level prompt to guide the model's behavior.
        input_message (str): The user's input message that influences the model's response.
        temperature (float, optional): A parameter controlling the randomness of the output.
            Higher values (e.g., 1.0) make the output more random, while lower values (e.g., 0.2) make it more deterministic.

    Returns:
        str or None: The generated model output as a string. Returns None if no choices are available in the response.
    """
    client = openai.OpenAI()

    response = client.chat.completions.create(
        model=constants.OPENAI_MODEL_CONTAINER,
        temperature=temperature,
        messages=[{"role": "system", "content": prompt}, {'role': 'user', 'content': input_message}]
    )

    return response.choices[0].message.content if response.choices else None, response.usage.model_dump()


def generate_container(words_number: int) -> str:
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
        temperature=constants.CONTAINER_TEMPERATURE
    )

    container = container.replace('.\n\n', '. ').replace('\n\n', '. ').replace('\n', '').replace(' . ', ' ')
    if len(container.split()) < words_number:
        return generate_container(words_number)
    return container, usage
