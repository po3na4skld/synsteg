CONTAINER_GENERATION_PROMPT = """Your task is to generate an essay on a given topic in English with required length.
You can generate longer text but not shorter as the number of words in text is really important.
You should always response with generated text only. Do not use special characters in your text like: (){}[]
"""

ALL_SYNONYMS_GENERATION_PROMPT = """
Act as a professional linguist with 30 years of experience in this field.
You will be given a context (list of words). 

Eg: ["A", "simple", "example" "of" "the", "input"]

Your current task is to create a list of synonyms for each word so that:
 - For each word, there should be 2, 4, or 8 synonyms (the number may vary from word to word).
 - Ensure that synonyms perfectly match the context (previous and next words).
 - Maintain the capitalization of synonyms like in original word.
 - Created list of synonyms for specific word should not contain that specific word.
 - No need to create synonyms for numbers.

Always format your response in JSON format as follows:
```json
{
    "words": [
        {"word_1": ["synonym_1", "synonym_2", "synonym_3", "synonym_n"]}, 
        {"word_2": ["synonym_1", "synonym_2", "synonym_3", "synonym_4"]},
        ...,
        {"word_n": ["synonym_1", "synonym_2"]}
    ]
}
```
"""

CONTAINER_GENERATION_INPUT = "Topic: {topic}, Length: {words_number} words."

ALL_SYNONYMS_GENERATION_INPUT = """Text: {context}"""
