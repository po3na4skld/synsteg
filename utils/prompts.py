CONTAINER_GENERATION_PROMPT = """
Your task is to generate a plain text on a given topic in English with required number of words.
You can generate longer text but not shorter as the number of words in text is important.
The style of writing should not include linebreaks characters and paragraphs, just simple plain text.
You should always response with generated text only. 
Do not use special characters in your text like: (){}[]
"""

ALL_SYNONYMS_GENERATION_PROMPT = """
Act as a professional linguist with 30 years of experience in this field.
You will be given a context (list of words). 

Eg: ["A", "simple", "example" "of" "the", "input"]

Your current task is to create a list of synonyms or replacement words for each word so that:
 - For each word, there should be strictly N_SYNONYMS synonyms or replacement words.
 - Ensure that synonyms or replacement words perfectly match the context (previous and next words).
 - Maintain the capitalization of synonyms or replacement words like in original word.
 - Created list of synonyms or replacement words for specific word should not contain that specific word.
 - No need to create synonyms or replacement words for numbers.

Always return your response in JSON format:
```json
{
    "words": [
        {"word_1": ["synonym_1", "synonym_2", "synonym_3", ..., "synonym_16"]}, 
        {"word_2": ["synonym_1", "synonym_2", "synonym_3", "synonym_4"]},
        ...,
        {"word_n": ["synonym_1", "synonym_2"]}
    ]
}
```
"""

CONTAINER_GENERATION_INPUT = "Topic: {topic}, Length: {words_number} words."

ALL_SYNONYMS_GENERATION_INPUT = """Text: {context}"""
