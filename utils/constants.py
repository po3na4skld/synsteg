CONTAINER_BUFFER = 1.25
CONTAINER_TEMPERATURE = 0.9

N_ASCII_BITS = 8
MAX_BITS_PER_WORD = 5
MAX_ADDITIONAL_BITS_MULTIPLIER = 3

OPENAI_MODEL_CONTAINER = "gpt-4o"
OPENAI_MODEL_SYNONYMS = "gpt-4o"

SPECIAL_TOKENS = [",", ".", "!", "?", "..."]
PUNCTUATION = ",.?!"
BRACKETS = "[](){}"

SYNONYM_MAP = list[dict[str, dict[str, str]]]

TOPICS = [
    "The Impact of Technology on Society",
    "Climate Change and its Effects",
    "The Importance of Education in Modern World",
    "Social Media and its Influence on Relationships",
    "Ethical Dilemmas in Artificial Intelligence",
    "Globalization and Cultural Identity",
    "The Pros and Cons of Space Exploration",
    "The Role of Government in Healthcare",
    "Environmental Conservation and Sustainable Living",
    "Cybersecurity Challenges in the Digital Age",
    "Implications of Genetic Engineering",
    "Mental Health Awareness and Stigma",
    "The Future of Work: Automation and Job Displacement",
    "Challenges of Overpopulation",
    "Media Bias and its Impact on Public Opinion",
    "The Significance of Renewable Energy Sources",
    "The Ethics of Cloning and Reproductive Technologies",
    "Effects of Urbanization on Society",
    "Freedom of Speech in the Online World",
    "Privacy Concerns in the Age of Big Data",
    "Impact of Social Media on Youth",
    "The Role of Arts in Promoting Change",
    "Racial and Gender Equality in the Workplace",
    "The Power of Propaganda and Mass Media",
    "Importance of Cultural Diversity",
    "The Ethics of Animal Testing",
    "Youth Participation in Politics",
    "Aging Population and Healthcare Challenges",
    "The Influence of Pop Culture on Society",
    "Benefits and Risks of Nanotechnology",
    "The Role of Family in Shaping Individual Identity",
    "Economic Inequality and its Effects",
    "Education Reform in the 21st Century",
    "Pros and Cons of AI in Education",
    "The Impact of Colonialism on Indigenous Peoples",
    "Challenges of Mental Health in the Elderly",
    "Social Responsibility of Corporations",
    "The Rise of Virtual Reality: Opportunities and Concerns",
    "Censorship in the Arts and Media",
    "Impacts of Mass Tourism on Local Cultures",
    "Role of Women in Leadership Positions",
    "Ethical Consumption and its Influence",
    "The Digital Divide: Access to Technology",
    "Effects of Video Games on Behavior and Cognition",
    "The Changing Dynamics of Friendship in the Digital Era",
    "The Philosophy of Happiness and Well-being",
    "Understanding the Gig Economy",
    "The Role of Government in Promoting Healthy Lifestyles",
    "Exploring Alternative Economic Systems",
    "Criminal Justice Reform and Rehabilitation",
    "The Art of Persuasion: Media and Advertising"
]

PAGE_SIZE = 20


class Procedures:
    ENCODING = "Encoding"
    DECODING = "Decoding"
    values = [ENCODING, DECODING]


class DBMethods:
    INSERT = "insert"
    UPDATE = "update"
    DELETE = "delete"
    values = [INSERT, UPDATE, DELETE]


PASSWORD_HASH = """
49c958022a2c126e7d0f624fb0f9e048b7a277c8275aed85de10719cdf25c7c8f
2da117feb40463a46bd976fa2b7260b36eb39518258319d3b753b0534016b94""".replace("\n", "")
