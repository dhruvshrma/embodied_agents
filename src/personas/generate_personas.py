import random

from faker import Faker
from pydantic import ValidationError
from personas.Persona import Persona

fake = Faker()

base_template = {
    "traits": {
        "personality": ["introvert", "extrovert"],
        "temperament": ["open-minded", "closed-minded"],
        "change_orientation": ["change-oriented", "status-quo", "flexible"],
        "opinions_description": [
            "strongly held",
            "weakly held",
            "fickle-minded",
        ],
        "interests": [
            "sports",
            "music",
            "art",
            "policy",
            "science",
            "technology",
        ],
    },
    "status": ["student", "employed", "unemployed", "retired"],
}


def generate_persona(template):
    persona_data = {"name": fake.name(), "age": random.randint(18, 80)}

    traits = []
    for category, trait_list in template["traits"].items():
        traits.append(random.choice(trait_list))
    persona_data["traits"] = ", ".join(traits)

    persona_data["status"] = random.choice(template["status"])

    try:
        return Persona(**persona_data)
    except ValidationError as e:
        print(f"Validation error: {e}")
        return None
