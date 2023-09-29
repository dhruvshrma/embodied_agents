import pytest

from personas.generate_personas import generate_persona
from personas.Persona import Persona


# add template as pytest fixture
@pytest.fixture
def template():
    template = {
        "traits": {
            "personality": ["introvert", "extrovert"],
            "temperament": ["melancholic", "choleric", "sanguine", "phlegmatic"],
            "interests": [
                "sports",
                "music",
                "art",
                "politics",
                "science",
                "technology",
            ],
        },
        "status": ["student", "employed", "unemployed", "retired"],
    }
    return template


def test_generate_persona(template):
    persona = generate_persona(template)
    assert isinstance(persona, Persona)
    assert isinstance(persona.name, str)
    assert isinstance(persona.age, int)
    assert isinstance(persona.traits, str)
    assert isinstance(persona.status, str)
    assert persona.age >= 18
    assert persona.age <= 80
    assert persona.status in template["status"]

    generated_traits = persona.traits.split(
        ", "
    )  # Split the traits string into individual traits
    for trait in generated_traits:
        # Check that each trait belongs to one of the trait categories in the template
        assert any(trait in trait_list for trait_list in template["traits"].values())
