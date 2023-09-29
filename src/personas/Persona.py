from pydantic import BaseModel, constr, conint


class Persona(BaseModel):
    name: constr(strip_whitespace=True, min_length=2, max_length=100)
    age: conint(ge=18, le=80)
    traits: str
    status: str
