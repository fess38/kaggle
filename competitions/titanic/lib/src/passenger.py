from dataclasses import dataclass

PassengerId = int


@dataclass
class Passenger:
    id: PassengerId
    pclass: int
    name: str
    sex: str
    age: float | None
    sib_sp: int
    parch: int
    ticket: str
    fare: float | None
    cabin: str | None
    embarked: str | None


@dataclass
class Label:
    id: PassengerId
    label: bool
