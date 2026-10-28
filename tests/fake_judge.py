import json
from types import SimpleNamespace


class FakeJudgeClient:
    """Stands in for the OpenAI client: plays back JSON answers in order."""

    def __init__(self, answers: list[dict]) -> None:
        self.answers = list(answers)
        self.calls: list[dict] = []
        self.responses = self  # so client.responses.create() lands on create()

    def create(self, **kwargs) -> SimpleNamespace:
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=json.dumps(self.answers.pop(0)))
