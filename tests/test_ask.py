from types import SimpleNamespace

from app import ask


def test_chat_keeps_using_the_same_conversation(monkeypatch, capsys):
    questions = iter(["最初の質問", "続けて質問", "/exit"])

    class FakeConversation:
        def __init__(self):
            self.messages = []

        def send_message(self, message):
            self.messages.append(message)
            return SimpleNamespace(text="回答", usage_metadata=None)

    conversation = FakeConversation()
    created = {}

    class FakeChats:
        def create(self, *, model, config):
            created["model"] = model
            created["config"] = config
            return conversation

    client = SimpleNamespace(chats=FakeChats())
    monkeypatch.setattr(ask.genai, "Client", lambda api_key: client)
    monkeypatch.setattr(ask, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", lambda prompt: next(questions))

    ask.chat()

    assert conversation.messages == ["最初の質問", "続けて質問"]
    assert created["model"] == ask.MODEL
    assert created["config"].temperature == ask.TEMPERATURE
