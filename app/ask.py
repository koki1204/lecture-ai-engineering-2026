"""第1回: 素のLLMに質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 講座に固有の既定質問をする
    uv run python -m app.ask --chat     # 会話履歴を保って対話する

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse

from google import genai
from google.genai import errors
from google.genai import types

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
TEMPERATURE = 0.7


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    response = client.models.generate_content(
        model=MODEL,
        contents=question,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )

    usage = response.usage_metadata
    if usage is not None:
        print(
            f"[token] 入力: {usage.prompt_token_count}, "
            f"出力: {usage.candidates_token_count}"
        )

    return response.text


def chat() -> None:
    client = genai.Client(api_key=get_api_key())
    conversation = client.chats.create(
        model=MODEL,
        config=types.GenerateContentConfig(temperature=TEMPERATURE),
    )

    print("[会話] /exit または /quit で終了します。")
    while True:
        try:
            question = input("あなた> ").strip()
        except EOFError:
            print()
            return

        if question.lower() in {"/exit", "/quit"}:
            return
        if not question:
            continue

        response = conversation.send_message(question)
        usage = response.usage_metadata
        if usage is not None:
            print(
                f"[token] 入力: {usage.prompt_token_count}, "
                f"出力: {usage.candidates_token_count}"
            )
        print(f"AI> {response.text}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    parser.add_argument("--chat", action="store_true", help="会話履歴を保って対話する")
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            if args.question != DEFAULT_QUESTION:
                parser.error("対話モードでは質問を指定しないでください")
            chat()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
