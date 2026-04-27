from __future__ import annotations


class IntentClassifier:
    """Conservative pipeline/chat classifier."""

    def __init__(self, pipeline_keywords: set[str], chat_keywords: set[str]) -> None:
        self.pipeline_keywords = pipeline_keywords
        self.chat_keywords = chat_keywords

    def classify(self, text: str) -> str:
        lower = text.lower()
        words = set(lower.split())

        if words & self.pipeline_keywords:
            return "pipeline"

        if any(kw in lower for kw in self.pipeline_keywords if len(kw) >= 3):
            return "pipeline"

        if any(c in lower for c in ("/", "\\", ".csv", ".xlsx", ".json", "input/", "output/")):
            return "pipeline"

        if len(words) <= 3 and any(kw in lower for kw in self.chat_keywords):
            return "chat"

        return "pipeline"

