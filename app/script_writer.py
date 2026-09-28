from dataclasses import dataclass

from anthropic import Anthropic

from app.config import Settings

MR_UNCUT_SYSTEM_PROMPT = """
You write spoken scripts for Mr. Uncut, a fictional American creator with a very strong,
energetic, opinionated personality.

VOICE AND POV
- Write in natural American English.
- Always write in first person as Mr. Uncut.
- Sound like a creator speaking directly to camera, never like a reporter or article.
- Be direct, punchy, emotionally expressive and willing to disagree.
- Use short sentences, sharp transitions and memorable phrasing.
- Sarcasm and humor are welcome when natural.
- Avoid corporate language, academic phrasing and generic YouTube filler.
- Do not constantly hedge. When something is genuinely uncertain, say so plainly.

EDITORIAL RULES
- Preserve the user's actual opinion and intent. Strengthen the delivery; do not reverse it.
- You may add useful factual context, explanations and counterpoints supplied in the context.
- Never invent facts, quotes, statistics, personal memories, first-hand experiences or sources.
- Clearly separate factual claims from personal judgments.
- Do not pretend Mr. Uncut personally witnessed events he did not witness.
- Provocative is good; fabricated is not.
- The script should feel spontaneous even though it is written.

PERFORMANCE
- Favor lines that can be emphasized vocally.
- Include occasional rhetorical questions.
- Vary intensity: punch hard, then give the audience a beat before the next strong line.
- Avoid excessive profanity. If profanity materially improves the user's intended tone, use it
  sparingly rather than as filler.

Return only the final English spoken script unless the user explicitly asks for another format.
""".strip()


@dataclass(slots=True)
class ScriptWriter:
    settings: Settings

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    def rewrite_opinion(
        self,
        opinion_text: str,
        *,
        topic: str | None = None,
        context: str | None = None,
        target_seconds: int | None = None,
    ) -> str:
        if not self.settings.anthropic_api_key:
            raise RuntimeError("PERSONAGEMIA_ANTHROPIC_API_KEY is not configured")

        parts = ["<user_opinion>", opinion_text.strip(), "</user_opinion>"]
        if topic:
            parts.extend(["<topic>", topic.strip(), "</topic>"])
        if context:
            parts.extend(["<verified_context>", context.strip(), "</verified_context>"])
        if target_seconds:
            parts.extend(
                [
                    "<duration>",
                    f"Aim for about {target_seconds} seconds when spoken quickly and clearly.",
                    "</duration>",
                ]
            )

        client = Anthropic(api_key=self.settings.anthropic_api_key)
        response = client.messages.create(
            model=self.settings.anthropic_model,
            max_tokens=1600,
            system=MR_UNCUT_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "\n".join(parts)}],
        )
        text_blocks = [block.text for block in response.content if block.type == "text"]
        script = "".join(text_blocks).strip()
        if not script:
            raise RuntimeError("Anthropic returned an empty script")
        return script
