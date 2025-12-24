"""
Assignment 6: Alien Language Translator
Few-Shot + Chain of Thought - Decode alien messages using examples and reasoning

Your mission: First contact! Decode alien communications using pattern
recognition and logical deduction!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


@dataclass
class Translation:
    alien_text: str
    human_text: str
    confidence: float
    reasoning_steps: List[str]
    cultural_notes: str


class AlienTranslator:
    """
    AI-powered alien language translator using few-shot examples and CoT reasoning.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.3)
        self.translation_examples = self._load_examples()
        self.decoder_chain = None
        self._setup_chains()

    def _load_examples(self) -> List[dict]:
        """
        TODO #1: Create example alien translations with reasoning.

        Include: symbols, translation, step-by-step decoding logic
        """

        examples = [
            {
                "alien": "◈◈◈ ▲▲ ●",
                "reasoning": "Step 1: ◈◈◈ appears to be quantity (3 symbols)\nStep 2: ▲▲ represents object type\nStep 3: ● is singular marker\nStep 4: Pattern suggests 'three ships approaching'",
                "translation": "Three ships approaching",
                "pattern": "quantity-object-verb",
            },
            # TODO: Add more examples with reasoning chains
            {
    "alien": "♦♦ ◯◯◯ ▼",
    "reasoning": "Step 1: ♦♦ indicates a pair\nStep 2: ◯◯◯ refers to energy units\nStep 3: ▼ implies descent or landing\nStep 4: Message means 'two energy pods landing'",
    "translation": "Two energy pods landing",
    "pattern": "quantity-object-action",
},

        ]

        return examples

    def _setup_chains(self):
        """
        TODO #2: Create few-shot CoT chain for translation.

        Combine pattern examples with reasoning steps.
        """

        example_prompt = PromptTemplate(
        input_variables=["alien", "reasoning", "translation"],
        template=(
        "Alien message: {alien}\n"
        "Reasoning:\n{reasoning}\n"
        "Human translation: {translation}\n"
    ),
)

        few_shot_prompt = FewShotPromptTemplate(
    examples=self.translation_examples,
    example_prompt=example_prompt,
    prefix=(
        "You are an expert alien language translator.\n"
        "Use the examples below to decode new alien messages.\n"
        "Explain reasoning step by step, then give the translation.\n\n"
    ),
    suffix=(
        "Alien message: {alien_message}\n"
        "Reasoning:\n"
    ),
    input_variables=["alien_message"],
)

        self.decoder_chain = few_shot_prompt | self.llm


    def translate(self, alien_message: str) -> Translation:
        """
        TODO #3: Translate alien message using examples and reasoning.

        Args:
            alien_message: Message to decode

        Returns:
            Translation with reasoning
        """

        response = self.decoder_chain.invoke(
    {"alien_message": alien_message}
)

        text = response.content.strip()

        # Simple parsing (safe for assignment)
        lines = text.split("\n")
        reasoning_steps = [line for line in lines if line.lower().startswith("step")]
        translation_line = next(
            (line for line in lines if "translation" in line.lower()),
            "",
       )

        human_text = translation_line.split(":", 1)[-1].strip() if ":" in translation_line else text

        return Translation(
    alien_text=alien_message,
    human_text=human_text,
    confidence=0.7,
    reasoning_steps=reasoning_steps,
    cultural_notes="Likely references shared alien cultural symbols.",
)



def test_translator():
    translator = AlienTranslator()

    test_messages = ["◈◈◈◈◈ ▲▲▲ ● ◆", "♦♦ ◯◯◯ ▼ ★★★★", "△△△ ◈ ■■ ◆◆◆"]

    print("👽 ALIEN LANGUAGE TRANSLATOR 👽")
    print("=" * 70)

    for msg in test_messages:
        result = translator.translate(msg)
        print(f"\nAlien: {msg}")
        print(f"Translation: {result.human_text}")
        print(f"Confidence: {result.confidence:.0%}")
        print("-" * 70)


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_translator()
