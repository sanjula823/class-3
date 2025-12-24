"""
Assignment 8: Superhero Power Balancer
All Concepts Combined - Master all prompting techniques together

Your mission: Balance superhero powers for the ultimate fighting game
using every prompting technique you've learned!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate


class PowerType(Enum):
    PHYSICAL = "physical"
    ENERGY = "energy"
    MENTAL = "mental"
    REALITY = "reality"
    TECH = "technology"
    MAGIC = "magic"


@dataclass
class Hero:
    name: str
    abilities: List[str]
    power_type: str
    power_level: float
    weaknesses: List[str]
    synergies: List[str]


@dataclass
class BalanceReport:
    hero: Hero
    analysis_method: str  # Which prompting method was used
    power_rating: float
    balance_issues: List[str]
    suggested_changes: List[str]
    team_synergies: Dict[str, float]
    counter_picks: List[str]


class PowerBalancer:
    """
    AI-powered game balancer using all prompting techniques.
    Combines zero-shot, few-shot, and CoT for comprehensive analysis.
    """

    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.llm = ChatOpenAI(model=model_name, temperature=0.4)
        self.ability_analyzer = None  # Zero-shot
        self.type_classifier = None  # Few-shot
        self.interaction_calculator = None  # CoT
        self.balance_detector = None  # Combined
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up chains for each prompting technique.

        Create:
        1. Zero-shot for novel ability analysis
        2. Few-shot for power type classification
        3. CoT for interaction calculations
        4. Combined for balance detection
        """

        # TODO: Zero-shot for ability analysis
        ability_template = PromptTemplate.from_template(
            """Analyze this superhero ability for game balance.

[TODO: Add zero-shot instructions for:
- Power level estimation
- Potential exploits
- Counter-play options]

Ability: {ability_description}

Analysis:"""
        )

        # TODO: Few-shot for power classification
        type_examples = [
            {
                "ability": "Super strength and invulnerability",
                "type": "physical",
                "reasoning": "Direct physical enhancement",
            },
            # TODO: Add more examples
        ]

        # TODO: CoT for interaction calculations
        interaction_template = PromptTemplate.from_template(
            """Calculate how these abilities interact in combat.

Ability 1: {ability1}
Ability 2: {ability2}

Let's think step by step about their interaction:"""
        )

        self.ability_analyzer = ability_template | self.llm
        self.interaction_calculator = interaction_template | self.llm

    def analyze_hero_zero_shot(self, hero: Hero) -> Dict[str, any]:
        """
        TODO #2: Analyze hero abilities using zero-shot prompting.

        For novel, unique abilities without examples.
        """

        results = []
        for ability in hero.abilities:
            response = self.ability_analyzer.invoke(
                 {"ability_description": ability}
             ).content
        results.append(response)

        return {
    "power_level": min(10.0, len(hero.abilities) * 3.0),
    "exploits": results,
    "counters": ["Team coordination", "Cooldown limits"],
}


    def classify_power_few_shot(self, abilities: List[str]) -> str:
        """
        TODO #3: Classify power type using few-shot examples.

        Match patterns from example heroes.
        """

        joined = " ".join(abilities).lower()
        if any(x in joined for x in ["mind", "telepathy", "memory"]):
              return PowerType.MENTAL.value
        if any(x in joined for x in ["time", "reality", "probability"]):
              return PowerType.REALITY.value
        return PowerType.PHYSICAL.value


    def calculate_synergy_cot(self, hero1: Hero, hero2: Hero) -> float:
        """
        TODO #4: Calculate team synergy using Chain of Thought.

        Step-by-step reasoning for ability interactions.
        """

        response = self.interaction_calculator.invoke(
    {
        "ability1": ", ".join(hero1.abilities),
        "ability2": ", ".join(hero2.abilities),
    }
).content

        return 0.8 if "synergy" in response.lower() else 0.5


    def detect_imbalance_combined(self, hero: Hero, meta: List[Hero]) -> BalanceReport:
        """
        TODO #5: Detect balance issues using ALL techniques.

        Orchestrate all methods for comprehensive analysis.
        """

        # TODO: Combine all techniques:
        # - Zero-shot for unique aspects
        # - Few-shot for patterns
        # - CoT for complex interactions

        analysis = self.analyze_hero_zero_shot(hero)
        power_rating = analysis["power_level"]

        issues = []
        if power_rating > 8:
         issues.append("Overpowered ability stacking")

        return BalanceReport(
    hero=hero,
    analysis_method="zero-shot + CoT",
    power_rating=power_rating,
    balance_issues=issues,
    suggested_changes=["Increase cooldowns", "Limit ability overlap"],
    team_synergies={h.name: self.calculate_synergy_cot(hero, h) for h in meta if h != hero},
    counter_picks=["High mobility heroes", "Silence abilities"],
)


    def auto_balance(self, hero: Hero, target_power: float) -> Hero:
        """
        TODO #6 (Bonus): Automatically adjust hero for target power level.

        Use all techniques to create balanced version.
        """

        # TODO: Implement auto-balancing

        return hero


def test_balancer():
    balancer = PowerBalancer()

    test_heroes = [
        Hero(
            name="Chronos",
            abilities=["Time manipulation", "Temporal loops", "Age acceleration"],
            power_type="reality",
            power_level=0.0,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Mindweaver",
            abilities=["Telepathy", "Illusion creation", "Memory manipulation"],
            power_type="mental",
            power_level=0.0,
            weaknesses=[],
            synergies=[],
        ),
        Hero(
            name="Quantum",
            abilities=["Teleportation", "Probability manipulation", "Phase shifting"],
            power_type="reality",
            power_level=0.0,
            weaknesses=[],
            synergies=[],
        ),
    ]

    print("⚡ SUPERHERO POWER BALANCER ⚡")
    print("=" * 70)

    for hero in test_heroes:
        print(f"\n🦸 Hero: {hero.name}")
        print(f"Abilities: {', '.join(hero.abilities)}")

        # Test zero-shot analysis
        analysis = balancer.analyze_hero_zero_shot(hero)
        print(f"Power Level: {analysis.get('power_level', 0):.1f}/10")

        # Test few-shot classification
        power_type = balancer.classify_power_few_shot(hero.abilities)
        print(f"Power Type: {power_type}")

        # Test CoT synergy
        if len(test_heroes) > 1:
            synergy = balancer.calculate_synergy_cot(hero, test_heroes[0])
            print(f"Synergy with {test_heroes[0].name}: {synergy:.0%}")

        print("-" * 70)

    # Test combined balance detection
    print("\n🎯 BALANCE ANALYSIS (All Techniques):")
    print("=" * 70)

    report = balancer.detect_imbalance_combined(test_heroes[0], test_heroes)

    print(f"Hero: {report.hero.name}")
    print(f"Analysis Method: {report.analysis_method}")
    print(f"Power Rating: {report.power_rating:.1f}/10")

    if report.balance_issues:
        print("Balance Issues:")
        for issue in report.balance_issues:
            print(f"  ⚠️ {issue}")

    if report.suggested_changes:
        print("Suggested Changes:")
        for change in report.suggested_changes:
            print(f"  ✓ {change}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY")
    else:
        test_balancer()
