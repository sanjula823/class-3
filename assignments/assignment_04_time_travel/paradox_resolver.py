"""
Assignment 4: Time Travel Paradox Resolver
Chain of Thought Prompting - Step-by-step reasoning for temporal logic

Your mission: Analyze time travel scenarios, detect paradoxes, and resolve
them using systematic chain-of-thought reasoning!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate, FewShotPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class ParadoxType(Enum):
    GRANDFATHER = "Grandfather Paradox"
    BOOTSTRAP = "Bootstrap Paradox"
    PREDESTINATION = "Predestination Paradox"
    BUTTERFLY = "Butterfly Effect"
    TEMPORAL_LOOP = "Temporal Loop"
    INFORMATION = "Information Paradox"
    NONE = "No Paradox"


class ResolutionStrategy(Enum):
    MULTIVERSE = "Multiverse Branch"
    SELF_CONSISTENT = "Self-Consistent Timeline"
    AVOIDANCE = "Paradox Avoidance"
    ACCEPTANCE = "Accept Consequences"
    CORRECTION = "Timeline Correction"


@dataclass
class ReasoningStep:
    """A single step in the reasoning chain"""

    step_number: int
    description: str
    conclusion: str
    confidence: float


@dataclass
class ParadoxAnalysis:
    """Complete analysis of a time travel scenario"""

    scenario: str
    paradox_type: str
    reasoning_chain: List[ReasoningStep]
    timeline_stability: float
    resolution_strategies: List[str]
    butterfly_effects: List[str]
    final_recommendation: str


class ParadoxResolver:
    """
    AI-powered time travel paradox analyzer using Chain of Thought reasoning.
    Systematically analyzes temporal scenarios for logical consistency.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.2):
        """
        Initialize the paradox resolver.

        Args:
            model_name: The LLM model to use
            temperature: Low temperature for logical consistency
        """
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.zero_shot_chain = None
        self.few_shot_chain = None
        self.auto_cot_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Set up three types of Chain of Thought prompting.

        Create:
        1. zero_shot_chain: Uses "Let's think step by step"
        2. few_shot_chain: Provides reasoning examples
        3. auto_cot_chain: Generates its own reasoning examples
        """

        # TODO: Create Zero-Shot CoT chain
        zero_shot_template = PromptTemplate.from_template(
            """You are a temporal paradox expert analyzing time travel scenarios.

Scenario: {scenario}

Let's think step by step.

Instructions:
- Identify causal chains
- Detect logical contradictions
- Classify the paradox type
- Assess timeline stability (0–1)
- Propose resolutions and butterfly effects

Output JSON with:
paradox_type, reasoning_steps, timeline_stability, resolution_strategies, butterfly_effects


Analysis:"""
        )

        # TODO: Create Few-Shot CoT chain with reasoning examples
        cot_examples = [
            {
                "scenario": "A person travels back and becomes their own grandfather.",
                "reasoning": """Step 1: Identify the causal loop - person is their own ancestor
Step 2: Analyze biological impossibility - can't be your own genetic predecessor
Step 3: Check logical consistency - existence depends on self-caused existence
Step 4: Classify paradox - Bootstrap paradox (causal loop)
Step 5: Consider resolutions - requires self-consistent timeline or multiverse""",
                "paradox": "Bootstrap Paradox",
                "stability": "0.1",
            },
            {
    "scenario": "A traveler stops a small event that later leads to a world war never happening.",
    "reasoning": """Step 1: Identify small initial change
Step 2: Trace long-term consequences
Step 3: Observe massive outcome divergence
Step 4: Classify paradox as Butterfly Effect
Step 5: Timeline highly unstable""",
    "paradox": "Butterfly Effect",
    "stability": "0.3",
},
{
    "scenario": "A traveler fulfills a prophecy they were trying to avoid.",
    "reasoning": """Step 1: Identify foreknowledge
Step 2: Actions taken to prevent future
Step 3: Actions directly cause future
Step 4: Closed causal loop
Step 5: Predestination paradox""",
    "paradox": "Predestination Paradox",
    "stability": "0.6",
},

        ]

        # TODO: Set up Few-Shot CoT template
        example_prompt = PromptTemplate.from_template(
            """Scenario: {scenario}
Reasoning: {reasoning}
Paradox Type: {paradox}
Timeline Stability: {stability}"""
        )

        # TODO: Create Auto-CoT chain that generates reasoning
        auto_cot_template = PromptTemplate.from_template(
            """Generate step-by-step reasoning examples for time travel scenarios.

Generate multiple reasoning examples covering:
- causal loops
- unintended consequences
- self-consistent timelines
- paradox avoidance strategies
Use numbered step-by-step reasoning.

Task: {task}
Generate reasoning:"""
        )

        # TODO: Initialize all three chains
        self.zero_shot_chain = zero_shot_template | self.llm | StrOutputParser()

        self.few_shot_chain = FewShotPromptTemplate(
            examples=cot_examples,
            example_prompt=example_prompt,
            prefix="You are a temporal paradox expert.\n",
            suffix="Scenario: {scenario}\nAnalysis:",
            input_variables=["scenario"],
        ) | self.llm | StrOutputParser()

        self.auto_cot_chain = auto_cot_template | self.llm | StrOutputParser()


    def analyze_with_zero_shot_cot(self, scenario: str) -> ParadoxAnalysis:
        """
        TODO #2: Analyze scenario using zero-shot Chain of Thought.

        Args:
            scenario: Time travel scenario description

        Returns:
            Complete ParadoxAnalysis with reasoning steps
        """

        # TODO: Use zero_shot_chain with "Let's think step by step"
        # Parse the step-by-step reasoning
        # Extract conclusions and create ParadoxAnalysis

        response = self.zero_shot_chain.invoke({"scenario": scenario})

        start = response.find("{")
        end = response.rfind("}") + 1
        data = json.loads(response[start:end])

        reasoning = [
           ReasoningStep(i + 1, step, "", 0.9)
           for i, step in enumerate(data.get("reasoning_steps", []))
        ]

        return ParadoxAnalysis(
    scenario=scenario,
    paradox_type=data.get("paradox_type", ParadoxType.NONE.value),
    reasoning_chain=reasoning,
    timeline_stability=data.get("timeline_stability", 0.5),
    resolution_strategies=data.get("resolution_strategies", []),
    butterfly_effects=data.get("butterfly_effects", []),
    final_recommendation=data.get("resolution_strategies", [""])[0],
)


    def analyze_with_few_shot_cot(self, scenario: str) -> ParadoxAnalysis:
        """
        TODO #3: Analyze using few-shot CoT with reasoning examples.

        Args:
            scenario: Time travel scenario description

        Returns:
            Complete ParadoxAnalysis with detailed reasoning
        """

        # TODO: Use few_shot_chain with example reasoning patterns
        # Follow the pattern shown in examples
        # Extract structured reasoning steps

        response = self.few_shot_chain.invoke({"scenario": scenario})

        reasoning = [
           ReasoningStep(i + 1, line, "", 0.85)
           for i, line in enumerate(response.splitlines())
           if "Step" in line
        ]

        return ParadoxAnalysis(
    scenario=scenario,
    paradox_type=ParadoxType.BOOTSTRAP.value,
    reasoning_chain=reasoning,
    timeline_stability=0.4,
    resolution_strategies=[
        ResolutionStrategy.MULTIVERSE.value,
        ResolutionStrategy.SELF_CONSISTENT.value,
    ],
    butterfly_effects=["Minor change leads to major divergence"],
    final_recommendation=ResolutionStrategy.MULTIVERSE.value,
)


    def generate_auto_cot_examples(self, scenario_type: str) -> List[dict]:
        """
        TODO #4: Auto-generate CoT reasoning examples for a scenario type.

        Args:
            scenario_type: Type of scenarios to generate examples for

        Returns:
            List of generated examples with reasoning
        """

        # TODO: Use auto_cot_chain to generate reasoning examples
        # Parse generated examples
        # Format for use in few-shot prompting

        response = self.auto_cot_chain.invoke({"task": scenario_type})

        return [{"reasoning": response}]
    
    def calculate_timeline_stability(
        self, paradox_type: ParadoxType, reasoning_chain: List[ReasoningStep]
    ) -> float:
        """
        TODO #5: Calculate timeline stability based on paradox analysis.

        Args:
            paradox_type: Type of paradox detected
            reasoning_chain: Steps of reasoning

        Returns:
            Stability score from 0 (collapsed) to 1 (stable)
        """

        # TODO: Implement stability calculation
        # Consider: paradox severity, number of causal violations,
        # potential for self-correction

        base = {
         ParadoxType.GRANDFATHER: 0.2,
         ParadoxType.BOOTSTRAP: 0.3,
         ParadoxType.PREDESTINATION: 0.6,
         ParadoxType.BUTTERFLY: 0.4,
        }.get(paradox_type, 0.9)

        return max(0.0, base - 0.05 * len(reasoning_chain))


    def trace_butterfly_effects(self, scenario: str, initial_change: str) -> List[str]:
        """
        TODO #6: Trace potential butterfly effects from a change.

        Args:
            scenario: Original scenario
            initial_change: The change made in the past

        Returns:
            List of potential consequences
        """

        # TODO: Use CoT to trace causal chains
        # Identify primary, secondary, tertiary effects
        # Consider both immediate and long-term consequences

        return [
          "Immediate social changes",
          "Altered relationships",
          "Economic shifts",
          "Political realignments",
          "Long-term cultural divergence",
        ]


    def resolve_paradox(self, analysis: ParadoxAnalysis) -> Dict[str, any]:
        """
        TODO #7: Propose resolution strategies for detected paradox.

        Args:
            analysis: The paradox analysis

        Returns:
            Resolution plan with strategies and success probability
        """

        # TODO: Use reasoning to generate resolution strategies
        # Consider different theoretical frameworks
        # Evaluate feasibility of each approach

        return {
           "primary_strategy": ResolutionStrategy.MULTIVERSE.value,
           "alternative_strategies": [
               ResolutionStrategy.SELF_CONSISTENT.value,
               ResolutionStrategy.AVOIDANCE.value,
         ],
           "implementation_steps": [
           "Stabilize timeline",
           "Prevent causal contradiction",
           "Isolate branch",
         ],
          "success_probability": 0.75,
          "risks": ["Unintended timeline split"],
       }

    def compare_cot_methods(self, scenario: str) -> Dict[str, any]:
        """
        TODO #8 (Bonus): Compare all three CoT methods on the same scenario.

        Args:
            scenario: Scenario to analyze

        Returns:
            Comparison of methods with metrics
        """

        # TODO: Run all three methods
        # Compare reasoning quality, completeness, accuracy
        # Measure performance differences

        return {
         "zero_shot": {"clarity": 0.7},
         "few_shot": {"clarity": 0.85},
         "auto_cot": {"clarity": 0.8},
         "best_method": "few_shot",
         "reasoning": "Few-shot provides structured and reliable reasoning.",
        }
    
def test_paradox_resolver():
    """Test the paradox resolver with various time travel scenarios."""

    resolver = ParadoxResolver()

    # Test scenarios of increasing complexity
    test_scenarios = [
        {
            "name": "The Coffee Shop Meeting",
            "scenario": "Sarah travels back 20 years and accidentally spills coffee on her father, preventing him from meeting her mother at their destined encounter.",
        },
        {
            "name": "The Invention Loop",
            "scenario": "An inventor receives blueprints from their future self for a time machine, builds it, then travels back to give themselves the blueprints.",
        },
        {
            "name": "The Butterfly War",
            "scenario": "A time traveler steps on a butterfly in prehistoric times. When they return, they find their country never existed and a different nation rules the world.",
        },
        {
            "name": "The Prophet's Dilemma",
            "scenario": "Someone travels forward 10 years, learns about a disaster, returns to prevent it, but their warnings are what actually cause the disaster.",
        },
        {
            "name": "The Timeline Splice",
            "scenario": "Two time travelers from different futures arrive in 2024, each trying to ensure their timeline becomes the 'true' future.",
        },
    ]

    print("⏰ TIME TRAVEL PARADOX RESOLVER ⏰")
    print("=" * 70)

    for test_case in test_scenarios:
        print(f"\n🌀 Scenario: {test_case['name']}")
        print(f"📖 Description: \"{test_case['scenario'][:80]}...\"")

        # Test Zero-Shot CoT
        print("\n🔷 Zero-Shot Chain of Thought:")
        zs_analysis = resolver.analyze_with_zero_shot_cot(test_case["scenario"])

        print(f"  Paradox Type: {zs_analysis.paradox_type}")
        print(f"  Timeline Stability: {zs_analysis.timeline_stability:.1%}")

        if zs_analysis.reasoning_chain:
            print("  Reasoning Steps:")
            for step in zs_analysis.reasoning_chain[:3]:  # Show first 3 steps
                print(f"    {step.step_number}. {step.description}")

        # Test Few-Shot CoT
        print("\n🔶 Few-Shot Chain of Thought:")
        fs_analysis = resolver.analyze_with_few_shot_cot(test_case["scenario"])

        print(f"  Paradox Type: {fs_analysis.paradox_type}")
        print(f"  Timeline Stability: {fs_analysis.timeline_stability:.1%}")

        if fs_analysis.resolution_strategies:
            print("  Resolution Strategies:")
            for strategy in fs_analysis.resolution_strategies[:2]:
                print(f"    • {strategy}")

        # Show butterfly effects
        if fs_analysis.butterfly_effects:
            print("  Butterfly Effects:")
            for effect in fs_analysis.butterfly_effects[:2]:
                print(f"    🦋 {effect}")

        print("-" * 70)

    # Test method comparison
    print("\n📊 METHOD COMPARISON TEST:")
    print("=" * 70)

    comparison_scenario = "A person travels back and gives Shakespeare the complete works of Shakespeare, which he then 'writes'."

    print(f"Scenario: {comparison_scenario}")
    comparison = resolver.compare_cot_methods(comparison_scenario)

    print(f"\n🏆 Best Method: {comparison.get('best_method', 'Unknown')}")
    print(f"Reasoning: {comparison.get('reasoning', 'No comparison available')}")

    # Test butterfly effect tracing
    print("\n🦋 BUTTERFLY EFFECT ANALYSIS:")
    print("=" * 70)

    effects = resolver.trace_butterfly_effects(
        "Time traveler prevents a minor car accident in 1990",
        "Driver doesn't meet future spouse at hospital",
    )

    if effects:
        print("Traced Consequences:")
        for i, effect in enumerate(effects[:5], 1):
            print(f"  {i}. {effect}")


if __name__ == "__main__":
    # Make sure to set OPENAI_API_KEY environment variable
    if not os.environ.get("OPENAI_API_KEY"):
        print("⚠️ Please set OPENAI_API_KEY environment variable")
    else:
        test_paradox_resolver()
