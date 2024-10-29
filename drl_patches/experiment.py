from dataclasses import dataclass
from enum import Enum
from typing import Union

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

Embedding = Union[torch.tensor, np.array]


class PatchValidator:
    """
    Responsible for validating the patch with the framework of static code analysis.
    "AWS KUBE"????
    """

    def __init__(self) -> None:
        pass

    def calculate(self, patch: str) -> bool:
        """
        Validate the patch with the static code analysis framework.
        Returns True if the patch is valid, False otherwise.
        """
        pass


class PatchEnvironment:
    """
    The environment that the agent interacts with, providing states, actions, and rewards.
    """

    def __init__(self, vulnerability_info: str, code_snippets: DataLoader[str]) -> None:
        self.vulnerability_info = vulnerability_info
        self.code_snippets = code_snippets
        self.patch_validator = PatchValidator()

    def step(self, action: str) -> tuple[str, float, bool]:
        """
        Executes the action, validates the patch, and calculates the reward.
        Returns a new state, reward, and a done flag indicating the end of an episode.
        """
        # Validate the action (patch) using the static code analysis framework
        is_valid = self.patch_validator.calculate(action)

        # Calculate reward based on the validation
        reward = (
            self._calculate_reward(action) if is_valid else -1.0
        )  # Penalize invalid patches

        # Define if the episode is done (could be based on reward threshold or other criteria)
        done = reward > 0.8  # Example threshold

        # Return the new state (e.g., next code snippet), reward, and done flag
        next_state = "Next code snippet or prompt embedding here"
        return next_state, reward, done

    def reset(self) -> str:
        """
        Reset the environment and return the initial state.
        """
        return "Initial state or prompt"

    def _calculate_reward(self, patch: str) -> float:
        """
        Reward calculation for a generated patch.
        """
        return self.patch_validator.calculate(patch)  # TODO(Rui): Something something


class MutationType(Enum, str):
    GENERATE = "generate"
    CROSSOVER = "crossover"
    EXPAND = "expand"
    SHORTEN = "shorten"
    REPHRASE = "rephrase"


class Mutator:
    """
    Responsible for mutating the patch.
    Can be wither "generate", "crossover", "expand", "shorten", "rephrase"

    """

    @classmethod
    def mutate(cls, patch: str, mutation_type: MutationType) -> str:
        """
        Mutate the patch.

        # Make a call to Llama or something to change the prompt based on the mutation type!
        # Each mutation type will have a different way of changing the prompt -> different inintal prompt for each
        """
        pass


@dataclass
class AgentState:
    current_prompt: Union[
        torch.tensor, np.array
    ]  # Current prompt for the agent (Embedding)
    last_patch: Union[
        torch.tensor, np.array
    ]  # Last response from the agent (Embedding)


class PatchAgent:
    """
    The agent responsible for generating patches using a DRL-based policy.
    """

    def __init__(self, model: AutoTokenizer, tokenizer: AutoModelForCausalLM) -> None:
        self.tokenizer = model
        self.model = tokenizer
        self.state = None

    def act(self, state: str) -> str:
        """
        Decide on an action based on the current state.
        """
        # Use a transformer model to generate a patch based on state
        inputs = self.tokenizer(state, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def update_policy(self, reward: float):
        """
        Update the agent's policy based on the received reward (e.g., using policy gradients).
        """
        # Integrate your preferred RL algorithm (e.g., policy gradients or DDPG) here
        pass


class AdversarialPatchFramework:
    """
    The DRL framework for training the agent to generate adversarial patches.
    """

    def __init__(
        self, vulnerability_info: str, code_snippets: DataLoader[str], model_name: str
    ) -> None:
        self.env = PatchEnvironment(vulnerability_info, code_snippets)
        self.agent = PatchAgent(model_name=model_name)

    def run(self, max_steps: int) -> None:
        """
        Run the adversarial patch generation for the specified number of steps.
        """
        for episode in range(max_steps):
            state = self.env.reset()  # Initial state
            done = False
            cumulative_reward = 0.0

            while not done:
                # Agent selects an action based on the state
                action = self.agent.act(state)

                # Environment executes the action and returns the next state, reward, and done flag
                next_state, reward, done = self.env.step(action)

                # Update the agent's policy based on the reward
                self.agent.update_policy(reward)

                # Accumulate reward
                cumulative_reward += reward

                # Update state
                state = next_state

            print(
                f"Episode {episode + 1}/{max_steps}, Total Reward: {cumulative_reward}"
            )
