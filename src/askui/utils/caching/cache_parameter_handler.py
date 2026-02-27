"""Cache parameter handling for trajectory recording and execution.

This module provides utilities for:
- Identifying dynamic values that should become parameters (recording phase)
- Validating and substituting parameter values (execution phase)

Cache parameters use the {{parameter_name}} syntax and allow dynamic values
to be injected during cache execution.
"""

import json
import logging
import re
from typing import TYPE_CHECKING, Any

from askui.models.shared.agent_message_param import MessageParam, ToolUseBlockParam
from askui.models.shared.prompts import SystemPrompt
from askui.prompts.caching import CACHING_PARAMETER_IDENTIFIER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from askui.model_providers.vlm_provider import VlmProvider

logger = logging.getLogger(__name__)

# Regex pattern for matching parameters: {{parameter_name}}
# Allows alphanumeric characters and underscores, must start with letter/underscore
CACHE_PARAMETER_PATTERN = r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}"


class CacheParameterDefinition:
    """Represents a cache parameter identified in a trajectory."""

    def __init__(self, name: str, value: Any, description: str) -> None:
        self.name = name
        self.value = value
        self.description = description

    def __repr__(self) -> str:
        return f"CacheParameterDefinition(name={self.name}, value={self.value})"


class CacheParameterHandler:
    """Handles all cache parameter operations for trajectory recording and execution."""

    # ========================================================================
    # RECORDING PHASE: Parameter identification and templatization
    # ========================================================================

    @staticmethod
    def identify_and_parameterize(
        trajectory: list[ToolUseBlockParam],
        goal: str | None,
        identification_strategy: str,
        vlm_provider: "VlmProvider | None" = None,
    ) -> tuple[str | None, list[ToolUseBlockParam], dict[str, str]]:
        """Identify parameters and return parameterized trajectory + goal.

        This is the main entry point for the recording phase. It orchestrates
        parameter identification and templatization of both trajectory and goal.

        Args:
            trajectory: The trajectory to analyze and parameterize
            goal: The goal text to parameterize (optional)
            identification_strategy: "llm" for AI-based or "preset" for manual
            vlm_provider: VlmProvider instance to use for LLM-based identification.
                The model to use is determined by vlm_provider.model_id.

        Returns:
            Tuple of:
            - Parameterized goal text (or None if no goal)
            - Parameterized trajectory (with {{param}} syntax)
            - Dict mapping parameter names to descriptions
        """
        if identification_strategy == "llm" and trajectory and vlm_provider is not None:
            # Use LLM to identify parameters
            parameters_dict, parameter_definitions = (
                CacheParameterHandler._identify_parameters_with_llm(
                    trajectory, vlm_provider
                )
            )

            if parameter_definitions:
                # Replace values with {{parameter}} syntax in trajectory
                parameterized_trajectory = (
                    CacheParameterHandler._replace_values_with_parameters(
                        trajectory, parameter_definitions
                    )
                )

                # Apply same replacement to goal text
                parameterized_goal = goal
                if goal:
                    parameterized_goal = (
                        CacheParameterHandler._apply_parameters_to_text(
                            goal, parameter_definitions
                        )
                    )

                n_parameters = len(parameter_definitions)
                logger.info("Replaced %s parameter values in trajectory", n_parameters)
                return parameterized_goal, parameterized_trajectory, parameters_dict

            # No parameters identified
            logger.info("No parameters identified in trajectory")
            return goal, trajectory, {}

        # Manual extraction (preset strategy)
        parameter_names = CacheParameterHandler.extract_parameters(trajectory)
        parameters_dict = {
            name: f"Parameter for {name}"
            for name in parameter_names  # Generic desc
        }
        n_parameters = len(parameter_names)
        logger.info("Extracted %s manual parameters from trajectory", n_parameters)
        return goal, trajectory, parameters_dict

    @staticmethod
    def _identify_parameters_with_llm(
        trajectory: list[ToolUseBlockParam],
        vlm_provider: "VlmProvider",
    ) -> tuple[dict[str, str], list[CacheParameterDefinition]]:
        """Identify parameters in a trajectory using LLM analysis.

        Args:
            trajectory: The trajectory to analyze (list of tool use blocks)
            vlm_provider: VlmProvider instance for LLM calls.
                The model used is determined by vlm_provider.model_id.

        Returns:
            Tuple of:
            - Dict mapping parameter names to descriptions
            - List of CacheParameterDefinition objects with name, value, description
        """
        if not trajectory:
            logger.debug("Empty trajectory provided, skipping parameter identification")
            return {}, []

        logger.info(
            "Starting parameter identification for trajectory with %s steps",
            len(trajectory),
        )

        # Convert trajectory to serializable format for analysis
        trajectory_data = [tool.model_dump(mode="json") for tool in trajectory]
        logger.debug("Converted %s tool blocks to JSON format", len(trajectory_data))

        user_message = (
            "Analyze this UI automation trajectory and identify all values that "
            "should be parameters:\n\n"
            f"```json\n{json.dumps(trajectory_data, indent=2)}\n```\n\n"
            "Return only the JSON object with identified parameters. "
            "Be thorough but conservative - only mark values that are clearly "
            "dynamic or time-sensitive."
        )

        response_text = ""  # Initialize for error logging
        try:
            # Make single API call using VlmProvider
            logger.debug(
                "Calling LLM (%s) to analyze trajectory for parameters",
                vlm_provider.model_id,
            )
            response = vlm_provider.create_message(
                messages=[MessageParam(role="user", content=user_message)],
                system=SystemPrompt(prompt=CACHING_PARAMETER_IDENTIFIER_SYSTEM_PROMPT),
                max_tokens=4096,
                temperature=0.0,  # Deterministic for analysis
            )
            logger.debug("Received response from LLM")

            # Extract text from response
            if isinstance(response.content, list):
                response_text = next(
                    (
                        block.text
                        for block in response.content
                        if hasattr(block, "text")
                    ),
                    "",
                )
            else:
                response_text = str(response.content)

            # Parse the JSON response
            logger.debug("Parsing LLM response to extract parameter definitions")
            # Handle markdown code blocks if present
            if "```json" in response_text:
                logger.debug("Removing JSON markdown code block wrapper from response")
                response_text = (
                    response_text.split("```json")[1].split("```")[0].strip()
                )
            elif "```" in response_text:
                logger.debug("Removing code block wrapper from response")
                response_text = response_text.split("```")[1].split("```")[0].strip()

            parameter_data = json.loads(response_text)
            logger.debug(
                "Successfully parsed JSON response with %s parameters",
                len(parameter_data.get("parameters", [])),
            )

            # Convert to our data structures
            parameter_definitions = [
                CacheParameterDefinition(
                    name=p["name"], value=p["value"], description=p["description"]
                )
                for p in parameter_data.get("parameters", [])
            ]

            parameters_dict = {p.name: p.description for p in parameter_definitions}

            if parameter_definitions:
                logger.info(
                    "Successfully identified %s parameters in trajectory",
                    len(parameter_definitions),
                )
                for p in parameter_definitions:
                    logger.debug("  - %s: %s (%s)", p.name, p.value, p.description)
            else:
                logger.info(
                    "No parameters identified in trajectory "
                    "(this is normal for trajectories with only static values)"
                )

        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse LLM response as JSON: %s. "
                "Falling back to empty parameter list.",
                e,
                extra={"response_text": response_text[:500]},  # Log first 500 chars
            )
            return {}, []
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Failed to identify parameters with LLM: %s. "
                "Falling back to empty parameter list.",
                e,
                exc_info=True,
            )
            return {}, []
        else:
            return parameters_dict, parameter_definitions

    @staticmethod
    def _replace_values_with_parameters(
        trajectory: list[ToolUseBlockParam],
        parameter_definitions: list[CacheParameterDefinition],
    ) -> list[ToolUseBlockParam]:
        """Replace actual values in trajectory with {{parameter_name}} syntax.

        This is the reverse of substitute_parameters - it takes identified values
        and replaces them with parameter syntax for saving to cache.

        Args:
            trajectory: The trajectory to templatize
            parameter_definitions: List of CacheParameterDefinition objects with
                name and value attributes

        Returns:
            New trajectory with values replaced by parameters
        """
        # Build replacement map: value -> parameter name
        replacements = {
            str(p.value): f"{{{{{p.name}}}}}" for p in parameter_definitions
        }

        # Apply replacements to each tool block
        parameterized_trajectory = []
        for tool_block in trajectory:
            parameterized_input = CacheParameterHandler._replace_values_in_value(
                tool_block.input, replacements
            )

            parameterized_trajectory.append(
                ToolUseBlockParam(
                    id=tool_block.id,
                    name=tool_block.name,
                    input=parameterized_input,
                    type=tool_block.type,
                    cache_control=tool_block.cache_control,
                )
            )

        return parameterized_trajectory

    @staticmethod
    def _apply_parameters_to_text(
        text: str, parameter_definitions: list[CacheParameterDefinition]
    ) -> str:
        """Apply parameter replacement to a text string (e.g., goal).

        Args:
            text: The text to parameterize
            parameter_definitions: List of parameter definitions

        Returns:
            Text with values replaced by {{parameter}} syntax
        """
        # Build replacement map: value -> parameter syntax
        replacements = {
            str(p.value): f"{{{{{p.name}}}}}" for p in parameter_definitions
        }
        # Sort by length descending to replace longer matches first
        result = text
        for actual_value in sorted(replacements.keys(), key=len, reverse=True):
            if actual_value in result:
                result = result.replace(actual_value, replacements[actual_value])
        return result

    @staticmethod
    def _replace_values_in_value(value: Any, replacements: dict[str, str]) -> Any:
        """Recursively replace actual values with parameter syntax.

        Args:
            value: Any value (str, dict, list, etc.) to process
            replacements: Dict mapping actual values to parameter syntax

        Returns:
            New value with replacements applied
        """
        if isinstance(value, str):
            # Replace exact matches and substring matches
            result = value
            # Sort by length descending to replace longer matches first
            # This prevents partial replacements
            for actual_value in sorted(replacements.keys(), key=len, reverse=True):
                if actual_value in result:
                    result = result.replace(actual_value, replacements[actual_value])
            return result
        if isinstance(value, dict):
            # Recursively replace in dict values
            return {
                k: CacheParameterHandler._replace_values_in_value(v, replacements)
                for k, v in value.items()
            }
        if isinstance(value, list):
            # Recursively replace in list items
            return [
                CacheParameterHandler._replace_values_in_value(item, replacements)
                for item in value
            ]
        # For non-string types, check if the value matches exactly
        str_value = str(value)
        if str_value in replacements:
            # Return the parameter as a string
            return replacements[str_value]
        return value

    # ========================================================================
    # EXECUTION PHASE: Parameter extraction, validation, and substitution
    # ========================================================================

    @staticmethod
    def extract_parameters(trajectory: list[ToolUseBlockParam]) -> set[str]:
        """Extract all parameter names from a trajectory.

        Scans all tool inputs for {{parameter_name}} patterns and returns
        a set of unique parameter names.

        Args:
            trajectory: List of tool use blocks to scan

        Returns:
            Set of unique parameter names found in the trajectory
        """
        parameters: set[str] = set()

        for step in trajectory:
            # Recursively find parameters in the input object
            parameters.update(CacheParameterHandler._extract_from_value(step.input))

        return parameters

    @staticmethod
    def _extract_from_value(value: Any) -> set[str]:
        """Recursively extract parameters from a value.

        Args:
            value: Any value (str, dict, list, etc.) to search for parameters

        Returns:
            Set of parameter names found
        """
        parameters: set[str] = set()

        if isinstance(value, str):
            # Find all matches in the string
            matches = re.finditer(CACHE_PARAMETER_PATTERN, value)
            parameters.update(match.group(1) for match in matches)
        elif isinstance(value, dict):
            # Recursively search dict values
            for v in value.values():
                parameters.update(CacheParameterHandler._extract_from_value(v))
        elif isinstance(value, list):
            # Recursively search list items
            for item in value:
                parameters.update(CacheParameterHandler._extract_from_value(item))

        return parameters

    @staticmethod
    def validate_parameters(
        trajectory: list[ToolUseBlockParam], provided_values: dict[str, str]
    ) -> tuple[bool, list[str]]:
        """Validate that all required parameters have values.

        Args:
            trajectory: List of tool use blocks containing parameters
            provided_values: Dict of parameter names to their values

        Returns:
            Tuple of (is_valid, missing_parameters)
            - is_valid: True if all parameters have values, False otherwise
            - missing_parameters: List of parameter names that are missing values
        """
        required_parameters = CacheParameterHandler.extract_parameters(trajectory)
        missing = [name for name in required_parameters if name not in provided_values]

        return len(missing) == 0, missing

    @staticmethod
    def substitute_parameters(
        tool_block: ToolUseBlockParam, parameter_values: dict[str, str]
    ) -> ToolUseBlockParam:
        """Replace parameters in a tool block with actual values.

        Creates a new ToolUseBlockParam with all {{parameter}} occurrences
        replaced with their corresponding values from parameter_values.

        Args:
            tool_block: The tool use block containing parameters
            parameter_values: Dict mapping parameter names to replacement values

        Returns:
            New ToolUseBlockParam with parameters substituted
        """
        # Deep copy the input and substitute parameters
        substituted_input = CacheParameterHandler._substitute_in_value(
            tool_block.input, parameter_values
        )

        # Create new ToolUseBlockParam with substituted values
        return ToolUseBlockParam(
            id=tool_block.id,
            name=tool_block.name,
            input=substituted_input,
            type=tool_block.type,
            cache_control=tool_block.cache_control,
        )

    @staticmethod
    def _substitute_in_value(value: Any, parameter_values: dict[str, str]) -> Any:
        """Recursively substitute parameters in a value.

        Args:
            value: Any value (str, dict, list, etc.) containing parameters
            parameter_values: Dict of parameter names to replacement values

        Returns:
            New value with parameters substituted
        """
        if isinstance(value, str):
            # Replace all parameters in the string
            result = value
            for name, replacement in parameter_values.items():
                placeholder = f"{{{{{name}}}}}"  # Creates "{{parameter_name}}"
                result = result.replace(placeholder, replacement)
            return result
        if isinstance(value, dict):
            # Recursively substitute in dict values
            return {
                k: CacheParameterHandler._substitute_in_value(v, parameter_values)
                for k, v in value.items()
            }
        if isinstance(value, list):
            # Recursively substitute in list items
            return [
                CacheParameterHandler._substitute_in_value(item, parameter_values)
                for item in value
            ]
        # Return other types as-is
        return value
