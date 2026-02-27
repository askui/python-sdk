CACHING_PARAMETER_IDENTIFIER_SYSTEM_PROMPT = """You are analyzing UI automation \
trajectories to identify values that should be parameterized as parameters.

Identify values that are likely to change between executions, such as:
- Dates and timestamps (e.g., "2025-12-11", "10:30 AM", "2025-12-11T14:30:00Z")
- Usernames, emails, names (e.g., "john.doe", "test@example.com", "John Smith")
- Session IDs, tokens, UUIDs, API keys
- Dynamic text that references current state or time-sensitive information
- File paths with user-specific or time-specific components
- Temporary or generated identifiers

DO NOT mark as parameters:
- UI element coordinates (x, y positions)
- Fixed button labels or static UI text
- Configuration values that don't change (e.g., timeouts, retry counts)
- Generic action names like "click", "type", "scroll"
- Tool names
- Boolean values or common constants

For each parameter, provide:
1. A descriptive name in snake_case (e.g., "current_date", "user_email")
2. The actual value found in the trajectory
3. A brief description of what it represents

Return your analysis as a JSON object with this structure:
{
  "parameters": [
    {
      "name": "current_date",
      "value": "2025-12-11",
      "description": "Current date in YYYY-MM-DD format"
    }
  ]
}

If no parameters are found, return an empty parameters array."""


CACHE_USE_PROMPT = (
    "<TRAJECTORY USE>\n"
    "    You can use precomputed trajectories to make the execution of the "
    "task more robust and faster!\n"
    "    To do so, first use the RetrieveCachedTestExecutions tool to check "
    "which trajectories are available for you.\n"
    "    The details what each trajectory that is available for you does are "
    "at the end of this prompt.\n"
    "    A trajectory contains all necessary mouse movements, clicks, and "
    "typing actions from a previously successful execution.\n"
    "    If there is a trajectory available for a step you need to take, "
    "always use it!\n"
    "    You can execute a trajectory with the ExecuteCachedExecution tool.\n"
    "    After a trajectory was executed, make sure to verify the results! "
    "While it works most of the time, occasionally, the execution can be "
    "(partly) incorrect. So make sure to verify if everything is filled out "
    "as expected, and make corrections where necessary!\n"
    "    </TRAJECTORY USE>\n"
    "    <TRAJECTORY DETAILS>\n"
    "    There are several trajectories available to you.\n"
    "    Their filename is a unique testID.\n"
    "    If executed using the ExecuteCachedExecution tool, a trajectory will "
    "automatically execute all necessary steps for the test with that id.\n"
    "    </TRAJECTORY DETAILS>\n"
)
