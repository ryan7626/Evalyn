"""Ephemeral test generation module for Evalyn.

Uses the LLM to dynamically generate test scripts for required categories
that lack existing test coverage.
"""

import json
from pathlib import Path

from evalyn.llm.base import BaseLLM
from evalyn.utils.log import get_logger

logger = get_logger("runner.test_generator")


SYSTEM_PROMPT = """\
You are an expert test automation engineer agent. Your task is to dynamically generate an ephemeral \
test script to verify specific code changes for an untested requirement.

You will be given:
1. The target repository context (`CLAUDE.md`) dictating the test framework and tech stack.
2. The name and description of the test category we need coverage for.
3. The list of relevant source files that were changed.

Your objective is to write the COMPLETE, self-contained test file content that verifies these changes.
You must absolutely respect the testing framework (e.g., pytest, jest, vitest) defined in the repository context.

Important Constraints:
- Produce ONLY a raw JSON response. No markdown wrappers around the JSON block.
- The JSON object must match this schema:
{
  "filename": "path/to/test_file",
  "content": "raw string content of the code file",
  "run_command": "the bash command to execute THIS SINGLE test file"
}
- Do NOT generate tests that interact with live external production services unless mocked.
- Do NOT include markdown code fences (like ```json). Just the raw braces.

Browser / E2E Testing Instructions:
- If the required testing implies UI, end-to-end, or browser interactions, you MUST generate a synchronous Python Playwright script.
- The script must use standard synchronous playwright syntax: `from playwright.sync_api import sync_playwright`.
- **IMPORTANT**: Set `headless=False` when launching Chromium so the user can visually see the automation running! (e.g. `p.chromium.launch(headless=False)`)
- **IMPORTANT**: Add a `page.wait_for_timeout(10000)` (10 seconds) at the end of your test execution so the user has time to actually watch what the agent did before the browser closes!
- The script must be completely self-executing in a standard python environment.
- Use `python3` instead of `python` for your `run_command` bash hook to ensure compatibility on macOS.
- If a target Local Dev Server URL is provided, ensure your Playwright script navigates to it!
"""

USER_PROMPT_TEMPLATE = """\
## Repository Context (CLAUDE.md)
{claude_md_content}

## Testing Requirement
Category: {category_name}
Description: {category_description}

## Relevant Changed Files
{relevant_files_list}

## Local Execution Context
Target Dev Server Base URL: {base_url}
(If this is provided and your category requires browser interaction, point Playwright to it.)

Generate the ephemeral test script JSON object payload using the exact defined schema. \
The location to write the test script should ideally be within: `.agent/runs/ephemeral_tests/` \
(e.g., `.agent/runs/ephemeral_tests/{safe_filename}`).
"""


class TestGenerationError(Exception):
    pass


def generate_ephemeral_test(
    llm: BaseLLM,
    category_name: str,
    relevant_files: list[str],
    claude_md_content: str,
    run_dir: Path,
    base_url: str | None = None,
) -> dict[str, str]:
    """Generate an ephemeral test script via LLM.

    Args:
        llm: LLM provider
        category_name: Category to generate coverage for
        relevant_files: Files that changed
        claude_md_content: Target repo documentation
        run_dir: Current run directory string/path for scoping names

    Returns:
        dict containing 'filename', 'content', and 'run_command'.
    """
    logger.info(f"Generating ephemeral test suite for category: '{category_name}'")

    files_list = "\n".join(f"- {f}" for f in relevant_files) if relevant_files else "- (No specific files provided)"
    
    # Safe category name for the filename template
    safe_name = "".join(c if c.isalnum() else "_" for c in category_name).strip("_").lower()

    user_prompt = USER_PROMPT_TEMPLATE.format(
        claude_md_content=claude_md_content,
        category_name=category_name,
        category_description="Verification logic is required for changes in this category.",
        relevant_files_list=files_list,
        safe_filename=f"{safe_name}_test",
        base_url=base_url or "Not available (assume standard local host/port or mocked state)",
    )

    try:
        response = llm.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=2048,
        )
    except Exception as e:
        raise TestGenerationError(f"LLM generation failed: {e}")

    # Ensure format
    if not isinstance(response, dict):
        raise TestGenerationError("LLM did not return a valid dict")
    
    if "filename" not in response or "content" not in response or "run_command" not in response:
        raise TestGenerationError(f"LLM omitted required keys. Got: {list(response.keys())}")

    return response
