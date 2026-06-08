"""Category inference for Evalyn.

Uses the LLM to map changed files to test categories based on the
target repo's CLAUDE.md context.
"""

from dataclasses import dataclass
from typing import Optional

from evalyn.llm.base import BaseLLM
from evalyn.planner.diff_analyzer import ChangeSet
from evalyn.utils.log import get_logger

logger = get_logger("planner.category_inference")


@dataclass
class TestCategory:
    """A category of tests inferred from code changes."""

    name: str
    description: str
    relevant_files: list[str]
    suggested_test_commands: list[str]
    priority: str  # "required" | "optional"
    risk_level: str = "medium"  # "high" | "medium" | "low"
    affected_surface: str = ""
    reason: str = ""
    success_criteria: str = ""


SYSTEM_PROMPT = """\
You are a code analysis assistant for an automated testing agent. Your job is to \
analyze code changes in a repository and categorize them into logical test categories.

You will be given:
1. The repository's context document (CLAUDE.md) which describes the project structure, \
stack, and available commands.
2. A list of changed files.
3. A summary of commits.
4. The actual diff of changes.

Based on this information, produce a JSON array of test categories. Each category should \
represent a logical area of the codebase that was affected and needs testing.

Rules:
- Only include categories that are DIRECTLY related to the changes.
- Each category must have at least one suggested test command that would verify the changes.
- If you cannot determine a test command from the context, use a reasonable default \
and set the priority to "optional".
- Categories should be specific (e.g., "Auth Session Handling") not vague (e.g., "Backend").
- Required categories are those where changes are significant and testing is essential.
- Optional categories are for minor changes or areas where test coverage is uncertain.
- Risk level should reflect likely user impact and blast radius: "high", "medium", or "low".
- Success criteria should state what passing evidence would make the result actionable.
"""

USER_PROMPT_TEMPLATE = """\
## Repository Context (CLAUDE.md)

{claude_md_content}

## Changed Files ({num_files} files)

{files_list}

## Recent Commits ({num_commits} commits)

{commits_list}

## Diff (may be truncated)

{diff_text}

## Instructions

Analyze the changes above and produce a JSON array of test categories.

Each category object must have these fields:
- "name": string — short descriptive name
- "description": string — what changed and why it needs testing
- "relevant_files": string[] — which changed files relate to this category
- "suggested_test_commands": string[] — commands to run to test this category
- "priority": "required" | "optional"
- "risk_level": "high" | "medium" | "low"
- "affected_surface": string — the product or system area likely affected
- "reason": string — why this category matters for the changed code
- "success_criteria": string — what evidence should count as a useful pass

Respond with ONLY the JSON array. Example:
[
  {{
    "name": "Auth Session Handling",
    "description": "Login flow modified, session token validation changed",
    "relevant_files": ["src/auth/login.ts", "src/auth/session.ts"],
    "suggested_test_commands": ["npm test -- --grep auth"],
    "priority": "required",
    "risk_level": "high",
    "affected_surface": "Authentication and session renewal",
    "reason": "Session validation changes can block login or expose stale sessions.",
    "success_criteria": "Auth tests cover login, token refresh, and invalid session rejection."
  }}
]
"""


def infer_categories(
    llm: BaseLLM,
    changeset: ChangeSet,
    claude_md_content: str,
) -> list[TestCategory]:
    """Use the LLM to infer test categories from code changes.

    Args:
        llm: LLM provider instance.
        changeset: The analyzed changes.
        claude_md_content: Content of the target repo's CLAUDE.md.

    Returns:
        List of TestCategory objects.
    """
    if changeset.is_empty:
        logger.info("No changes to categorize")
        return []

    # Build commit descriptions
    commits_list = "\n".join(
        f"- {c.sha[:8]}: {c.message} ({c.author})"
        for c in changeset.commits
    ) or "No commits found"

    # Build file list
    files_list = "\n".join(f"- {f}" for f in changeset.files_changed)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        claude_md_content=claude_md_content,
        num_files=len(changeset.files_changed),
        files_list=files_list,
        num_commits=len(changeset.commits),
        commits_list=commits_list,
        diff_text=changeset.diff_text or "(no diff available)",
    )

    logger.info(f"Requesting category inference for {len(changeset.files_changed)} files")

    try:
        raw_categories = llm.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=4096,
        )
    except (ValueError, Exception) as e:
        logger.error(f"LLM category inference failed: {e}")
        # Return a single catch-all category so the plan isn't empty
        return [_fallback_category(changeset)]

    # Parse the response
    categories = _parse_categories(raw_categories)
    logger.info(f"Inferred {len(categories)} test categories")

    for cat in categories:
        logger.info(f"  [{cat.priority.upper()}] {cat.name}: {cat.description}")

    return categories


def _parse_categories(raw: dict | list) -> list[TestCategory]:
    """Parse LLM response into TestCategory objects.

    Handles both a direct array and a dict with a 'categories' key.
    """
    # Handle case where LLM wraps in an object
    if isinstance(raw, dict):
        if "categories" in raw:
            raw = raw["categories"]
        else:
            # Try to find an array in the first value
            for v in raw.values():
                if isinstance(v, list):
                    raw = v
                    break

    if not isinstance(raw, list):
        logger.warning(f"Unexpected LLM response type: {type(raw)}")
        return []

    categories = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        try:
            categories.append(TestCategory(
                name=str(item.get("name", "Unknown")),
                description=str(item.get("description", "")),
                relevant_files=item.get("relevant_files", []),
                suggested_test_commands=item.get("suggested_test_commands", []),
                priority=str(item.get("priority", "optional")).lower(),
                risk_level=_normalize_risk_level(item.get("risk_level")),
                affected_surface=str(item.get("affected_surface", "")),
                reason=str(item.get("reason", "")),
                success_criteria=str(item.get("success_criteria", "")),
            ))
        except (TypeError, ValueError) as e:
            logger.warning(f"Skipping malformed category: {e}")
            continue

    return categories


def _normalize_risk_level(value: object) -> str:
    """Return a valid risk level for plan/report metadata."""
    risk = str(value or "medium").lower()
    if risk not in {"high", "medium", "low"}:
        return "medium"
    return risk


def _fallback_category(changeset: ChangeSet) -> TestCategory:
    """Create a fallback category when LLM inference fails."""
    return TestCategory(
        name="All Changes (fallback)",
        description=(
            "LLM category inference failed. This is a catch-all category "
            "covering all changed files. Manual review recommended."
        ),
        relevant_files=changeset.files_changed,
        suggested_test_commands=[],
        priority="required",
        risk_level="medium",
        affected_surface="Changed files",
        reason="LLM inference failed, so Evalyn cannot narrow the risk surface.",
        success_criteria="A human reviews the fallback plan and adds the right test commands.",
    )
