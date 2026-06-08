"""LLM-powered fixer for Evalyn.

Takes failed test output and repo context, asks the LLM for a targeted fix,
validates the fix against safety rules, and applies it.
"""

import re
import sys
import difflib
from pathlib import Path
from typing import Optional

from evalyn.llm.base import BaseLLM
from evalyn.models import FixAttempt
from evalyn.utils.log import get_logger

logger = get_logger("runner.fixer")


SYSTEM_PROMPT = """\
You are a precise code repair assistant. You will be given:
1. A failed test output (stdout + stderr)
2. The test command that was run
3. The repository context (CLAUDE.md)
4. The contents of files that are likely related to the failure

Your job is to produce a SINGLE targeted fix. You must respond with valid JSON.

Rules:
- Fix ONLY the source code, never the test files.
- Make the MINIMAL change needed to fix the failure.
- Do NOT modify .env files, config files, or test files.
- Your fix must be a search-and-replace: provide the exact text to find and the exact replacement.
- If you are not confident in the fix, say so in the explanation.
- If the failure seems to be an environment issue (not a code bug), explain that and provide no fix.
- Simpler is better. A minimal modification that resolves the failure is preferred.
- Classify the fix as "root_cause", "symptom_patch", or "unknown".
"""

USER_PROMPT_TEMPLATE = """\
## Repository Context

{claude_md_content}

{history_context}
## Failed Test

- **Category**: {category}
- **Command**: `{command}`
- **Exit code**: {exit_code}

### stdout
```
{stdout}
```

### stderr
```
{stderr}
```

## Relevant Source Files

{source_files_content}

## Instructions

Analyze the failure and produce a fix. Respond with JSON:

{{
  "file_path": "relative/path/to/file.ts",
  "original_content": "exact text to find in the file",
  "replacement_content": "exact replacement text",
  "explanation": "why this fix should resolve the failure",
  "confidence": "high" | "medium" | "low",
  "diagnosis": "root_cause" | "symptom_patch" | "unknown"
}}

If you cannot produce a fix (e.g., environment issue, unclear failure), respond with:

{{
  "file_path": null,
  "original_content": null,
  "replacement_content": null,
  "explanation": "why no fix is possible",
  "confidence": "none",
  "diagnosis": "unknown"
}}
"""

# Patterns for files that must NEVER be modified
FORBIDDEN_PATTERNS = [
    r"\.env($|\.)",         # .env, .env.local, etc.
    r"__tests?__/",         # test directories
    r"\.test\.",            # .test.ts, .test.js, etc.
    r"\.spec\.",            # .spec.ts, .spec.js, etc.
    r"test_.*\.py$",        # Python test files
    r".*_test\.py$",        # Python test files (alt convention)
    r"conftest\.py$",       # pytest config
]


def generate_fix(
    llm: BaseLLM,
    repo_path: Path,
    category: str,
    command: str,
    exit_code: int,
    stdout: str,
    stderr: str,
    claude_md_content: str,
    relevant_files: list[str],
    previous_attempts_context: str = "",
) -> FixAttempt:
    """Ask the LLM for a fix hypothesis.

    Args:
        llm: LLM provider.
        repo_path: Target repository path.
        category: Test category name.
        command: The failed test command.
        exit_code: Exit code of the failed command.
        stdout: Captured stdout.
        stderr: Captured stderr.
        claude_md_content: Content of CLAUDE.md.
        relevant_files: List of relative file paths that may be related.
        previous_attempts_context: Information about prior failed fix attempts in this loop.

    Returns:
        FixAttempt with details of what was suggested.
    """
    # Read relevant source files
    source_files_content = _read_source_files(repo_path, relevant_files)

    history_str = ""
    if previous_attempts_context:
        history_str = f"## Previous Failed Attempts\n\n{previous_attempts_context}\n"

    user_prompt = USER_PROMPT_TEMPLATE.format(
        claude_md_content=claude_md_content,
        history_context=history_str,
        category=category,
        command=command,
        exit_code=exit_code,
        stdout=stdout[:3000],  # Cap to avoid huge prompts
        stderr=stderr[:2000],
        source_files_content=source_files_content,
    )

    logger.info(f"Requesting fix for category: {category}")

    try:
        response = llm.complete_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=4096,
        )
    except (ValueError, Exception) as e:
        logger.error(f"LLM fix request failed: {e}")
        return FixAttempt(
            file_path="",
            original_content="",
            replacement_content="",
            explanation=f"LLM request failed: {e}",
            applied=False,
            rejected_reason="LLM error",
        )

    return _process_fix_response(response, repo_path)


def _process_fix_response(response: dict, repo_path: Path) -> FixAttempt:
    """Validate the LLM's suggested fix without applying it yet."""
    file_path = response.get("file_path")
    original = response.get("original_content")
    replacement = response.get("replacement_content")
    explanation = response.get("explanation", "")
    confidence = _normalize_confidence(response.get("confidence"))
    diagnosis = _normalize_diagnosis(response.get("diagnosis"))

    # No fix suggested
    if file_path is None or original is None:
        logger.info(f"LLM declined to fix: {explanation}")
        return FixAttempt(
            file_path="",
            original_content="",
            replacement_content="",
            explanation=explanation,
            confidence=confidence,
            diagnosis=diagnosis,
            applied=False,
            rejected_reason="LLM declined",
        )

    fix = FixAttempt(
        file_path=file_path,
        original_content=original,
        replacement_content=replacement,
        explanation=explanation,
        confidence=confidence,
        diagnosis=diagnosis,
        diff_size=abs(len(replacement or "") - len(original or ""))
    )

    # Safety checks
    rejection = _validate_fix(fix, repo_path)
    if rejection:
        fix.rejected_reason = rejection
        fix.applied = False
        logger.warning(f"Fix validation rejected: {rejection}")

    return fix


def write_suggested_patch(
    fix: FixAttempt,
    repo_path: Path,
    patches_dir: Path,
    name: str,
) -> bool:
    """Write a patch suggestion without changing the working tree."""
    if fix.rejected_reason:
        return False

    rejection = _validate_fix(fix, repo_path)
    if rejection:
        fix.rejected_reason = rejection
        logger.warning(f"Patch suggestion rejected: {rejection}")
        return False

    full_path = repo_path / fix.file_path
    try:
        content = full_path.read_text(encoding="utf-8")
        if fix.original_content not in content:
            fix.rejected_reason = "Original content not found in file"
            logger.warning(f"Patch suggestion rejected: original content not found in {fix.file_path}")
            return False

        new_content = content.replace(fix.original_content, fix.replacement_content, 1)
        patch_lines = difflib.unified_diff(
            content.splitlines(keepends=True),
            new_content.splitlines(keepends=True),
            fromfile=f"a/{fix.file_path}",
            tofile=f"b/{fix.file_path}",
        )
        patch_text = "".join(patch_lines)
        if not patch_text.strip():
            fix.rejected_reason = "Suggested patch was empty"
            return False

        patches_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "fix"
        patch_path = patches_dir / f"{safe_name}.patch"
        patch_path.write_text(patch_text, encoding="utf-8")
        fix.suggested_only = True
        fix.suggested_patch_path = str(patch_path)
        logger.info(f"Patch suggestion written: {patch_path}")
        return True

    except OSError as e:
        fix.rejected_reason = f"File I/O error: {e}"
        logger.error(f"Failed to write patch suggestion: {e}")
        return False


def apply_fix(fix: FixAttempt, repo_path: Path) -> bool:
    """Apply the validated fix to the filesystem.

    Returns:
        True if successfully applied, False otherwise.
    """
    if fix.rejected_reason:
        return False

    full_path = repo_path / fix.file_path
    try:
        content = full_path.read_text(encoding="utf-8")
        if fix.original_content not in content:
            fix.rejected_reason = "Original content not found in file"
            fix.applied = False
            logger.warning(f"Fix rejected: original content not found in {fix.file_path}")
            return False

        new_content = content.replace(fix.original_content, fix.replacement_content, 1)
        full_path.write_text(new_content, encoding="utf-8")
        fix.applied = True
        logger.info(f"Fix applied to {fix.file_path}: {fix.explanation}")
        return True

    except OSError as e:
        fix.rejected_reason = f"File I/O error: {e}"
        fix.applied = False
        logger.error(f"Failed to apply fix: {e}")
        return False


def _validate_fix(fix: FixAttempt, repo_path: Path) -> Optional[str]:
    """Validate a fix against safety rules.

    Returns a rejection reason string, or None if the fix is valid.
    """
    # Check forbidden patterns
    for pattern in FORBIDDEN_PATTERNS:
        if re.search(pattern, fix.file_path):
            return f"Cannot modify protected file: {fix.file_path} (matches {pattern})"

    # Check file exists
    full_path = repo_path / fix.file_path
    if not full_path.is_file():
        return f"File does not exist: {fix.file_path}"

    # Check file is within repo (prevent path traversal)
    try:
        full_path.resolve().relative_to(repo_path.resolve())
    except ValueError:
        return f"File is outside repository: {fix.file_path}"

    # Check fix is not empty
    if not fix.original_content or not fix.replacement_content:
        return "Fix has empty original or replacement content"

    # Check original != replacement
    if fix.original_content == fix.replacement_content:
        return "Original and replacement content are identical"

    return None


def _normalize_confidence(value: object) -> str:
    confidence = str(value or "unknown").lower()
    if confidence not in {"high", "medium", "low", "none", "unknown"}:
        return "unknown"
    return confidence


def _normalize_diagnosis(value: object) -> str:
    diagnosis = str(value or "unknown").lower()
    if diagnosis not in {"root_cause", "symptom_patch", "unknown"}:
        return "unknown"
    return diagnosis


def _read_source_files(
    repo_path: Path,
    file_paths: list[str],
    *,
    max_files: int = 8,
    max_lines_per_file: int = 300,
) -> str:
    """Read and format source files for the LLM prompt."""
    sections = []

    for rel_path in file_paths[:max_files]:
        full_path = repo_path / rel_path

        # Skip files matching forbidden patterns
        if any(re.search(p, rel_path) for p in FORBIDDEN_PATTERNS):
            continue

        if not full_path.is_file():
            continue

        try:
            content = full_path.read_text(encoding="utf-8")
            lines = content.split("\n")
            if len(lines) > max_lines_per_file:
                content = "\n".join(lines[:max_lines_per_file])
                content += f"\n\n... (truncated, {len(lines) - max_lines_per_file} more lines)"

            sections.append(f"### `{rel_path}`\n```\n{content}\n```")
        except (OSError, UnicodeDecodeError):
            continue

    return "\n\n".join(sections) if sections else "_No relevant source files found_"
