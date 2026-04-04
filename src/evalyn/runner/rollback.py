"""Rollback management for Evalyn.

Maintains a patch history for each run, supporting per-fix snapshots
and full session rollback.
"""

from pathlib import Path

from evalyn.utils.git import create_patch, apply_patch, reset_hard
from evalyn.utils.log import get_logger

logger = get_logger("runner.rollback")


class RollbackManager:
    """Manages patch snapshots and rollback for a single run.

    Each fix attempt gets a numbered snapshot. If a fix fails, we can
    rollback to the previous state. On session end, we can rollback
    everything.

    Args:
        run_dir: Path to the current run directory.
        repo_path: Path to the target repository.
    """

    def __init__(self, run_dir: Path, repo_path: Path):
        self.run_dir = run_dir
        self.repo_path = repo_path
        self.patches_dir = run_dir / "patches"
        self.patches_dir.mkdir(parents=True, exist_ok=True)
        self._patch_counter = 0
        self._patches: list[Path] = []

    def create_snapshot(self, description: str) -> Path | None:
        """Save the current uncommitted changes as a numbered patch.

        Args:
            description: Short description for the patch filename.

        Returns:
            Path to the created patch file, or None if no changes to save.
        """
        self._patch_counter += 1
        safe_desc = description.replace(" ", "_").replace("/", "_")[:50]
        patch_name = f"{self._patch_counter:03d}_{safe_desc}.patch"
        patch_path = self.patches_dir / patch_name

        created = create_patch(self.repo_path, patch_path)

        if created:
            self._patches.append(patch_path)
            logger.info(f"Snapshot created: {patch_name}")
            return patch_path
        else:
            logger.debug("No changes to snapshot")
            self._patch_counter -= 1
            return None

    def rollback_last(self) -> bool:
        """Rollback the most recent fix by resetting to HEAD and
        re-applying all patches except the last one.

        Returns:
            True if rollback was performed, False if no patches to rollback.
        """
        if not self._patches:
            logger.warning("No patches to rollback")
            return False

        removed_patch = self._patches.pop()
        logger.info(f"Rolling back: {removed_patch.name}")

        # Reset to clean state
        reset_hard(self.repo_path)

        # Re-apply all remaining patches in order
        for patch in self._patches:
            if patch.is_file():
                logger.debug(f"Re-applying: {patch.name}")
                apply_patch(self.repo_path, patch)

        logger.info(
            f"Rollback complete. {len(self._patches)} patches remaining."
        )
        return True

    def rollback_all(self) -> None:
        """Full session rollback — reset everything to HEAD.

        This discards ALL modifications made during the run.
        """
        logger.info("Full session rollback — resetting to HEAD")
        reset_hard(self.repo_path)
        self._patches.clear()
        logger.info("All changes reverted")

    @property
    def patch_count(self) -> int:
        """Number of active patches."""
        return len(self._patches)

    @property
    def patch_files(self) -> list[Path]:
        """List of active patch files."""
        return list(self._patches)
