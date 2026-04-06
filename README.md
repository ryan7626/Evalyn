# Evalyn

Evalyn is a local autonomous overnight codebase testing and repair agent for monorepos.

It acts as a careful implementation partner: you code during the day, configure Evalyn to plan tests before you log off, approve the plan, and let it run overnight. By morning, Evalyn will have tested your changes, attempted atomic fixes for failures, and produced a detailed report — all on a safely isolated branch without making unapproved commits.

---

## Key Features

- **LLM-Powered Categorization:** Reads your git diffs and infers logical test categories using Claude.
- **Workflow-Driven Execution:** Enforces a manual planning and approval phase to ensure it only tests what you actually want tested.
- **Safe Iteration:** Always works on a new, isolated branch named per run (e.g., `evalyn/2026-04-04`).
- **Atomic Rollback:** Fixes are attempted one at a time and automatically snapshotted via `.patch` files. If a fix fails, it rolls back gracefully.
- **Zero Auto-Commits:** Evalyn never commits code. All fixes are left unstaged for your manual review in the morning.
- **Context-Aware:** Uses the target repository's own `CLAUDE.md` file as the single source of truth for the stack, rules, and commands.

---

## Installation

Evalyn is built in Python using standard tooling. 

```bash
# Clone the repository
git clone https://github.com/yourusername/Evalyn.git
cd Evalyn

# Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Evalyn (this will install the CLI tool 'evalyn' into your PATH)
pip install -e .
```

## Configuration

### 1. Set your API Key
Copy the `.env.example` file to create your own `.env` and provide your Anthropic API Key:

```bash
cp .env.example .env
```
Ensure you have exported the key to your shell environment before running Evalyn:
```bash
export ANTHROPIC_API_KEY="your-api-key"
```

### 2. Configure the Target Repository
For Evalyn to work on a repository, that target repository **must** contain a `CLAUDE.md` file in its root. This file acts as the project's persistent context, informing Evalyn about the tech stack, test commands (e.g. `npm run test`, `pytest`), and architecture rules.

---

## Usage

Evalyn is heavily workflow-driven and involves a 3-step cycle:

### 1. Plan
Analyze changes since the last run and have Evalyn propose a test plan.

```bash
evalyn plan --repo /path/to/target/project
```
This generates a timestamped `.agent/runs/<timestamp>/plan.md` in the target repo.

### 2. Review & Approve
Open the generated `plan.md` file. 
- You can add annotations, delete categories, or edit test commands directly.
- To approve the plan, change `Status: PENDING` to `Status: APPROVED` inside the markdown file.

### 3. Run
Execute the approved plan. Evalyn creates a separate, isolated branch for its workspace, starts any necessary services, runs tests, and applies iterative LLM-powered fixes if failures occur.

```bash
evalyn run --repo /path/to/target/project
```

- **On Success**: Once completed, the branch is retained. Evalyn drops a `report.md` summarizing exactly what passed, failed, and modified. You manually review and decide whether to merge the branch.
- **On Abort**: If the run is cancelled (`Ctrl+C`) or crashes due to an unexpected error, Evalyn instantly initiates an atomic rollback. It cleans the worktree, safely deletes its messy branch, and restores you exactly back to the main branch you started on. Your `plan.md` remains perfectly intact on your main branch for a flawless restart!

### Check Status
At any point, to see what Evalyn is currently doing or check the results of the last run:

```bash
evalyn status --repo /path/to/target/project
```

---

## How It Works Under The Hood

When Evalyn operates on a target repository, it provisions a `.agent/` runtime directory that acts as its isolated sandbox and state tracker:

```text
target-repo/
  CLAUDE.md                      ← Target context (Read-only for Evalyn)
  .agent/                        ← Evalyn Runtime Directory
    state.json                   ← Tracks branch states, history, and timestamps 
    runs/
      <YYYY-MM-DD_HHMMSS>/       ← Dedicated directory for the specific execution run
        plan.md                  ← The human-reviewed test plan
        report.md                ← Morning summary
        events.log               ← Verbose debug trail
        patches/                 ← Snapshot patches used for reverting failing fixes
        generated-tests/         ← Ephemeral tests injected by Evalyn
```

---

## Core Constraints
Evalyn is bound by rigid safety protocols:
- It **cannot** modify, delete, or skip existing tests in your codebase.
- It **cannot** modify environment variables or secret files (`.env`).
- It **will not** auto-commit changes.
- It operates safely on a standalone branch.
- It halts testing on the first blocking error, attempts a targeted fix, and will flag as "uncertain" if the fix fails rather than brute-forcing code modifications.
