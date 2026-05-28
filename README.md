# Evalyn

Evalyn is your overnight evaluation, testing and repair partner for monorepos.

Instead of waiting for CI/CD loops to fail, or writing tests at the end of a long day, you just tell Evalyn what you changed. You go to sleep. Evalyn runs the tests on a completely isolated branch, safely iterates on fixes if things break, and leaves a morning report summarizing what works and what needs your attention.

## How It Works

Evalyn operates in three simple steps:

1. **Plan:** You code during the day. Before leaving, run `evalyn plan`. Evalyn looks at your git diffs, infers what features changed, and drafts a simple markdown test plan.
2. **Approve:** You open the `plan.md` file. Add notes, cross out tests you don't care about, and change the status from `PENDING` to `APPROVED` — or just say `y` at the interactive prompt to approve and run immediately.
3. **Run:** Run `evalyn run`. Evalyn creates a fresh, isolated branch and gets to work.

By morning, you'll have a summary report. If it finds bugs while running your tests, it tries its best to patch the failure using the context from your codebase. It **never commits** these fixes — everything is carefully left unstaged for your final manual review.

## Key Principles

We built Evalyn to be completely safe to use on your main projects:

- **Isolated execution:** Always works on a new branch named `evalyn/YYYY-MM-DD_HHMMSS`.
- **Zero auto-commits:** It fixes code but leaves it unstaged. You decide what stays.
- **Never touches existing tests:** It evaluates your code against the tests, but it will never delete, modify, or skip your existing suites to artificially pass a run.
- **Graceful aborts:** If you ever cancel a run (`Ctrl+C`) or the OS sends `SIGTERM`, Evalyn immediately tears down the sandbox branch and puts you right back exactly where you started.

## Getting Started

Evalyn is built in Python.

```bash
# Clone and enter the repository
git clone https://github.com/yourusername/Evalyn.git
cd Evalyn

# Set up your local environment
python3 -m venv .venv
source .venv/bin/activate

# Install the evalyn CLI tool
pip install -e .

# Optional: Add Playwright if you need UI testing
pip install -e '.[browser]'
playwright install chromium
```

### Configuration

Evalyn needs an LLM API key to parse your code and figure out how to fix failures it finds.

```bash
cp .env.example .env
```

Export either Anthropic or OpenRouter keys to your shell before running:

```bash
export ANTHROPIC_API_KEY="your-api-key"
# or
export OPENROUTER_API_KEY="your-api-key"
```

### Setting Up a Target Repo

Run `evalyn init` in any project directory to scaffold the required files automatically:

```bash
evalyn init --repo /path/to/your/project
```

This creates:
- **`CLAUDE.md`** — fill this in with your stack (test commands, services, tech overview). Evalyn reads it as the source of truth for your project.
- **`.agent/config.yaml`** — optional per-repo overrides (LLM model, timeouts, max retries).

## Usage Commands

| Command | Description |
|---------|-------------|
| `evalyn init --repo <path>` | Scaffold `CLAUDE.md` and `.agent/config.yaml` in a new target repo |
| `evalyn plan --repo <path>` | Analyze changes and generate a test plan for approval |
| `evalyn run --repo <path>` | Execute the approved test plan |
| `evalyn run --repo <path> --dry-run` | Preview which categories would run without executing anything |
| `evalyn status --repo <path>` | Check the results and timestamp of the last run |
| `evalyn clean --repo <path> --all` | Remove stale `evalyn/*` branches and reset agent state |

---
*For deeper architectural context, internal decision logs, or to see how Evalyn's multi-hypothesis repair loop is structured behind the scenes, see the `claude.md` file.*
