# Tooling Inventory — M0 session (Claude Code, 2026-09-18)

What the agent actually had in this session (as listed by the harness), and what was used.
Nothing was installed (no skill, plugin, or package).

## Available

- Built-in tools: Bash, Read, Write, Edit, Agent (subagents: general-purpose, Explore, Plan, claude-code-guide, statusline-setup, claude), Skill, AskUserQuestion, Artifact, ReportFindings, ScheduleWakeup, SendFeedback, ToolSearch, ListAgents; deferred (not loaded): WebFetch, WebSearch, Monitor, NotebookEdit, worktree/plan-mode/cron tools, Claude Docs MCP tools, IDE MCP tools.
- Skills listed: dataviz, artifact-design, artifact-diagramming, artifact-capabilities, update-config, keybindings-help, code-review, simplify, fewer-permission-prompts, loop, schedule, claude-api, claude-in-chrome, run, init, security-review, anthropic-skills:{docs, docx, import-memory, morning, pdf, pptx, skill-creator, xlsx}.
- No dedicated skill exists for SSH, dataset audit, or Git — normal workflow used (not a blocker).

## Used per task

| Task | Tool used |
|---|---|
| Filesystem inspection | Bash (`find`, `ls`, `stat`, `du`-free listing), Read |
| Git | Bash `git` (with `GIT_OPTIONAL_LOCKS=0` for old repos) |
| Python | system `python3` 3.12.3 (stdlib + apt PyYAML) |
| DOCX/spec reading | stdlib extractor `tools/docx_extract.py` (the `anthropic-skills:docx` skill exists but was not invoked; pandoc/python-docx not installed and nothing was installed) |
| PDF (related doc) | `pdftotext` / `pdfinfo` (poppler, preinstalled) |
| SSH | OpenSSH client with `BatchMode=yes` (no interactive prompts) |
| Dataset audit | Bash `ls`/`find -type d` only (no decoding, no hashing) |
| Testing | stdlib `unittest` (pytest missing) |
