# Next Session Prompt

/mode: 심사위원
/think: 헤비
/web: auto
/tools: python|files

[TITLE]
Continue: <project title>

[CONTROLLING CONTEXT]
Use the following files as source of truth:
- docs/harness/PROJECT_STATE.md
- docs/harness/CLAIM_LEDGER.md
- docs/harness/VALIDATION_LEDGER.md
- <additional controlling files>

[CURRENT STATUS]
<short status>

[NON-NEGOTIABLE RULES]
- Do not report validation unless commands actually ran.
- Preserve claim status labels.
- Do not introduce mock physics results.
- Keep units/conventions explicit.

[TASK]
<next task>

[REQUIRED OUTPUT]
- changed files,
- validation commands,
- updated handoff summary,
- remaining risks.
