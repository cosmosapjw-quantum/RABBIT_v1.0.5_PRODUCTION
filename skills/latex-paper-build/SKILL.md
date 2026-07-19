---
name: latex-paper-build
description: Use when creating, editing, compiling, or auditing LaTeX manuscripts, research reports, bibliographies, figures, appendices, supplements, or arXiv/journal-ready source trees.
---

# LaTeX Paper Build Skill

## Purpose

Keep research manuscripts buildable, source-complete, and audit-ready.

Use this skill when:
- editing `.tex`, `.bib`, figures, tables, appendices, or supplements;
- preparing an internal draft, journal draft, or arXiv package;
- converting Markdown/research notes into LaTeX;
- checking unresolved references, citations, figure inclusion, or overfull boxes.

## Required workflow

1. Locate main TeX entrypoint.
2. Identify bibliography system:
   - BibTeX,
   - biblatex/biber,
   - inline thebibliography.
3. Build with `latexmk` if available.
4. Capture compile log.
5. Check:
   - undefined references,
   - undefined citations,
   - missing figures,
   - overfull hboxes,
   - duplicate labels,
   - stale auxiliary files if needed.
6. If figures are generated from scripts, verify source scripts exist.
7. Report build status honestly.

## Required output

```markdown
## Build status

## Entrypoint

## Commands run

## Errors

## Warnings requiring attention

## Figure/source completeness

## Bibliography status

## Remaining risks
```

## Hard prohibitions

- Do not claim PDF compiled unless a build command succeeded.
- Do not hide missing citations.
- Do not embed generated figures without preserving source scripts or provenance.
