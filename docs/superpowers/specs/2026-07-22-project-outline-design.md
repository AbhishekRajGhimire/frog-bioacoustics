# Beginner Project Outline Design

## Purpose

Create `docs/outline.md` as the beginner-friendly orientation guide for the Frog Bioacoustics repository.
The guide will explain what the project does, how its parts fit together, and how a new contributor can work safely.
It will be self-contained enough for a first session while linking to the detailed documents that remain authoritative.

## Audience

The primary reader is a beginner who may be unfamiliar with bioacoustics, machine learning, Python project layouts, or this repository.
The writing will use plain language and define project-specific terms before relying on them.
Commands will target PowerShell because that is the verified local workflow.

## Documentation role

The outline will sit between the root README and the detailed architecture and workflow documents.
It will provide a complete mental model and a safe starting path without copying every implementation detail.
Deep storage contracts will remain in `docs/architecture.md`.
Detailed operating procedures will remain in `docs/workflow.md`.
Sequencing and acceptance criteria will remain in `roadmap.md`.

## Proposed structure

### Project explanation

The opening will explain that the project turns long field recordings into fixed five-second spectrograms for human labeling, validated evaluation, future model training, and eventual frog activity analysis.
It will identify *Litoria aurea* as the current target and explain that the repository now has reproducible structure and trustworthy data validation.

### Current status

The document will state that Phases 1 and 2 are complete and Phase 3 is next.
It will explain that the current 13 confirmed positive examples are not enough for statistically stable model evaluation.
It will make clear that the current baseline is a pipeline sanity check rather than a production detector.

### Pipeline overview

The guide will show a simple text flow from raw recordings through spectrogram generation, human labeling, manifest validation, grouped evaluation, future model training, and future analytics.
Each stage will receive a short plain-language explanation.

### Repository structure

An annotated tree will explain the purpose of `raw`, `processed`, `labeled`, `config`, `scripts`, `src`, `tests`, `models`, `results`, `docs`, and `classifier_app`.
The tree will distinguish command entry points in `scripts` from reusable logic in `src/frog_classifier`.

### Tracked and generated content

The guide will explain which source files and empty directory markers belong in Git.
It will explain that recordings, generated spectrograms, labels, manifests, reports, models, and analytics outputs normally remain local and ignored.
It will emphasize that raw recordings are immutable source material and must not be deleted or overwritten by project commands.

### Getting started

The setup section will require Python 3.13 and `uv`.
It will provide the verified `uv sync --frozen` command and a small verification command.
It will explain the Windows launcher as an alternative entry point for the graphical labeler.

### Working with the pipeline

The document will provide copyable commands for spectrogram generation, labeling, manifest construction, strict baseline execution, and the complete test suite.
Each command will include a short explanation of when to run it and what it changes.
The labeling section will repeat the rule that uncertain examples must be skipped rather than labeled as background.

### Finding the right place to change

The guide will map command-line orchestration to `scripts`, reusable data behavior to `src/frog_classifier/data`, shared preprocessing values to `config/preprocessing.toml`, verification to `tests`, and explanations to `docs`.
It will advise contributors to keep command wrappers thin and put reusable behavior behind tested package interfaces.

### Safe contribution workflow

The workflow will guide a beginner through reading the relevant documentation, updating from Git, creating a focused branch, making one scoped change, adding or updating tests, running verification, inspecting Git status, and committing.
It will tell contributors not to modify unrelated code or generated dependency exports.

### First-contribution checklist

A compact checklist will cover environment setup, document selection, branch creation, implementation, tests, status inspection, and commit preparation.
It will be useful as a practical end-of-task review rather than a second detailed workflow.

### Rules and common mistakes

The guide will call out private-data commits, raw-data mutation, uncertain negative labels, recording-level leakage, direct edits to `requirements_labeler.txt`, missing tests, and confusion between generated artifacts and tracked source.
It will point readers to the manifest and grouped-fold system instead of recommending manual train and test directories.

### Further reading

The final section will link to the README, architecture, workflow, roadmap, preprocessing configuration, and tests.
Links will be repository-relative and valid from `docs/outline.md`.

## Writing style

The document will be beginner-friendly, direct, and practical.
Every full sentence will occupy its own physical Markdown line.
Technical terms will be introduced with a short explanation.
Commands will be copyable and will match the verified project entry points.
The document will not use the prohibited em dash character.

## Scope boundaries

This work will add only the orientation document and any narrowly required navigation link from existing documentation.
It will not rename the Python distribution, local folder, package imports, scripts, or existing documents.
It will not change application behavior, dependencies, preprocessing values, data, labels, models, or generated artifacts.

## Verification

Verification will confirm that every path and relative Markdown link in `docs/outline.md` resolves to an existing tracked target.
It will confirm that documented commands match current script entry points and the locked-environment workflow.
It will scan the new document for placeholders, contradictory status claims, ambiguous instructions, and the prohibited em dash character.
It will run the repository documentation and reproducibility tests, the full test suite, `git diff --check`, and `git status --short` before completion.

## Success criteria

A beginner can explain the project pipeline after reading the document.
A beginner can identify where data, commands, reusable code, tests, results, and documentation belong.
A beginner can set up the environment, run the major workflows, and make a small verified contribution without guessing about safety rules.
The guide complements the existing documentation without becoming a second source of detailed technical truth.
