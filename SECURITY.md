# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.3.x   | :white_check_mark: |
| < 0.3.0 | :x:                |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, email **1715136863@qq.com** with details.

We aim to acknowledge reports within 48 hours and provide a timeline for resolution within 5 business days.

## Scope

`ai-llm` handles sensitive data including API keys, user prompts, and model outputs. We take the following concerns seriously:

- Prompt injection vectors in template processing
- API key exposure through logs or error messages
- Insecure handling of user data in RAG pipelines
- Dependency supply chain risks
