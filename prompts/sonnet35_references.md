# Sonnet 3.5 Prompt — Reference Documentation

Sources consulted when authoring `sonnet35_universal.yaml`.

## Anthropic Prompt Engineering

- [Prompting Best Practices](https://platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices) — clear/direct over verbose, system vs user separation, few-shot examples
- [Structured Outputs](https://platform.claude.com/docs/en/build-with-claude/use-structured-outputs) — `response_format.json_schema` for guaranteed schema-valid JSON (preferred over prompting for raw JSON)
- [Vision](https://platform.claude.com/docs/en/build-with-claude/vision) — image before text for best results, optimal 1568px long edge, up to 600 images/request
- [PDF Support](https://platform.claude.com/docs/en/build-with-claude/pdf-support) — page-to-image + extracted text pipeline

## AWS Bedrock Integration

- [Claude on Amazon Bedrock](https://platform.claude.com/docs/en/build-with-claude/claude-on-amazon-bedrock) — model IDs, global vs regional endpoints, feature availability lag
- [Bedrock Structured Outputs](https://docs.aws.amazon.com/bedrock/latest/userguide/structured-output.html) — `json_schema` in `response_format`, schema cached 24h, incompatible with citations
- [Bedrock Claude Messages API](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages.html) — request format, `system` as top-level field, `anthropic_version` header
- [Bedrock Tool Use](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-anthropic-claude-messages-tool-use.html) — `strict: true` for schema-validated tool calls, `tool_choice` control
- [InvokeModel Examples](https://docs.aws.amazon.com/bedrock/latest/userguide/bedrock-runtime_example_bedrock-runtime_InvokeModel_AnthropicClaude_section.html) — Python/boto3 example code

## Key Decisions

| Decision | Rationale | Source |
|----------|-----------|--------|
| System/user prompt split | System for role + rules, user for task — Anthropic best practice | Prompting Best Practices |
| Structured Outputs over tool_use | Lower latency, simpler for extraction (no function-calling overhead) | Structured Outputs docs |
| Image before text in user message | Slight accuracy improvement per Anthropic testing | Vision docs |
| `anyOf` with `type: "null"` | Allows null for inapplicable fields while keeping `strict: true` | Structured Outputs docs |
| Concise prompting (no CRITICAL/STEP) | Sonnet follows instructions reliably; over-prompting is counterproductive | Prompting Best Practices |
| Model v2 (`20241022`) | v1 (`20240620`) sunset Dec 2025 | Claude on Bedrock |
