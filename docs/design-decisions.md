# Design decisions and trade-offs

## Keep the demo dependency-free

The repository was empty, so the prototype uses Python's HTTP server, SQLite, plain HTML/CSS/JavaScript, and `unittest`. This makes setup reproducible without package installation or API credentials. The alternative—adding a frontend framework, API framework, hosted database, and vector service—would add setup and operational complexity without improving the approval demonstration.

## Prefer transparent retrieval over pretending to have semantic embeddings

Policy passages are indexed using stable feature-hashed TF-IDF vectors and cosine similarity. When a question names a policy title, retrieval is restricted to that document; ranking combines passage metadata and passage text, weighted toward text to reduce irrelevant sentence matches caused by repeated titles. This gives a local retrieval path with no model downloads or external calls, but it is lexical and less robust than learned embeddings. A production system should evaluate a chosen embedding model and vector database against policy-specific retrieval benchmarks.

## Use deterministic extractive profiles

The provider interface and configurable router show where model selection belongs, while the shipped `local-small` / `local-reasoning` profiles extract source sentences deterministically. This avoids sending synthetic/customer content to a third party and makes the demo reproducible. Usage and cost are estimates using whitespace tokenization and illustrative per-profile rates; they are not actual inference or provider charges. The demo does not demonstrate generative quality, multiple commercial providers, or neural-model costs; those require a vetted provider adapter, prompt/version management, data governance, and measured model pricing.

## Enforce approvals with persisted state, not prompts

The application stores workflow state in SQLite and checks it again at execution. Approval is an explicit API operation from a distinct reviewer ID. This is stronger than prompt-only instructions and allows tests to prove that pending/rejected/expired requests cannot execute. Reviewer identity is not authenticated in this prototype, so the separation-of-duties check is illustrative and cannot establish real authorization.

## Minimize audit data

Events store actor role labels, opaque request IDs, policy references, model profile, usage, decisions, and outcomes, but not raw actor IDs, full user prompts, or customer IDs. A schema migration redacts legacy actor IDs and filters legacy event details. This supports a basic event timeline while reducing unnecessary personal data. Production retention, access, integrity, and deletion policies remain required.

## Let planners propose tool calls, but let code decide

The chat can turn a request into a tool call, but the planner (rules, or a local function-calling model) only produces a proposal. The service checks it against a registry that has no approve or execute tool, validates the arguments, requires the customer ID to appear in the staff message, and enforces the business precondition that only eligible customers get a draft. This keeps the language model outside the trust boundary for side effects: a hallucinated or injected tool call fails validation instead of reaching a back end. The cost is a small amount of per-tool validation code and a deliberately narrow rule-based intent detector.

## Make the audit trail tamper-evident

Each audit event stores the previous event's hash and its own SHA-256 over the canonical fields, so editing or deleting a row breaks the chain from that point. This is cheap, dependency-free, and easy to verify in a demo. It detects tampering but does not prevent it: someone with write access can rebuild the whole chain. Production should anchor the head hash periodically in write-once storage or a separate signing service.

## Treat guardrail blocks as refusals

When the optional guardian model flags a prompt, the assistant refuses and records which guardrail fired. Falling back to the offline provider in that case would still answer a prompt a safety control rejected. Transport errors and malformed model output still degrade to the offline provider, because those are availability problems, not safety verdicts.
