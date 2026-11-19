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
