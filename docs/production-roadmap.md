# Production-readiness roadmap

The current application is an offline, single-process demonstration using synthetic policies and customers. It is not safe for real eligibility decisions or customer requests.

## Scalability and reliability

- Replace the development HTTP server with a maintained production ASGI/WSGI server behind TLS and a managed ingress.
- Move SQLite state to a managed relational database with migrations, backup/restore, connection pooling, transactional state changes, and concurrency tests.
- Move in-memory retrieval to a versioned, monitored vector service; support policy updates, deletion, and re-indexing.
- Put long-running model and back-end calls behind a durable queue with timeouts, retries, rate limits, idempotency keys, and dead-letter handling.
- Add caching only for non-sensitive, versioned policy retrieval/results.

## Security and privacy

- Integrate enterprise authentication, authorization, reviewer roles, identity-bound separation of duties, CSRF protections, and tenant isolation.
- Enforce TLS, database encryption, managed secrets, least-privilege service identities, dependency scanning, and API abuse controls.
- Add data classification, PII minimization/redaction, retention/deletion workflows, consent, and privacy impact review.
- Treat retrieved documents and tool results as untrusted; expand injection tests and use provider-independent output validation.
- Penetration-test the deployed system. The current loopback server and synthetic identifiers are not production security controls.

## AI governance and evaluation

- Select approved model providers and versions; add configurable routing, fallbacks, prompt/version records, and change approvals.
- Validate retrieval quality, groundedness, citation support, unsupported-query behavior, bias, and adversarial robustness with a reviewed test corpus.
- Run offline and online evaluations continuously; monitor retrieval drift, human overrides, hallucination reports, and policy-version traceability.
- Implement enforceable human-approval rules based on action risk and authenticated permissions.
- Reconcile token/cost telemetry against provider billing; enforce budgets at both session and organization levels.

## Operations

- Add centralized structured logs, traces, metrics, dashboards, alerting, SLOs, incident response, and audit-log integrity controls.
- Establish CI/CD with automated tests, security checks, evaluation gates, environment promotion, rollback, and deployment approvals.
- Define support ownership, service SLAs, disaster recovery targets, capacity limits, and an incident process.
