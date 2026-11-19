"use strict";

const conversationId = crypto.randomUUID();
const actorId = "staff-demo";
let activeRequest = null;

const elements = {
  messages: document.querySelector("#messages"),
  chatStatus: document.querySelector("#chat-status"),
  eligibilityResult: document.querySelector("#eligibility-result"),
  requestResult: document.querySelector("#request-result"),
  pendingRequests: document.querySelector("#pending-requests"),
  approvedRequests: document.querySelector("#approved-requests"),
  auditEvents: document.querySelector("#audit-events"),
};

async function postJson(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) {
    throw new Error(result.error || `Request failed (${response.status}).`);
  }
  return result;
}

function showError(target, error) {
  target.textContent = error.message;
  target.classList.add("error");
}

function addMessage(role, text, result) {
  if (elements.messages.querySelector(".empty-state")) {
    elements.messages.replaceChildren();
  }
  const article = document.createElement("article");
  article.className = `message ${role}`;
  const heading = document.createElement("strong");
  heading.textContent = role === "assistant" ? "Assistant" : "You";
  const body = document.createElement("p");
  body.textContent = text;
  article.append(heading, body);

  if (result?.citations?.length) {
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    summary.textContent = `Sources (${result.citations.length})`;
    details.append(summary);
    for (const citation of result.citations) {
      const source = document.createElement("p");
      source.className = "citation";
      source.textContent = `${citation.document} · ${citation.section}: ${citation.excerpt}`;
      details.append(source);
    }
    article.append(details);
  }
  if (result?.model) {
    const metadata = document.createElement("small");
    metadata.textContent =
      `${result.provider} / ${result.model} · ` +
      `${result.usage.input_tokens} in / ${result.usage.output_tokens} out · ` +
      `$${result.usage.estimated_cost_usd.toFixed(6)} simulated · ` +
      `${result.usage.latency_ms} ms`;
    article.append(metadata);
  }
  elements.messages.append(article);
  elements.messages.scrollTop = elements.messages.scrollHeight;
}

document.querySelector("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = document.querySelector("#question").value.trim();
  if (!question) return;
  elements.chatStatus.textContent = "Searching approved policies…";
  elements.chatStatus.classList.remove("error");
  addMessage("user", question);
  try {
    const result = await postJson("/api/chat", {
      question,
      conversation_id: conversationId,
      actor_id: actorId,
    });
    addMessage("assistant", result.answer, result);
    elements.chatStatus.textContent = "";
    document.querySelector("#question").value = "";
    await refreshAudit();
  } catch (error) {
    elements.chatStatus.textContent = error.message;
    elements.chatStatus.classList.add("error");
  }
});

document.querySelector("#eligibility-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  elements.eligibilityResult.classList.remove("error");
  try {
    const result = await postJson("/api/eligibility", {
      customer_id: form.get("customer_id"),
      service_code: form.get("service_code"),
      actor_id: actorId,
      conversation_id: conversationId,
    });
    elements.eligibilityResult.textContent =
      `${result.eligible === null ? "Unable to verify" : result.eligible ? "Eligible" : "Not eligible"} — ${result.reason} ` +
      `Decision source: ${result.decision_source}. ${result.decision_type}`;
    await refreshAudit();
  } catch (error) {
    showError(elements.eligibilityResult, error);
  }
});

document.querySelector("#request-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const form = new FormData(event.currentTarget);
  elements.requestResult.classList.remove("error");
  try {
    activeRequest = await postJson("/api/requests", {
      customer_id: form.get("customer_id"),
      service_code: form.get("service_code"),
      actor_id: form.get("actor_id"),
      conversation_id: conversationId,
    });
    renderRequest();
    await refreshRequests();
    await refreshAudit();
  } catch (error) {
    showError(elements.requestResult, error);
  }
});

document.querySelector("#refresh-audit").addEventListener("click", refreshAudit);

async function reviewRequest(requestId, action) {
  const reviewerId = document.querySelector("#reviewer").value.trim();
  try {
    const updatedRequest = await postJson(`/api/requests/${requestId}/${action}`, {
      reviewer_id: reviewerId,
    });
    if (activeRequest?.id === updatedRequest.id) {
      activeRequest = updatedRequest;
      renderRequest();
    }
    await refreshRequests();
    await refreshAudit();
  } catch (error) {
    showError(elements.requestResult, error);
  }
}

async function executeRequest(requestId) {
  try {
    const updatedRequest = await postJson(`/api/requests/${requestId}/execute`, {});
    if (activeRequest?.id === updatedRequest.id) {
      activeRequest = updatedRequest;
      renderRequest();
    } else {
      elements.requestResult.textContent =
        `Request ${updatedRequest.id} · ${updatedRequest.state}` +
        (updatedRequest.external_reference
          ? ` · ${updatedRequest.external_reference} (${updatedRequest.external_status})`
          : "");
    }
    await refreshRequests();
    await refreshAudit();
  } catch (error) {
    showError(elements.requestResult, error);
  }
}

function renderRequest() {
  elements.requestResult.classList.remove("error");
  elements.requestResult.textContent =
    `Request ${activeRequest.id} · ${activeRequest.service_code} · ` +
    `${activeRequest.state}` +
    (activeRequest.external_reference
      ? ` · ${activeRequest.external_reference} (${activeRequest.external_status})`
      : "");
}

async function refreshRequests() {
  try {
    const [pendingResponse, approvedResponse] = await Promise.all([
      fetch("/api/requests?state=PENDING_APPROVAL", { cache: "no-store" }),
      fetch("/api/requests?state=APPROVED", { cache: "no-store" }),
    ]);
    const [pendingResult, approvedResult] = await Promise.all([
      pendingResponse.json(),
      approvedResponse.json(),
    ]);
    if (!pendingResponse.ok) {
      throw new Error(pendingResult.error || "Could not load pending requests.");
    }
    if (!approvedResponse.ok) {
      throw new Error(approvedResult.error || "Could not load approved requests.");
    }
    renderRequestQueue(elements.pendingRequests, pendingResult.requests, "pending");
    renderRequestQueue(elements.approvedRequests, approvedResult.requests, "approved");
  } catch (error) {
    for (const list of [elements.pendingRequests, elements.approvedRequests]) {
      list.replaceChildren();
      const item = document.createElement("li");
      item.className = "error";
      item.textContent = error.message;
      list.append(item);
    }
  }
}

function renderRequestQueue(list, requests, state) {
  list.replaceChildren();
  if (!requests.length) {
    const empty = document.createElement("li");
    empty.className = "empty-state";
    empty.textContent =
      state === "pending"
        ? "No requests pending approval."
        : "No approved requests awaiting execution.";
    list.append(empty);
    return;
  }
  for (const request of requests) {
    const item = document.createElement("li");
    const summary = document.createElement("p");
    summary.textContent =
      `${request.service_code} · customer ${request.customer_id} · ${request.id}`;
    const actions = document.createElement("div");
    actions.className = "button-row";
    if (state === "pending") {
      actions.append(
        createActionButton("Approve", "secondary", () =>
          reviewRequest(request.id, "approve")),
        createActionButton("Reject", "danger", () =>
          reviewRequest(request.id, "reject")),
      );
    } else {
      actions.append(
        createActionButton("Execute approved request", "", () =>
          executeRequest(request.id)),
      );
    }
    item.append(summary, actions);
    list.append(item);
  }
}

function createActionButton(label, className, action) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  if (className) button.className = className;
  button.addEventListener("click", action);
  return button;
}

async function refreshAudit() {
  try {
    const response = await fetch("/api/audit", { cache: "no-store" });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "Could not load audit history.");
    elements.auditEvents.replaceChildren();
    if (!result.events.length) {
      const empty = document.createElement("li");
      empty.className = "empty-state";
      empty.textContent = "No events recorded yet.";
      elements.auditEvents.append(empty);
      return;
    }
    for (const event of result.events) {
      const item = document.createElement("li");
      const title = document.createElement("strong");
      title.textContent = event.event_type.replaceAll("_", " ");
      const details = document.createElement("small");
      const summary =
        event.details.model ||
        event.details.outcome ||
        event.details.state ||
        event.details.guardrail ||
        "";
      details.textContent =
        `${event.timestamp} · ${event.actor_role}` +
        (summary ? ` · ${summary}` : "");
      item.append(title, details);
      elements.auditEvents.append(item);
    }
  } catch (error) {
    elements.auditEvents.replaceChildren();
    const item = document.createElement("li");
    item.className = "error";
    item.textContent = error.message;
    elements.auditEvents.append(item);
  }
}

refreshRequests();
refreshAudit();
