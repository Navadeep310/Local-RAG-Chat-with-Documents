const $ = (selector) => document.querySelector(selector);
const dropzone = $("#dropzone");
const fileInput = $("#file-input");
const message = $("#message");

function showMessage(text) { message.textContent = text; message.hidden = false; }
function clearMessage() { message.hidden = true; message.textContent = ""; }
function formatBytes(bytes) { return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`; }

async function loadHealth() {
  try {
    const response = await fetch("/api/health");
    const data = await response.json();
    const ready = data.ollama.available && data.ollama.embedding_model_ready && data.ollama.chat_model_ready;
    $("#connection-dot").classList.toggle("ready", ready);
    $("#connection-text").textContent = ready ? "Ollama connected" : "Ollama needs setup";
  } catch { $("#connection-text").textContent = "Backend unavailable"; }
}

async function loadDocuments() {
  const response = await fetch("/api/documents");
  const data = await response.json();
  $("#doc-count").textContent = `${data.documents.length} ${data.documents.length === 1 ? "file" : "files"}`;
  $("#documents").innerHTML = data.documents.map((doc) => `
    <article class="document-card"><div class="file-icon ${doc.file_type.toLowerCase()}">${doc.file_type}</div>
      <div><h4 title="${escapeHtml(doc.name)}">${escapeHtml(doc.name)}</h4><p>${doc.chunk_count} chunks · ${formatBytes(doc.size_bytes)}</p></div>
      <button class="delete-button" data-id="${doc.id}" aria-label="Remove ${escapeHtml(doc.name)}">×</button></article>`).join("");
  document.querySelectorAll(".delete-button").forEach((button) => button.addEventListener("click", () => deleteDocument(button.dataset.id)));
}

function escapeHtml(value) { const div = document.createElement("div"); div.textContent = value; return div.innerHTML.replaceAll('"', "&quot;").replaceAll("'", "&#39;"); }

async function deleteDocument(id) {
  const response = await fetch(`/api/documents/${id}`, { method: "DELETE" });
  if (!response.ok) { showMessage("We could not remove that document."); return; }
  await loadDocuments();
}

async function upload(file) {
  if (!$("#upload-progress").hidden) return;
  clearMessage(); $("#upload-progress").hidden = false;
  const formData = new FormData(); formData.append("file", file);
  try {
    const response = await fetch("/api/documents/upload", { method: "POST", body: formData });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || "Upload failed.");
    await loadDocuments();
  } catch (error) { showMessage(error.message); }
  finally { $("#upload-progress").hidden = true; loadHealth(); }
}

$("#browse-button").addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", () => { if (fileInput.files[0]) upload(fileInput.files[0]); fileInput.value = ""; });
["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => { event.preventDefault(); dropzone.classList.remove("dragover"); }));
dropzone.addEventListener("drop", (event) => { const file = event.dataTransfer.files[0]; if (file) upload(file); });

$("#chat-form").addEventListener("submit", async (event) => {
  event.preventDefault(); const input = $("#question"); const question = input.value.trim(); if (!question) return;
  if ($("#ask-button").disabled) return;
  const button = $("#ask-button"); button.disabled = true; clearMessage();
  const log = $("#chat-log"); const empty = log.querySelector(".empty-chat"); if (empty) empty.remove();
  const pending = document.createElement("div"); pending.className = "message-row"; pending.innerHTML = `<div class="question-bubble">${escapeHtml(question)}</div><div class="answer-label">Thinking with your sources…</div>`; log.appendChild(pending); log.scrollTop = log.scrollHeight;
  input.value = "";
  try {
    const response = await fetch("/api/chat", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ question, top_k: Number($("#top-k").value) }) });
    const data = await response.json(); if (!response.ok) throw new Error(data.detail || "Could not answer the question.");
    pending.innerHTML = `<div class="question-bubble">${escapeHtml(question)}</div><div class="answer-label">Answer</div><div class="answer">${escapeHtml(data.answer)}</div><div class="sources">${data.sources.map((source, index) => `<details class="source-chip"><summary>Source ${index + 1}: ${escapeHtml(source.document_name)} <b>${escapeHtml(source.location)} · cosine ${source.score}</b></summary><p>${escapeHtml(source.text)}</p></details>`).join("")}</div>`;
  } catch (error) { pending.remove(); input.value = question; showMessage(error.message); }
  finally { button.disabled = false; log.scrollTop = log.scrollHeight; }
});

$("#question").setAttribute("aria-label", "Question about your documents");
$("#question").maxLength = 2000;
loadHealth(); loadDocuments().catch(() => showMessage("Could not load the document library. Check the backend connection."));
