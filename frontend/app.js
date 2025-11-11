document.addEventListener("DOMContentLoaded", () => {
  const API_BASE_URL = window.__TURIY_API_BASE__ || "http://127.0.0.1:8000";

  let isLeftPaneCollapsed = false;
  let isRightPaneCollapsed = false;
  let model = "turiya-standard";
  let activeApi = "google";
  const chatHistoryState = [];
  const selectedDocuments = new Set();

  const escapeHtml = (unsafe = "") =>
    unsafe
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const renderMarkdown = (markdownText = "") => {
    const input = typeof markdownText === "string" ? markdownText : String(markdownText ?? "");
    if (!input.trim()) {
      return "";
    }
    let htmlOutput = input.replace(/\n/g, "<br>");
    try {
      if (window.marked && typeof window.marked.parse === "function") {
        htmlOutput = window.marked.parse(input);
      } else if (typeof window.marked === "function") {
        htmlOutput = window.marked(input);
      }
    } catch (err) {
      console.warn("Markdown rendering failed, falling back to plain text.", err);
      htmlOutput = input.replace(/\n/g, "<br>");
    }
    if (window.DOMPurify && typeof window.DOMPurify.sanitize === "function") {
      return window.DOMPurify.sanitize(htmlOutput);
    }
    return htmlOutput;
  };

  const docPane = document.getElementById("document-pane");
  const sourcePane = document.getElementById("source-pane");

  const collapseLeftBtn = document.getElementById("collapse-left-btn");
  const expandLeftBtn = document.getElementById("expand-left-btn");
  const collapseRightBtn = document.getElementById("collapse-right-btn");
  const expandRightBtn = document.getElementById("expand-right-btn");

  const leftPaneFooter = document.getElementById("left-pane-footer");
  const rightPaneFooter = document.getElementById("right-pane-footer");

  const modelBtnStandard = document.getElementById("model-btn-standard");
  const modelBtnThinking = document.getElementById("model-btn-thinking");

  const chatInput = document.getElementById("chat-input");
  const sendBtn = document.getElementById("send-btn");
  const chatHistory = document.getElementById("chat-history");
  const chatLoading = document.getElementById("chat-loading");

  const sourceList = document.getElementById("source-list");
  const documentsContainer = document.getElementById("documents-container");
    const uploadInput = document.getElementById("upload-input");
    const uploadButton = document.getElementById("upload-button");
  const rebuildButton = document
    .getElementById("left-pane-footer")
    ?.querySelector("button");

  const mobileMenuBtn = document.getElementById("mobile-menu-btn");
  const mobileSourceBtn = document.getElementById("mobile-source-btn");
  const mobileCloseLeftBtn = document.getElementById("mobile-close-left-btn");
  const mobileCloseRightBtn = document.getElementById("mobile-close-right-btn");
  const mobileOverlay = document.getElementById("mobile-overlay");

  function apiUrl(path) {
    return `${API_BASE_URL}${path}`;
  }

  function toggleLeftPane() {
    isLeftPaneCollapsed = !isLeftPaneCollapsed;
    const listContainer = documentsContainer;
    const uploadSection = docPane.querySelector(".p-4.flex-shrink-0");
    const title = docPane.querySelector("h2");

    if (isLeftPaneCollapsed) {
      docPane.classList.add("w-14", "collapsed");
      docPane.classList.remove("w-72");
      listContainer?.classList.add("hidden");
      uploadSection?.classList.add("hidden");
      title?.classList.add("hidden");
      leftPaneFooter?.classList.add("hidden");
      collapseLeftBtn?.classList.add("hidden");
      expandLeftBtn?.classList.remove("hidden");
            } else {
      docPane.classList.remove("w-14", "collapsed");
      docPane.classList.add("w-72");
      listContainer?.classList.remove("hidden");
      uploadSection?.classList.remove("hidden");
      title?.classList.remove("hidden");
      leftPaneFooter?.classList.remove("hidden");
      collapseLeftBtn?.classList.remove("hidden");
      expandLeftBtn?.classList.add("hidden");
    }
  }

  function toggleRightPane() {
    isRightPaneCollapsed = !isRightPaneCollapsed;
    const listContainer = sourcePane.querySelector(".flex-grow");
    const title = sourcePane.querySelector("h2");

    if (isRightPaneCollapsed) {
      sourcePane.classList.add("w-14", "collapsed");
      sourcePane.classList.remove("w-72");
      listContainer?.classList.add("hidden");
      title?.classList.add("hidden");
      rightPaneFooter?.classList.add("hidden");
      collapseRightBtn?.classList.add("hidden");
      expandRightBtn?.classList.remove("hidden");
    } else {
      sourcePane.classList.remove("w-14", "collapsed");
      sourcePane.classList.add("w-72");
      listContainer?.classList.remove("hidden");
      title?.classList.remove("hidden");
      rightPaneFooter?.classList.remove("hidden");
      collapseRightBtn?.classList.remove("hidden");
      expandRightBtn?.classList.add("hidden");
    }
  }

  collapseLeftBtn?.addEventListener("click", toggleLeftPane);
  expandLeftBtn?.addEventListener("click", toggleLeftPane);
  collapseRightBtn?.addEventListener("click", toggleRightPane);
  expandRightBtn?.addEventListener("click", toggleRightPane);

  function showMobileOverlay() {
    mobileOverlay?.classList.remove("hidden");
  }
  function hideMobileOverlay() {
    mobileOverlay?.classList.add("hidden");
    docPane.classList.add("-translate-x-full");
    sourcePane.classList.add("translate-x-full");
  }

  mobileMenuBtn?.addEventListener("click", () => {
    showMobileOverlay();
    docPane.classList.remove("-translate-x-full");
  });
  mobileSourceBtn?.addEventListener("click", () => {
    showMobileOverlay();
    sourcePane.classList.remove("translate-x-full");
  });
  mobileCloseLeftBtn?.addEventListener("click", hideMobileOverlay);
  mobileCloseRightBtn?.addEventListener("click", hideMobileOverlay);
  mobileOverlay?.addEventListener("click", hideMobileOverlay);

  modelBtnStandard?.addEventListener("click", () => {
    model = "turiya-standard";
    modelBtnStandard.classList.add("bg-blue-600", "text-white", "shadow-sm");
    modelBtnStandard.setAttribute("aria-pressed", "true");

    modelBtnThinking.classList.remove("bg-blue-600", "text-white", "shadow-sm");
    modelBtnThinking.setAttribute("aria-pressed", "false");
  });

  modelBtnThinking?.addEventListener("click", () => {
    model = "turiya-thinking";
    modelBtnThinking.classList.add("bg-blue-600", "text-white", "shadow-sm");
    modelBtnThinking.setAttribute("aria-pressed", "true");

    modelBtnStandard.classList.remove("bg-blue-600", "text-white", "shadow-sm");
    modelBtnStandard.setAttribute("aria-pressed", "false");
  });

  chatInput?.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = `${chatInput.scrollHeight}px`;
  });

  chatInput?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  });

  sendBtn?.addEventListener("click", () => handleSendMessage());

  uploadButton?.addEventListener("click", () => uploadInput?.click());
  uploadInput?.addEventListener("change", handleUpload);
  rebuildButton?.addEventListener("click", handleRebuildIndex);

  async function handleSendMessage(customPrompt, options = {}) {
    if (!chatInput) return;
    const prompt = (customPrompt ?? chatInput.value).trim();
    if (!prompt) return;

    const escalate = Boolean(options.escalate);
    const modeOverride = options.mode ?? null;

    const requestMode = escalate ? "turiya-thinking" : modeOverride ?? model;

    if (!options.skipUserRender) {
      renderUserMessage(prompt);
    }
    chatInput.value = "";
    chatInput.style.height = "auto";

    setLoading(true);

    try {
      const response = await fetch(apiUrl("/query"), {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: prompt,
          api: activeApi,
          mode: requestMode,
          escalate_to_thinking: escalate,
          chat_history: chatHistoryState,
          selected_documents: Array.from(selectedDocuments),
        }),
      });

      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Query failed");
      }

      const data = await response.json();
      const answer = data.answer || "I couldn't generate an answer.";
      renderAiMessage(answer, prompt, data.mode_used, { chart: data.chart });
      renderSources(data.sources || []);

      chatHistoryState.push([prompt, answer]);
        } catch (error) {
      renderAiMessage(`Error: ${error.message}`, prompt, requestMode, { isError: true });
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(event) {
    const file = event.target?.files?.[0];
    if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        try {
            uploadButton.disabled = true;
      uploadButton.textContent = "Uploading…";

      const response = await fetch(apiUrl("/upload"), {
                method: "POST",
                body: formData,
            });

            if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }

      await fetchDocuments();
      alert(`Uploaded ${file.name}. Consider rebuilding the index to include it.`);
        } catch (error) {
      alert(`Upload error: ${error.message}`);
        } finally {
            uploadButton.disabled = false;
      uploadButton.textContent = "Upload Documents";
            uploadInput.value = "";
        }
  }

  async function handleRebuildIndex() {
    if (!confirm("Rebuild the index now? This may take several minutes.")) return;

    try {
            rebuildButton.disabled = true;
      rebuildButton.textContent = "Rebuilding…";

      const response = await fetch(apiUrl("/rebuild-index"), {
                method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ api: activeApi, mode: model }),
            });

            if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Rebuild failed");
            }

      alert("Index rebuild complete.");
        } catch (error) {
      alert(`Rebuild error: ${error.message}`);
        } finally {
            rebuildButton.disabled = false;
      rebuildButton.textContent = "Re-Build Index";
    }
  }

  function setLoading(isLoading) {
    if (!sendBtn || !chatInput || !chatLoading) return;
    if (isLoading) {
      sendBtn.classList.add("hidden");
      chatLoading.classList.remove("hidden");
      chatInput.disabled = true;
    } else {
      sendBtn.classList.remove("hidden");
      chatLoading.classList.add("hidden");
      chatInput.disabled = false;
      chatInput.focus();
    }
  }

  function renderUserMessage(text) {
    if (!chatHistory) return;
    const messageEl = document.createElement("div");
    messageEl.className = "flex justify-end";
    const safeMessage = escapeHtml(text).replace(/\n/g, "<br>");
    messageEl.innerHTML = `
      <div class="p-4 bg-blue-100 text-gray-800 rounded-l-lg rounded-br-lg max-w-xl shadow-sm">
        <p>${safeMessage}</p>
      </div>
    `;
    chatHistory.appendChild(messageEl);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function renderAiMessage(text, originatingQuery, modeUsed, options = {}) {
    if (!chatHistory) return;
    const messageEl = document.createElement("div");
    messageEl.className = "flex items-start space-x-3 max-w-xl";
    const isError = options.isError;
    const chart = options.chart;
    const contentHtml = renderMarkdown(text);

    let chartHtml = "";
    if (chart && chart.data) {
      const title = escapeHtml(chart.title || "Generated chart");
      const description = chart.description ? escapeHtml(chart.description) : "";
      const mime = chart.mime_type || "image/png";
      chartHtml = `
        <div class="chart-wrapper mt-3">
          <img class="generated-chart" src="data:${mime};base64,${chart.data}" alt="${title}" />
          ${description ? `<p class="chart-description mt-2">${description}</p>` : ""}
        </div>`;
    }

    const actionBar = isError
      ? ""
      : `
        <div class="action-bar flex items-center space-x-2 text-gray-500 mt-3 pt-2 border-t border-gray-200">
          <button class="p-1 rounded-md hover:bg-gray-200" data-action="copy" aria-label="Copy message" title="Copy message">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="pointer-events-none"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"></rect><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"></path></svg>
          </button>
          <button class="p-1 rounded-md hover:bg-gray-200" data-action="thumb-up" aria-label="Good response" title="Good response">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="pointer-events-none"><path d="M7 10v12"></path><path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h3Z"></path></svg>
          </button>
          <button class="p-1 rounded-md hover:bg-gray-200" data-action="thumb-down" aria-label="Bad response" title="Bad response">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="pointer-events-none"><path d="M17 14V2"></path><path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-3Z"></path></svg>
          </button>
          <button class="p-1 rounded-md hover:bg-gray-200" data-action="regenerate" aria-label="Regenerate with thinking model" title="Regenerate with thinking model" data-query="${originatingQuery ?? ""}" data-mode="${modeUsed ?? model}">
            <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="pointer-events-none"><path d="M12 3a6.364 6.364 0 0 0 9 9 9 9 0 1 1-9-9Z"></path><path d="M10 12h.01"></path><path d="M12 10v.01"></path><path d="M14 12h.01"></path><path d="M12 14v.01"></path><path d="M10.031 8.916 7.5 12l2.53 3.084"></path><path d="m13.97 15.084 2.53-3.084-2.53-3.084"></path></svg>
          </button>
        </div>`;

    messageEl.innerHTML = `
      <div class="flex-shrink-0 w-8 h-8 rounded-full ${isError ? "bg-red-500" : "bg-blue-500"} flex items-center justify-center text-white font-semibold text-sm">
        ${isError ? "!" : "T"}
      </div>
      <div class="flex-grow p-4 ${isError ? "bg-red-50" : "bg-gray-50"} rounded-r-lg rounded-bl-lg shadow-sm">
        <div class="text-gray-800 leading-relaxed space-y-3 markdown-body">${contentHtml || escapeHtml(text).replace(/\n/g, "<br>")}</div>
        ${chartHtml}
        ${originatingQuery && !isError ? actionBar : ""}
      </div>
    `;
    chatHistory.appendChild(messageEl);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  chatHistory?.addEventListener("click", (e) => {
    const button = e.target.closest("button[data-action]");
    if (!button) return;

    const action = button.dataset.action;
    const messageText = button
      .closest(".flex-grow")
      ?.querySelector("p")
      ?.textContent?.trim();

    switch (action) {
      case "copy":
        if (!messageText) return;
        navigator.clipboard.writeText(messageText).then(() => {
          const originalIcon = button.innerHTML;
          button.innerHTML =
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="text-green-500"><path d="M20 6 9 17l-5-5"></path></svg>';
          button.disabled = true;
          setTimeout(() => {
            button.innerHTML = originalIcon;
            button.disabled = false;
          }, 1500);
        });
        break;
      case "thumb-up":
        button.classList.toggle("text-blue-500");
        break;
      case "thumb-down":
        button.classList.toggle("text-red-500");
        break;
      case "regenerate": {
        const previousQuery = button.dataset.query;
        if (!previousQuery) {
          alert("Original prompt unavailable for regeneration.");
          return;
        }
        handleSendMessage(previousQuery, { escalate: true, skipUserRender: true });
        break;
      }
    }
  });

  function renderSources(sources) {
    if (!sourceList) return;
    if (!Array.isArray(sources) || sources.length === 0) {
      sourceList.innerHTML = "<p class=\"text-sm text-gray-500\">No sources returned.</p>";
            return;
        }
    sourceList.innerHTML = "";
    sources.forEach((source) => {
      const tileEl = document.createElement("div");
      tileEl.className = "source-tile border rounded-lg p-3 cursor-pointer hover:shadow-sm";
      const sourceName = source.metadata?.source || "Unknown";
      const pageNumber = source.metadata?.page ?? 0;
      const snippet = source.page_content || "";
      tileEl.innerHTML = `
        <div class="flex justify-between items-center text-sm font-semibold text-gray-600">
          <span>${sourceName}${pageNumber ? `, pg. ${pageNumber}` : ""}</span>
          <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="chevron transition-transform duration-200"><path d="m6 9 6 6 6-6"></path></svg>
        </div>
        <div class="source-content text-sm text-gray-700 mt-2 space-y-2 max-h-24 overflow-hidden">
          <p>${snippet.slice(0, 240)}${snippet.length > 240 ? "…" : ""}</p>
                    </div>
        <div class="source-content-expanded hidden text-sm text-gray-700 mt-2 space-y-2">
          <p>${snippet.replace(/\n/g, "<br>")}</p>
                </div>
      `;
      sourceList.appendChild(tileEl);
    });
  }

  sourceList?.addEventListener("click", (e) => {
    const tile = e.target.closest(".source-tile");
    if (!tile) return;

    const chevron = tile.querySelector(".chevron");
    const collapsedContent = tile.querySelector(".source-content");
    const expandedContent = tile.querySelector(".source-content-expanded");
    const isExpanded = expandedContent?.classList.contains("hidden");

    if (isExpanded) {
      chevron?.classList.add("rotate-180");
      collapsedContent?.classList.add("hidden");
      expandedContent?.classList.remove("hidden");
    } else {
      chevron?.classList.remove("rotate-180");
      collapsedContent?.classList.remove("hidden");
      expandedContent?.classList.add("hidden");
    }
  });

  async function fetchDocuments() {
    if (!documentsContainer) return;
    try {
      documentsContainer.innerHTML =
        '<h3 class="text-sm font-semibold text-gray-500 uppercase">Available Documents</h3><p class="text-sm text-gray-500">Loading…</p>';
      const response = await fetch(apiUrl("/documents"));
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.detail || "Failed to load documents");
      }
      const documents = await response.json();
      renderDocumentList(documents);
    } catch (error) {
      documentsContainer.innerHTML = `<p class="text-sm text-red-500">${error.message}</p>`;
    }
  }

  function renderDocumentList(documents) {
    if (!documentsContainer) return;
    documentsContainer.innerHTML =
      '<h3 class="text-sm font-semibold text-gray-500 uppercase">Available Documents</h3>';

    if (!documents || documents.length === 0) {
      documentsContainer.innerHTML += '<p class="text-sm text-gray-500">No documents uploaded.</p>';
      return;
    }

    documents.forEach((docName) => {
      const item = document.createElement("div");
      item.className = "flex items-center space-x-3 p-2 rounded-lg hover:bg-gray-100 cursor-pointer";
      item.dataset.docName = docName;
      item.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="flex-shrink-0 text-gray-500"><path d="M14.5 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V7.5L14.5 2z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><line x1="10" y1="9" x2="8" y2="9"></line></svg>
        <span class="truncate text-sm font-medium">${docName}</span>
      `;
      item.addEventListener("click", () => toggleDocumentSelection(item, docName));
      if (selectedDocuments.has(docName)) {
        item.classList.add("bg-blue-50", "border", "border-blue-200");
      }
      documentsContainer.appendChild(item);
    });
  }

  function toggleDocumentSelection(element, docName) {
    if (selectedDocuments.has(docName)) {
      selectedDocuments.delete(docName);
      element.classList.remove("bg-blue-50", "border", "border-blue-200");
    } else {
      selectedDocuments.add(docName);
      element.classList.add("bg-blue-50", "border", "border-blue-200");
    }
  }

  // Initial bootstrapping
  fetchDocuments();
});


