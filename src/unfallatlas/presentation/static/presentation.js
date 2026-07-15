window.UnfallatlasPresentation = (() => {
  "use strict";

  document.documentElement.classList.add("js");

  const plotlyPayloads = new Map();
  const plotlyWaiters = new Map();
  const loadedScripts = new Map();
  const plotlyLoads = new Map();

  function registerPlotlyPayload(payloadKey, payload) {
    plotlyPayloads.set(payloadKey, payload);
    const waiters = plotlyWaiters.get(payloadKey) || [];
    waiters.forEach((resolve) => resolve(payload));
    plotlyWaiters.delete(payloadKey);
  }

  function loadScript(src) {
    if (!loadedScripts.has(src)) {
      loadedScripts.set(
        src,
        new Promise((resolve, reject) => {
          const script = document.createElement("script");
          script.src = src;
          script.onload = resolve;
          script.onerror = () =>
            reject(new Error(`Lokales Asset konnte nicht geladen werden: ${src}`));
          document.head.append(script);
        }),
      );
    }
    return loadedScripts.get(src);
  }

  function waitForPayload(id) {
    if (plotlyPayloads.has(id)) return Promise.resolve(plotlyPayloads.get(id));
    return new Promise((resolve) => {
      const waiters = plotlyWaiters.get(id) || [];
      waiters.push(resolve);
      plotlyWaiters.set(id, waiters);
    });
  }

  async function loadPlotly(container) {
    if (container.dataset.loaded === "true") return;
    if (plotlyLoads.has(container)) return plotlyLoads.get(container);
    const loading = (async () => {
      try {
        await loadScript(document.body.dataset.plotlyRuntime);
        const pendingPayload = waitForPayload(container.dataset.payloadKey);
        await loadScript(container.dataset.asset);
        const payload = await pendingPayload;
        container.replaceChildren();
        await window.Plotly.newPlot(container, payload.data, payload.layout, {
          responsive: true,
          displaylogo: false,
        });
        container.dataset.loaded = "true";
      } catch (error) {
        const message = document.createElement("p");
        message.className = "output-load-error";
        message.textContent =
          error instanceof Error ? error.message : "Grafik konnte nicht geladen werden.";
        container.replaceChildren(message);
      } finally {
        plotlyLoads.delete(container);
      }
    })();
    plotlyLoads.set(container, loading);
    return loading;
  }

  function readStorage(key) {
    try {
      return window.sessionStorage.getItem(key);
    } catch {
      return null;
    }
  }

  function writeStorage(key, value) {
    try {
      window.sessionStorage.setItem(key, value);
    } catch {
      return;
    }
  }

  function initialize() {
    const body = document.body;
    const statusRegion = document.querySelector("[data-status-region]");
    const toc = document.querySelector(".toc");
    const tocButton = document.querySelector('[data-action="toggle-toc"]');
    const backToTop = document.querySelector(".back-to-top");
    const snapshot = body.getAttribute("data-snapshot-sha256") || "unknown";
    const storagePrefix = `unfallatlas-presentation:${snapshot}`;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    let tocReturnFocus = null;

    function announce(message) {
      if (statusRegion) statusRegion.textContent = message;
    }

    function storageKey(details, index) {
      const type = details.classList.contains("code-cell") ? "code" : "output";
      return `${storagePrefix}:details:${type}:${index}`;
    }

    const detailsElements = [...document.querySelectorAll("details")];
    detailsElements.forEach((details, index) => {
      const saved = readStorage(storageKey(details, index));
      if (saved !== null) details.open = saved === "open";
      const summary = details.querySelector(":scope > summary");
      const synchronize = (loadOutput = true) => {
        if (summary) summary.setAttribute("aria-expanded", String(details.open));
        writeStorage(storageKey(details, index), details.open ? "open" : "closed");
        if (loadOutput && details.open && details.classList.contains("output-cell")) {
          details.querySelectorAll(".plotly-output").forEach(loadPlotly);
        }
      };
      synchronize(false);
      details.addEventListener("toggle", () => synchronize());
    });

    function setDetails(selector, open) {
      document.querySelectorAll(selector).forEach((details) => {
        details.open = open;
        if (open && details.classList.contains("output-cell")) {
          details.querySelectorAll(".plotly-output").forEach(loadPlotly);
        }
      });
      announce(open ? "Bereiche geöffnet." : "Bereiche geschlossen.");
    }

    function closeToc() {
      if (!toc) return;
      toc.dataset.open = "false";
      tocButton?.setAttribute("aria-expanded", "false");
      tocReturnFocus?.focus();
      tocReturnFocus = null;
    }

    function openToc(trigger) {
      if (!toc) return;
      tocReturnFocus = trigger;
      toc.dataset.open = "true";
      tocButton?.setAttribute("aria-expanded", "true");
      toc.querySelector("a, button")?.focus();
    }

    document.addEventListener("click", async (event) => {
      const button = event.target.closest("button[data-action]");
      if (!button) return;
      const action = button.dataset.action;
      if (action === "show-all-code") setDetails("details.code-cell", true);
      if (action === "hide-all-code") setDetails("details.code-cell", false);
      if (action === "show-all-output") setDetails("details.output-cell", true);
      if (action === "hide-all-output") setDetails("details.output-cell", false);
      if (action === "toggle-toc") {
        if (toc?.dataset.open === "true") closeToc();
        else openToc(button);
      }
      if (action === "close-toc") closeToc();
      if (action === "back-to-top") {
        window.scrollTo({top: 0, behavior: reduceMotion.matches ? "auto" : "smooth"});
      }
      if (action === "toggle-expand") {
        const region = document.getElementById(button.getAttribute("aria-controls"));
        region?.classList.toggle("is-expanded");
        button.setAttribute("aria-expanded", String(region?.classList.contains("is-expanded")));
      }
      if (action === "print") {
        announce("Grafiken werden für den Druck vorbereitet.");
        await Promise.allSettled([...document.querySelectorAll(".plotly-output")].map(loadPlotly));
        window.print();
      }
    });

    document.addEventListener("keydown", (event) => {
      if (event.key === "Escape" && toc?.dataset.open === "true") closeToc();
    });

    document.querySelectorAll(".table-scroll, .text-output").forEach((region, index) => {
      if (region.scrollHeight <= region.clientHeight && region.scrollWidth <= region.clientWidth) return;
      if (!region.id) region.id = `expandable-output-${index + 1}`;
      const button = document.createElement("button");
      button.type = "button";
      button.className = "expand-output";
      button.dataset.action = "toggle-expand";
      button.setAttribute("aria-controls", region.id);
      button.setAttribute("aria-expanded", "false");
      button.textContent = "Vollständig anzeigen";
      region.insertAdjacentElement("afterend", button);
    });

    const plotObserver = new IntersectionObserver(
      (entries) => {
        entries.filter((entry) => entry.isIntersecting).forEach((entry) => {
          loadPlotly(entry.target);
          plotObserver.unobserve(entry.target);
        });
      },
      {rootMargin: "400px 0px"},
    );
    document.querySelectorAll(".plotly-output").forEach((container) => plotObserver.observe(container));

    const headingById = new Map(
      [...document.querySelectorAll(".heading-anchor[id]")].map((anchor) => [
        anchor.id,
        document.querySelector(`.toc-link[href="#${CSS.escape(anchor.id)}"]`),
      ]),
    );
    const headingObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const link = headingById.get(entry.target.id);
          if (!link || !entry.isIntersecting) return;
          document.querySelectorAll(".toc-link[aria-current]").forEach((current) =>
            current.removeAttribute("aria-current"),
          );
          link.setAttribute("aria-current", "location");
        });
      },
      {rootMargin: "-15% 0px -70%"},
    );
    headingById.forEach((_, id) => headingObserver.observe(document.getElementById(id)));

    const header = document.querySelector(".presentation-header");
    if (header && backToTop) {
      new IntersectionObserver(([entry]) => {
        backToTop.dataset.visible = String(!entry.isIntersecting);
      }).observe(header);
    }

    document.querySelectorAll(".toc-link").forEach((link) => link.addEventListener("click", closeToc));
    window.addEventListener("beforeprint", () => {
      document.querySelectorAll("details").forEach((details) => {
        details.open = true;
      });
      document.querySelectorAll(".plotly-output").forEach(loadPlotly);
    });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();

  return {registerPlotlyPayload, loadPlotly};
})();
