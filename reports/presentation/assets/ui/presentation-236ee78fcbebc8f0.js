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
        container.dataset.loaded = "error";
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

  // Injected next to the theme toggle so both the landing page and every
  // notebook export carry the same corner control without templating it twice.
  function initRepoLink() {
    const link = document.createElement("a");
    link.className = "repo-link";
    link.href = "https://github.com/jonasyr/unfallatlas-qua3ck";
    link.target = "_blank";
    link.rel = "noopener";
    link.setAttribute("aria-label", "GitHub-Repository öffnen (neuer Tab)");
    link.innerHTML =
      '<svg viewBox="0 0 16 16" aria-hidden="true"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 8.01 0 0 0 16 8c0-4.42-3.58-8-8-8Z"/></svg>';
    document.body.append(link);
  }

  const THEME_STORAGE_KEY = "unfallatlas-theme";
  const darkSchemeQuery = window.matchMedia("(prefers-color-scheme: dark)");

  function currentTheme() {
    const explicit = document.documentElement.getAttribute("data-theme");
    if (explicit === "light" || explicit === "dark") return explicit;
    return darkSchemeQuery.matches ? "dark" : "light";
  }

  function applyTheme(theme, button) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      window.localStorage.setItem(THEME_STORAGE_KEY, theme);
    } catch {
      // Private browsing / disabled storage: theme still applies for this
      // page view, it just will not persist to the next one.
    }
    button?.setAttribute("aria-pressed", String(theme === "dark"));
  }

  // Injected once here (not templated per-page) so every page that loads
  // this script - the landing page and every notebook export - gets the
  // same toggle without duplicating markup in two separate Jinja templates.
  function initThemeToggle() {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "theme-toggle";
    button.setAttribute("aria-label", "Farbschema umschalten (hell/dunkel)");
    button.setAttribute("aria-pressed", String(currentTheme() === "dark"));
    button.innerHTML =
      '<svg class="icon-sun" viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/><path d="M12 2v2.5M12 19.5V22M4.2 4.2l1.8 1.8M18 18l1.8 1.8M2 12h2.5M19.5 12H22M4.2 19.8 6 18M18 6l1.8-1.8"/></svg>' +
      '<svg class="icon-moon" viewBox="0 0 24 24" aria-hidden="true"><path d="M20.5 14.7A8.2 8.2 0 1 1 9.3 3.5a6.7 6.7 0 0 0 11.2 11.2Z"/></svg>';
    button.addEventListener("click", () => {
      applyTheme(currentTheme() === "dark" ? "light" : "dark", button);
    });
    darkSchemeQuery.addEventListener("change", () => {
      if (!document.documentElement.getAttribute("data-theme")) {
        button.setAttribute("aria-pressed", String(currentTheme() === "dark"));
      }
    });
    document.body.append(button);
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

  function containsTex() {
    const texPattern = /\\\(|\\\[|\$\$|\\begin\s*\{/;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      if (node.parentElement?.closest("code, pre, script, style, textarea")) continue;
      if (texPattern.test(node.nodeValue || "")) return true;
    }
    return false;
  }

  function initializeMathJax() {
    const runtime = document.body.dataset.mathjaxRuntime;
    if (!runtime || !containsTex()) return;
    loadScript(runtime).catch((error) => {
      const message = document.createElement("p");
      message.className = "output-load-error";
      message.textContent =
        error instanceof Error ? error.message : "Formeln konnten nicht geladen werden.";
      document.querySelector(".notebook-main")?.prepend(message);
    });
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

    initThemeToggle();
    initRepoLink();

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
      let synchronizedOpen;
      const synchronize = (loadOutput = true) => {
        if (summary) summary.setAttribute("aria-expanded", String(details.open));
        writeStorage(storageKey(details, index), details.open ? "open" : "closed");
        synchronizedOpen = details.open;
        if (loadOutput && details.open && details.classList.contains("output-cell")) {
          details.querySelectorAll(".plotly-output").forEach(loadPlotly);
        }
      };
      synchronize(false);
      details.addEventListener("toggle", () => {
        if (details.open !== synchronizedOpen) synchronize();
      });
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
    // Long notebooks produce a TOC taller than its sticky viewport; keep the
    // active entry visible by nudging the TOC's own scroll position whenever
    // the highlight would otherwise sit outside it.
    function revealActiveTocLink(link) {
      if (!toc || toc.scrollHeight <= toc.clientHeight) return;
      const linkRect = link.getBoundingClientRect();
      const tocRect = toc.getBoundingClientRect();
      if (linkRect.top >= tocRect.top && linkRect.bottom <= tocRect.bottom) return;
      const target =
        toc.scrollTop + linkRect.top - tocRect.top - toc.clientHeight / 2 + linkRect.height / 2;
      toc.scrollTo({top: target, behavior: reduceMotion.matches ? "auto" : "smooth"});
    }

    const headingObserver = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          const link = headingById.get(entry.target.id);
          if (!link || !entry.isIntersecting) return;
          document.querySelectorAll(".toc-link[aria-current]").forEach((current) =>
            current.removeAttribute("aria-current"),
          );
          link.setAttribute("aria-current", "location");
          revealActiveTocLink(link);
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
    initializeMathJax();
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
