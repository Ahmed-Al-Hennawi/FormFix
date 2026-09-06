/* ==========================================================================
   FormFix - /analyse page behaviour

   Loaded after main.js, on the analysis page only. Three small jobs:

     * the reference-technique lightbox  (open / close / Escape / backdrop)
     * the score count-up                (0 -> the real score)
     * scrolling the results into view   (mobile, where they sit below)

   Everything degrades: without this file the page still renders, the results
   still read correctly, and the reference card simply does nothing.
   ========================================================================== */

(function () {
  "use strict";

  var doc = document;
  var root = doc.documentElement;
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ----------------------------------------------------------------------
     Teardown - a Streamlit rerun re-injects this file.
     ---------------------------------------------------------------------- */
  var cleanups = [];

  function on(target, type, handler, options) {
    if (!target) return;
    target.addEventListener(type, handler, options);
    cleanups.push(function () {
      target.removeEventListener(type, handler, options);
    });
  }

  window.FormFixAnalyseTeardown = function () {
    cleanups.forEach(function (fn) {
      try { fn(); } catch (e) {}
    });
    cleanups = [];
  };

  /* ----------------------------------------------------------------------
     Reference lightbox

     The modal is emitted inside a Streamlit block, which can establish a
     containing block for fixed positioning. Moving it to <body> - the same
     trick main.js uses for the atmosphere - guarantees it covers the viewport.
     Only the newest copy survives a rerun.
     ---------------------------------------------------------------------- */
  var modal = doc.querySelector("[data-ax-modal]");

  if (modal) {
    Array.prototype.forEach.call(doc.querySelectorAll("[data-ax-modal]"), function (el) {
      if (el !== modal && el.parentElement === doc.body) el.remove();
    });
    if (modal.parentElement !== doc.body) doc.body.appendChild(modal);
  }

  var lastTrigger = null;

  function media() {
    return modal ? modal.querySelector("video") : null;
  }

  function openModal(trigger) {
    if (!modal) return;
    lastTrigger = trigger || null;
    modal.hidden = false;
    root.classList.add("ax-modal-open");
    if (doc.body) doc.body.classList.add("ax-modal-open");

    var closer = modal.querySelector(".ax-modal__close");
    if (closer) closer.focus({ preventScroll: true });
  }

  function closeModal() {
    if (!modal || modal.hidden) return;

    var video = media();
    if (video) {
      try { video.pause(); } catch (e) {}
    }

    modal.hidden = true;
    root.classList.remove("ax-modal-open");
    if (doc.body) doc.body.classList.remove("ax-modal-open");

    if (lastTrigger && doc.contains(lastTrigger)) {
      lastTrigger.focus({ preventScroll: true });
    }
    lastTrigger = null;
  }

  cleanups.push(closeModal);

  on(doc, "click", function (event) {
    if (!event.target.closest) return;

    var opener = event.target.closest("[data-ax-open]");
    if (opener) {
      event.preventDefault();
      openModal(opener);
      return;
    }

    if (event.target.closest("[data-ax-close]")) {
      event.preventDefault();
      closeModal();
    }
  });

  on(doc, "keydown", function (event) {
    if (event.key !== "Escape" || !modal || modal.hidden) return;
    closeModal();
  });

  /* Keep focus inside the dialog while it is open. */
  on(doc, "focusin", function (event) {
    if (!modal || modal.hidden) return;
    var panel = modal.querySelector(".ax-modal__panel");
    if (panel && !panel.contains(event.target)) {
      var closer = modal.querySelector(".ax-modal__close");
      if (closer) closer.focus({ preventScroll: true });
    }
  });

  /* ----------------------------------------------------------------------
     Reveal safety net

     main.js observes [data-animate] elements when it is installed. Anything
     that arrives afterwards - a rerun that renders the results - would stay at
     opacity 0 if that observer had already finished with the page. This second
     observer only ever picks up what is still unrevealed, so the two can never
     fight over the same element.
     ---------------------------------------------------------------------- */
  var pending = doc.querySelectorAll("[data-animate]:not(.is-inview)");

  if (pending.length) {
    if (reduced || !("IntersectionObserver" in window)) {
      Array.prototype.forEach.call(pending, function (el) {
        el.classList.add("is-inview");
      });
    } else {
      var revealer = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          if (!entry.isIntersecting) return;
          entry.target.classList.add("is-inview");
          revealer.unobserve(entry.target);
        });
      }, { rootMargin: "0px 0px -8% 0px", threshold: 0.04 });

      Array.prototype.forEach.call(pending, function (el) { revealer.observe(el); });
      cleanups.push(function () { revealer.disconnect(); });
    }
  }

  /* ----------------------------------------------------------------------
     Score count-up
     The ring is animated by CSS; the number is counted here so the two land
     together.
     ---------------------------------------------------------------------- */
  Array.prototype.forEach.call(doc.querySelectorAll("[data-ax-count-to]"), function (el) {
    var target = Number(el.getAttribute("data-ax-count-to"));
    if (!isFinite(target)) return;

    if (reduced) {
      el.textContent = String(Math.round(target));
      return;
    }

    var duration = 1400;
    var delay = 150;
    var started = null;
    var frame = 0;

    function step(now) {
      if (started === null) started = now;
      var elapsed = now - started;
      if (elapsed < delay) {
        frame = requestAnimationFrame(step);
        return;
      }
      var t = Math.min(1, (elapsed - delay) / duration);
      // the same easing curve as --ease-out, approximated
      var eased = 1 - Math.pow(1 - t, 3);
      el.textContent = String(Math.round(target * eased));
      if (t < 1) frame = requestAnimationFrame(step);
    }

    el.textContent = "0";
    frame = requestAnimationFrame(step);
    cleanups.push(function () { cancelAnimationFrame(frame); });
  });

  /* ----------------------------------------------------------------------
     Bring the results into view

     On a phone the results sit underneath the upload panel, so a completed
     analysis would otherwise finish off-screen. Only scrolls when the results
     really are below the fold, and never fights a user who has already
     scrolled there.
     ---------------------------------------------------------------------- */
  var results = doc.querySelector("[data-ax-results]");

  if (results && !window.FormFixAnalyseScrolled) {
    var rect = results.getBoundingClientRect();
    if (rect.top > window.innerHeight * 0.9) {
      window.FormFixAnalyseScrolled = true;
      var timer = setTimeout(function () {
        results.scrollIntoView({
          behavior: reduced ? "auto" : "smooth",
          block: "start"
        });
      }, 120);
      cleanups.push(function () { clearTimeout(timer); });
    }
  }

  if (!results) {
    // A new run resets the one-shot scroll.
    window.FormFixAnalyseScrolled = false;
  }
})();
