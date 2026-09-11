/* ==========================================================================
   FormFix - /analyse page behaviour

   Loaded after main.js, only on the analysis page. Three jobs:

     * the reference-technique lightbox  (open / close / Escape / backdrop)
     * the score count-up                (0 -> the real score)
     * scrolling the results into view   (mobile, where they sit below)

   Without this the page still works, the reference card just does nothing.
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

     Moved to <body> (like the background in main.js) so position: fixed covers
     the viewport. Only the newest copy is kept after a rerun.
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

  /* keep focus inside the dialog while it's open */
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

     Results rendered after main.js has set up its observer would stay at
     opacity 0, so this picks up anything still not revealed.
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
     The ring is animated in CSS, the number here, so they finish together.
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
      // roughly the same curve as --ease-out
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

     On a phone the results are under the upload panel, so scroll to them when
     they're ready - only if they're actually off-screen.
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
    // a new run resets the one-time scroll
    window.FormFixAnalyseScrolled = false;
  }
})();
