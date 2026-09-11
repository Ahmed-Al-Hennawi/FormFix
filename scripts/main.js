/* ==========================================================================
   FormFix - Streamlit behaviour layer

   The small bit of JS the site needs, injected by utils/styling.py. Replaces
   the GSAP + ScrollTrigger code from the prototype with native APIs:

     * scroll progress line      - one rAF-throttled scroll handler
     * sticky nav surface        - class toggle on the same handler
     * active section indication - IntersectionObserver
     * smooth in-page navigation - scrollTo({behavior:'smooth'})
     * scroll reveals            - IntersectionObserver -> .is-inview
     * scrubbed progress line    - the "how it works" rail
     * exercise explorer         - class-driven CSS transitions
     * hero mouse parallax       - pointer-fine only

   If this never runs, everything is still visible (the reveal CSS only applies
   under html.ff-motion, which is added here).
   ========================================================================== */

(function () {
  "use strict";

  var doc = document;
  var root = doc.documentElement;
  var reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /* ----------------------------------------------------------------------
     Teardown - Streamlit reruns inject this again, so the old copy has to
     remove its listeners and observers first.
     ---------------------------------------------------------------------- */
  var cleanups = [];

  function on(target, type, handler, options) {
    target.addEventListener(type, handler, options);
    cleanups.push(function () {
      target.removeEventListener(type, handler, options);
    });
  }

  function track(observer) {
    cleanups.push(function () {
      observer.disconnect();
    });
    return observer;
  }

  window.FormFixTeardown = function () {
    cleanups.forEach(function (fn) {
      try { fn(); } catch (e) {}
    });
    cleanups = [];
  };

  /* ----------------------------------------------------------------------
     Scroll container

     Streamlit scrolls its main section, not <html>, and the element name has
     changed between versions, so it's found by measuring instead of a fixed
     selector.
     ---------------------------------------------------------------------- */
  var SCROLLER_SELECTORS = [
    'section.stMain',
    '[data-testid="stMain"]',
    'section.main',
    '[data-testid="stAppViewContainer"] > section',
    '[data-testid="stAppViewContainer"]'
  ];

  function findScroller() {
    for (var i = 0; i < SCROLLER_SELECTORS.length; i++) {
      var el = doc.querySelector(SCROLLER_SELECTORS[i]);
      if (el && el.scrollHeight - el.clientHeight > 4) return el;
    }
    return doc.scrollingElement || root;
  }

  var scroller = findScroller();

  function isDocumentScroller(el) {
    return el === root || el === doc.body || el === doc.scrollingElement;
  }

  function scrollTop() {
    return isDocumentScroller(scroller)
      ? window.scrollY || window.pageYOffset || 0
      : scroller.scrollTop;
  }

  function scrollRange() {
    if (isDocumentScroller(scroller)) {
      var full = Math.max(root.scrollHeight, doc.body ? doc.body.scrollHeight : 0);
      return Math.max(0, full - window.innerHeight);
    }
    return Math.max(0, scroller.scrollHeight - scroller.clientHeight);
  }

  /* ----------------------------------------------------------------------
     Move the fixed layers to <body>

     The background, progress line and nav need to be fixed to the viewport,
     not a Streamlit block.
     ---------------------------------------------------------------------- */
  function hoist(selector) {
    var el = doc.querySelector(selector);
    if (el && el.parentElement !== doc.body) doc.body.appendChild(el);
    return el;
  }

  var atmosphere = hoist(".atmosphere");
  var progressWrap = hoist(".scroll-progress");
  var navWrap = hoist(".nav-wrap");
  var progressBar = progressWrap ? progressWrap.querySelector(".scroll-progress__bar") : null;

  /* ----------------------------------------------------------------------
     Scroll progress line + sticky nav surface
        top of page -> 0      halfway -> ~0.5      bottom -> 1
     ---------------------------------------------------------------------- */
  var queued = false;
  var range = scrollRange();

  function applyScrollState() {
    queued = false;

    var y = scrollTop();
    var progress = range > 0 ? Math.min(1, Math.max(0, y / range)) : 0;

    if (progressBar) {
      progressBar.style.transform = "scaleX(" + progress + ")";
    }

    if (navWrap) {
      navWrap.classList.toggle("is-scrolled", y > 8);
    }

    updateScrubs(y);
  }

  function requestScrollState() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(applyScrollState);
  }

  function remeasure() {
    scroller = findScroller();
    range = scrollRange();
    applyScrollState();
  }

  // scroll doesn't bubble, but capture phase catches it from any element
  on(doc, "scroll", function (event) {
    var target = event.target;
    if (target && target.nodeType === 1 && target !== scroller) {
      if (target.scrollHeight - target.clientHeight > 4) scroller = target;
    }
    requestScrollState();
  }, { capture: true, passive: true });

  on(window, "scroll", requestScrollState, { passive: true });
  on(window, "resize", remeasure);
  on(window, "orientationchange", remeasure);
  on(window, "load", remeasure);

  if (typeof ResizeObserver !== "undefined") {
    var roQueued = false;
    var ro = track(new ResizeObserver(function () {
      if (roQueued) return;
      roQueued = true;
      requestAnimationFrame(function () {
        roQueued = false;
        remeasure();
      });
    }));
    ro.observe(root);
    if (doc.body) ro.observe(doc.body);
  }

  if (doc.fonts && doc.fonts.ready) {
    doc.fonts.ready.then(remeasure).catch(function () {});
  }

  // late-loading images change the page height
  Array.prototype.forEach.call(doc.querySelectorAll(".ff-page img"), function (img) {
    if (!img.complete) on(img, "load", remeasure);
  });

  /* ----------------------------------------------------------------------
     Smooth in-page navigation
     ---------------------------------------------------------------------- */
  var NAV_OFFSET = 76;

  function scrollToSection(id) {
    var target = doc.getElementById(id);
    if (!target) return;

    scroller = findScroller();
    var behavior = reduced ? "auto" : "smooth";

    if (isDocumentScroller(scroller)) {
      var absolute = target.getBoundingClientRect().top + (window.scrollY || 0) - NAV_OFFSET;
      window.scrollTo({ top: Math.max(0, absolute), behavior: behavior });
    } else {
      var delta = target.getBoundingClientRect().top - scroller.getBoundingClientRect().top;
      scroller.scrollTo({ top: Math.max(0, scroller.scrollTop + delta - NAV_OFFSET), behavior: behavior });
    }
  }

  on(doc, "click", function (event) {
    var link = event.target.closest ? event.target.closest("[data-scroll-to]") : null;
    if (!link) return;
    event.preventDefault();
    closeMobileMenu();
    scrollToSection(link.getAttribute("data-scroll-to"));
  });

  /* ----------------------------------------------------------------------
     Mobile navigation
     ---------------------------------------------------------------------- */
  var navToggle = doc.querySelector(".nav__toggle");
  var mobileMenu = doc.getElementById("mobile-menu");

  function closeMobileMenu() {
    if (!navToggle || !mobileMenu) return;
    navToggle.setAttribute("aria-expanded", "false");
    mobileMenu.hidden = true;
    if (navWrap) navWrap.classList.remove("is-open");
  }

  if (navToggle && mobileMenu) {
    on(navToggle, "click", function () {
      var open = navToggle.getAttribute("aria-expanded") === "true";
      navToggle.setAttribute("aria-expanded", String(!open));
      mobileMenu.hidden = open;
      if (navWrap) navWrap.classList.toggle("is-open", !open);
    });
  }

  /* ----------------------------------------------------------------------
     Scroll reveals
     ---------------------------------------------------------------------- */
  if (!reduced && "IntersectionObserver" in window) {
    root.classList.add("ff-motion");

    var revealObserver = track(new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-inview");
        revealObserver.unobserve(entry.target);
      });
    }, { rootMargin: "0px 0px -12% 0px", threshold: 0.06 }));

    Array.prototype.forEach.call(doc.querySelectorAll("[data-animate]"), function (el) {
      revealObserver.observe(el);
    });

    /* steps light up when in view and dim again scrolling back up */
    var stepObserver = track(new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        entry.target.classList.toggle("is-active", entry.isIntersecting);
      });
    }, { rootMargin: "-38% 0px -30% 0px", threshold: 0 }));

    Array.prototype.forEach.call(doc.querySelectorAll(".step"), function (step) {
      stepObserver.observe(step);
    });

    /* active section in the nav */
    var navLinks = Array.prototype.slice.call(doc.querySelectorAll(".nav__link[data-scroll-to]"));
    var sectionObserver = track(new IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        if (!entry.isIntersecting) return;
        var id = entry.target.id;
        navLinks.forEach(function (link) {
          link.classList.toggle("is-current", link.getAttribute("data-scroll-to") === id);
        });
      });
    }, { rootMargin: "-45% 0px -50% 0px", threshold: 0 }));

    navLinks.forEach(function (link) {
      var section = doc.getElementById(link.getAttribute("data-scroll-to"));
      if (section) sectionObserver.observe(section);
    });
  } else {
    // reduced motion or no IntersectionObserver: just show everything
    Array.prototype.forEach.call(doc.querySelectorAll("[data-animate]"), function (el) {
      el.classList.add("is-inview");
    });
    Array.prototype.forEach.call(doc.querySelectorAll(".step"), function (step) {
      step.classList.add("is-active");
    });
  }

  /* ----------------------------------------------------------------------
     Scrubbed effects
       - the "how it works" rail draws as the section passes
       - the outro logo grows slightly on approach
     ---------------------------------------------------------------------- */
  var howFlow = doc.querySelector(".how__flow");
  var howProgress = doc.querySelector(".how__line-progress");
  var outroLogo = doc.querySelector(".outro__logo");

  function updateScrubs() {
    var viewport = window.innerHeight;

    if (howFlow && howProgress) {
      var rect = howFlow.getBoundingClientRect();
      var start = viewport * 0.7;
      var end = -rect.height + viewport * 0.55;
      var span = start - end;
      var value = span > 0 ? (start - rect.top) / span : 0;
      howProgress.style.transform = "scaleY(" + Math.min(1, Math.max(0, value)) + ")";
    }

    if (outroLogo) {
      var logoRect = outroLogo.getBoundingClientRect();
      var from = viewport * 0.9;
      var to = viewport * 0.3;
      var t = (from - logoRect.top) / (from - to);
      t = Math.min(1, Math.max(0, t));
      outroLogo.style.transform = "scale(" + (0.85 + 0.15 * t).toFixed(3) + ")";
    }
  }

  /* ----------------------------------------------------------------------
     Hero mouse parallax
     ---------------------------------------------------------------------- */
  var heroStage = doc.querySelector(".hero__stage");
  var fine = window.matchMedia("(pointer: fine)").matches;

  if (heroStage && fine && !reduced) {
    var layers = Array.prototype.slice.call(heroStage.querySelectorAll("[data-depth]"));
    var pointerQueued = false;
    var pointer = { x: 0, y: 0 };

    var applyParallax = function () {
      pointerQueued = false;
      layers.forEach(function (layer) {
        var depth = Number(layer.getAttribute("data-depth")) || 1;
        layer.style.setProperty("--px", (pointer.x * 10 * depth).toFixed(2) + "px");
        layer.style.setProperty("--py", (pointer.y * 8 * depth).toFixed(2) + "px");
        layer.style.translate = (pointer.x * 10 * depth).toFixed(2) + "px " +
                                (pointer.y * 8 * depth).toFixed(2) + "px";
      });
    };

    on(window, "mousemove", function (event) {
      pointer.x = event.clientX / window.innerWidth - 0.5;
      pointer.y = event.clientY / window.innerHeight - 0.5;
      if (pointerQueued) return;
      pointerQueued = true;
      requestAnimationFrame(applyParallax);
    }, { passive: true });
  }

  /* ----------------------------------------------------------------------
     Exercise explorer

     All three panels are already in the DOM, switching just moves the
     is-current / is-leaving classes and the CSS does the transition.
     ---------------------------------------------------------------------- */
  var explorer = doc.querySelector(".explorer__stage-wrap");

  if (explorer) {
    var tabs = Array.prototype.slice.call(explorer.querySelectorAll(".tab"));
    var figures = Array.prototype.slice.call(explorer.querySelectorAll(".ex-figure"));
    var panels = Array.prototype.slice.call(explorer.querySelectorAll(".explorer__panel"));
    var metrics = Array.prototype.slice.call(explorer.querySelectorAll(".explorer__metric"));
    var previews = Array.prototype.slice.call(explorer.querySelectorAll(".explorer__preview-item"));
    var info = explorer.querySelector(".explorer__info");
    var count = explorer.querySelector(".explorer__count-current");
    var glow = explorer.querySelector(".explorer__glow");
    var poolA = atmosphere ? atmosphere.querySelector(".atmosphere__pool--a") : null;
    var poolB = atmosphere ? atmosphere.querySelector(".atmosphere__pool--b") : null;

    var current = 0;
    var animating = false;
    var total = figures.length;

    function showOnly(list, index) {
      list.forEach(function (el, i) {
        el.hidden = i !== index;
      });
    }

    function setAtmosphere(figure) {
      var a = figure.getAttribute("data-atm-a");
      var b = figure.getAttribute("data-atm-b");
      if (!a || !b) return;
      root.style.setProperty("--atm-a", a);
      root.style.setProperty("--atm-b", b);
      if (poolA) poolA.style.background = "radial-gradient(circle, " + a + " 0%, transparent 62%)";
      if (poolB) poolB.style.background = "radial-gradient(circle, " + b + " 0%, transparent 60%)";
      if (glow) {
        glow.style.background =
          "radial-gradient(circle, rgba(38, 84, 170, 0.22) 0%, " + a + " 40%, transparent 66%)";
      }
    }

    function setActiveTab(index) {
      tabs.forEach(function (tab, i) {
        tab.classList.toggle("is-active", i === index);
        tab.setAttribute("aria-selected", String(i === index));
      });
    }

    function switchExercise(index, direction) {
      index = ((index % total) + total) % total;
      if (index === current || animating) return;

      var dir = direction !== undefined ? direction : (index > current ? 1 : -1);
      var outgoing = figures[current];
      var incoming = figures[index];

      setActiveTab(index);
      setAtmosphere(incoming);
      if (doc.body) doc.body.setAttribute("data-exercise", incoming.getAttribute("data-exercise"));

      outgoing.style.setProperty("--dir", dir);
      incoming.style.setProperty("--dir", dir);

      if (reduced) {
        outgoing.classList.remove("is-current");
        incoming.classList.add("is-current");
        showOnly(panels, index);
        showOnly(metrics, index);
        showOnly(previews, (index + 1) % total);
        if (count) count.textContent = String(index + 1).padStart(2, "0");
        current = index;
        return;
      }

      animating = true;

      outgoing.classList.remove("is-current");
      outgoing.classList.add("is-leaving");
      if (info) info.classList.add("is-swapping");

      // swap the text while the stage is empty, then bring the new figure in
      var swapTimer = setTimeout(function () {
        showOnly(panels, index);
        showOnly(metrics, index);
        showOnly(previews, (index + 1) % total);
        if (count) count.textContent = String(index + 1).padStart(2, "0");
        if (info) info.classList.remove("is-swapping");

        incoming.classList.add("is-current");
        incoming.classList.add("is-entering");
        // force a reflow so the entrance animation restarts every time
        void incoming.offsetWidth;
      }, 380);

      var settleTimer = setTimeout(function () {
        outgoing.classList.remove("is-leaving");
        outgoing.style.removeProperty("--dir");
        incoming.classList.remove("is-entering");
        animating = false;
        remeasure();
      }, 1350);

      cleanups.push(function () {
        clearTimeout(swapTimer);
        clearTimeout(settleTimer);
      });

      current = index;
    }

    tabs.forEach(function (tab) {
      on(tab, "click", function () {
        switchExercise(Number(tab.getAttribute("data-index")));
      });
    });

    Array.prototype.forEach.call(explorer.querySelectorAll(".arrow"), function (arrow) {
      var dir = Number(arrow.getAttribute("data-dir"));
      on(arrow, "click", function () {
        switchExercise(current + dir, dir);
      });
    });

    Array.prototype.forEach.call(explorer.querySelectorAll(".explorer__preview"), function (preview) {
      on(preview, "click", function () {
        switchExercise(current + 1, 1);
      });
    });

    on(explorer, "keydown", function (event) {
      if (event.key === "ArrowRight") switchExercise(current + 1, 1);
      if (event.key === "ArrowLeft") switchExercise(current - 1, -1);
    });

    // Initial paint
    if (figures.length) {
      figures[0].classList.add("is-current");
      setAtmosphere(figures[0]);
    }
    showOnly(panels, 0);
    showOnly(metrics, 0);
    showOnly(previews, 1 % Math.max(1, total));
  }

  /* ----------------------------------------------------------------------
     Kick everything off
     ---------------------------------------------------------------------- */
  remeasure();
  requestAnimationFrame(remeasure);
  setTimeout(remeasure, 400);
  setTimeout(remeasure, 1200);
})();
