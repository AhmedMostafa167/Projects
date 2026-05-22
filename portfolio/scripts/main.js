/* =========================================
   Main interactions
   - Nav scroll state + mobile toggle
   - Reveal-on-scroll (IntersectionObserver)
   - Counter animations
   - Typed-text rotator
   - Mouse-tracked bento-card glow
   ========================================= */

(function () {
    'use strict';

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // ---------- Nav: scroll state + mobile toggle ----------
    const nav = document.querySelector('.nav');
    const navToggle = document.querySelector('.nav__toggle');
    const navLinks = document.querySelector('.nav__links');

    function onScroll() {
        if (window.scrollY > 8) nav.classList.add('is-scrolled');
        else nav.classList.remove('is-scrolled');
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    onScroll();

    if (navToggle && navLinks) {
        navToggle.addEventListener('click', () => {
            const open = navLinks.classList.toggle('is-open');
            navToggle.setAttribute('aria-expanded', String(open));
        });

        // Close on link click (mobile)
        navLinks.querySelectorAll('a').forEach((a) => {
            a.addEventListener('click', () => {
                navLinks.classList.remove('is-open');
                navToggle.setAttribute('aria-expanded', 'false');
            });
        });
    }

    // ---------- Reveal-on-scroll with staggering ----------
    const revealEls = document.querySelectorAll('.reveal');

    // Pre-assign stagger indexes to children of grids so adjacent cards
    // come in at different times.
    document.querySelectorAll('.bento, .skills__grid, .timeline, .contact__grid').forEach((grid) => {
        const children = grid.querySelectorAll('.reveal');
        children.forEach((c, i) => {
            c.style.setProperty('--i', i);
        });
    });

    if ('IntersectionObserver' in window && !reduceMotion) {
        const io = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (entry.isIntersecting) {
                        entry.target.classList.add('is-visible');
                        io.unobserve(entry.target);
                    }
                });
            },
            { threshold: 0.12, rootMargin: '0px 0px -60px 0px' }
        );
        revealEls.forEach((el) => io.observe(el));
    } else {
        // No-IntersectionObserver fallback: just show everything
        revealEls.forEach((el) => el.classList.add('is-visible'));
    }

    // ---------- Animated counters ----------
    const counters = document.querySelectorAll('[data-counter]');
    if (counters.length && 'IntersectionObserver' in window && !reduceMotion) {
        const counterIo = new IntersectionObserver(
            (entries) => {
                entries.forEach((entry) => {
                    if (!entry.isIntersecting) return;
                    const el = entry.target;
                    const target = parseInt(el.dataset.counter, 10);
                    const suffix = el.dataset.suffix || '';
                    const duration = 1400;
                    const start = performance.now();

                    function tick(now) {
                        const elapsed = now - start;
                        const progress = Math.min(elapsed / duration, 1);
                        // Ease-out-cubic
                        const eased = 1 - Math.pow(1 - progress, 3);
                        const value = Math.round(target * eased);
                        el.textContent = value + suffix;
                        if (progress < 1) requestAnimationFrame(tick);
                    }
                    requestAnimationFrame(tick);
                    counterIo.unobserve(el);
                });
            },
            { threshold: 0.5 }
        );
        counters.forEach((c) => counterIo.observe(c));
    } else {
        counters.forEach((c) => {
            c.textContent = c.dataset.counter + (c.dataset.suffix || '');
        });
    }

    // ---------- Typed-text rotator (hero role) ----------
    const typingEl = document.querySelector('[data-typing]');
    if (typingEl && !reduceMotion) {
        let words = [];
        try {
            words = JSON.parse(typingEl.dataset.words);
        } catch (e) {
            words = ['NLP Engineer'];
        }

        let wordIdx = 0;
        let charIdx = 0;
        let deleting = false;

        function tick() {
            const word = words[wordIdx];
            if (!deleting) {
                charIdx++;
                typingEl.textContent = word.slice(0, charIdx);
                if (charIdx === word.length) {
                    deleting = true;
                    setTimeout(tick, 1600);
                    return;
                }
                setTimeout(tick, 80 + Math.random() * 60);
            } else {
                charIdx--;
                typingEl.textContent = word.slice(0, charIdx);
                if (charIdx === 0) {
                    deleting = false;
                    wordIdx = (wordIdx + 1) % words.length;
                    setTimeout(tick, 250);
                    return;
                }
                setTimeout(tick, 38);
            }
        }
        // Start with first word visible so the layout doesn't flash
        typingEl.textContent = words[0];
        charIdx = words[0].length;
        setTimeout(() => {
            deleting = true;
            tick();
        }, 2200);
    } else if (typingEl) {
        const words = JSON.parse(typingEl.dataset.words || '["NLP Engineer"]');
        typingEl.textContent = words[0];
    }

    // ---------- Mouse-tracked glow on bento cards ----------
    if (!reduceMotion) {
        document.querySelectorAll('.bento__card').forEach((card) => {
            card.addEventListener('mousemove', (e) => {
                const rect = card.getBoundingClientRect();
                const mx = ((e.clientX - rect.left) / rect.width) * 100;
                const my = ((e.clientY - rect.top) / rect.height) * 100;
                card.style.setProperty('--mx', `${mx}%`);
                card.style.setProperty('--my', `${my}%`);
            });
        });
    }
})();
