/* =========================================
   Animated neural-network background.
   Lightweight Canvas API — no deps.
   Respects prefers-reduced-motion.
   ========================================= */

(function () {
    const canvas = document.getElementById('neural-bg');
    if (!canvas) return;

    const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduceMotion) {
        canvas.style.display = 'none';
        return;
    }

    const ctx = canvas.getContext('2d', { alpha: true });
    const dpr = Math.min(window.devicePixelRatio || 1, 2);

    let width = 0;
    let height = 0;
    let nodes = [];
    let mouse = { x: -1000, y: -1000 };
    let rafId = null;

    function resize() {
        width = window.innerWidth;
        height = window.innerHeight;
        canvas.width = width * dpr;
        canvas.height = height * dpr;
        canvas.style.width = `${width}px`;
        canvas.style.height = `${height}px`;
        ctx.scale(dpr, dpr);

        // Density tuned to screen area but capped for perf
        const target = Math.min(Math.floor((width * height) / 14000), 90);
        nodes = Array.from({ length: target }, () => spawn());
    }

    function spawn() {
        return {
            x: Math.random() * width,
            y: Math.random() * height,
            vx: (Math.random() - 0.5) * 0.25,
            vy: (Math.random() - 0.5) * 0.25,
            radius: Math.random() * 1.6 + 0.6,
            // Mix purple + cyan
            hue: Math.random() > 0.6 ? 188 : 270,
        };
    }

    function step() {
        ctx.clearRect(0, 0, width, height);

        // Update + draw nodes
        for (const n of nodes) {
            n.x += n.vx;
            n.y += n.vy;

            // Wrap
            if (n.x < 0) n.x = width;
            else if (n.x > width) n.x = 0;
            if (n.y < 0) n.y = height;
            else if (n.y > height) n.y = 0;

            // Mouse repel — very subtle
            const dx = n.x - mouse.x;
            const dy = n.y - mouse.y;
            const distSq = dx * dx + dy * dy;
            if (distSq < 14400) { // 120px
                const dist = Math.sqrt(distSq);
                const force = (120 - dist) / 120;
                n.x += (dx / dist) * force * 1.5;
                n.y += (dy / dist) * force * 1.5;
            }

            ctx.beginPath();
            ctx.fillStyle = `hsla(${n.hue}, 85%, 65%, 0.7)`;
            ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
            ctx.fill();
        }

        // Connect nearby nodes with subtle lines.
        // O(n^2) but with n ~ 90 this is fine; we early-exit on dist.
        const linkDistSq = 22500; // 150px squared
        for (let i = 0; i < nodes.length; i++) {
            for (let j = i + 1; j < nodes.length; j++) {
                const a = nodes[i];
                const b = nodes[j];
                const dx = a.x - b.x;
                const dy = a.y - b.y;
                const dSq = dx * dx + dy * dy;
                if (dSq < linkDistSq) {
                    const alpha = (1 - dSq / linkDistSq) * 0.35;
                    ctx.strokeStyle = `hsla(${(a.hue + b.hue) / 2}, 70%, 60%, ${alpha})`;
                    ctx.lineWidth = 0.7;
                    ctx.beginPath();
                    ctx.moveTo(a.x, a.y);
                    ctx.lineTo(b.x, b.y);
                    ctx.stroke();
                }
            }
        }

        rafId = requestAnimationFrame(step);
    }

    // Throttled resize handler
    let resizeTimer = null;
    function onResize() {
        clearTimeout(resizeTimer);
        resizeTimer = setTimeout(resize, 150);
    }

    function onMouseMove(e) {
        mouse.x = e.clientX;
        mouse.y = e.clientY;
    }

    function onMouseLeave() {
        mouse.x = -1000;
        mouse.y = -1000;
    }

    // Pause when tab is hidden — saves CPU/battery
    function onVisibilityChange() {
        if (document.hidden) {
            cancelAnimationFrame(rafId);
            rafId = null;
        } else if (!rafId) {
            rafId = requestAnimationFrame(step);
        }
    }

    window.addEventListener('resize', onResize);
    window.addEventListener('mousemove', onMouseMove, { passive: true });
    window.addEventListener('mouseleave', onMouseLeave, { passive: true });
    document.addEventListener('visibilitychange', onVisibilityChange);

    resize();
    rafId = requestAnimationFrame(step);
})();
