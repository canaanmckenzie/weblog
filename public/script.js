document.addEventListener('DOMContentLoaded', () => {

  // ==========================================
  // PROCEDURAL HAND-DRAWN UNDERLINES
  // ==========================================

  /**
   * Simple seeded random number generator
   */
  function seededRandom(seed) {
    const x = Math.sin(seed) * 10000;
    return x - Math.floor(x);
  }

  /**
   * Hash a string to get a consistent numeric seed.
   * 
   * WHY WE DO THIS:
   * We want the "hand-drawn" randomnes to be DETERMINISTIC.
   * If we just used Math.random(), the squiggly lines would change every time
   * you refreshed the page, which feels glitchy.
   * 
   * By hashing the link text + URL, "About" always gets the exact same squiggle.
   */
  function hashString(str) {
    let hash = 0;
    // Standard "djb2" like hash algorithm
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32bit integer
    }
    return Math.abs(hash);
  }

  /**
   * Generate a unique hand-drawn underline SVG using Rough.js
   */
  function generateHandDrawnUnderline(seed) {
    if (typeof rough === 'undefined') return ''; // Safety check

    const gen = rough.generator();
    const width = 200;
    const height = 12;

    // Generate a rough line with high roughness for "sketchy" look
    const drawable = gen.line(0, 6, width, 6, {
      roughness: 2.2,
      bowing: 1.5,
      stroke: '#333',
      strokeWidth: 2,
      disableMultiStroke: false,
      seed: Math.floor(seed)
    });

    // Convert Rough.js drawable to SVG path data
    let pathD = '';
    if (drawable.sets && drawable.sets.length > 0) {
      drawable.sets.forEach(set => {
        set.ops.forEach(op => {
          const data = op.data;
          switch (op.op) {
            case 'move':
              pathD += `M ${data[0]} ${data[1]} `;
              break;
            case 'bcurveTo':
              pathD += `C ${data[0]} ${data[1]}, ${data[2]} ${data[3]}, ${data[4]} ${data[5]} `;
              break;
            case 'lineTo':
              pathD += `L ${data[0]} ${data[1]} `;
              break;
          }
        });
      });
    }

    // Create SVG 
    const svg = `<svg xmlns='http://www.w3.org/2000/svg' viewBox='-5 -5 ${width + 10} ${height + 10}' preserveAspectRatio='none'><path d='${pathD.trim()}' stroke='#333' stroke-width='1.5' fill='none' stroke-linecap='round' stroke-linejoin='round'/></svg>`;

    return `url("data:image/svg+xml,${encodeURIComponent(svg)}")`;
  }

  /**
   * Apply unique underlines to all links on the page.
   * 
   * LOGIC:
   * We iterate through every link that matches our "hand-drawn worthy" criteria.
   * For each link, we generate a unique SVG background image based on its text
   * and an index-based seed. This ensures the squiggles are consistent but distinct.
   */
  function applyHandDrawnUnderlines() {
    // SELECTING LINKS
    // We want to apply the hand-drawn effect to most links, but NOT all.
    // Exclusions:
    // 1. .nav-logo: The site logo should be clean
    // 2. .back-arrow: The hand-drawn back arrow at top of posts
    // 3. [data-footnote-ref]: Footnote numbers in text (marked-footnote uses data attribs)
    // 4. [data-footnote-backref]: Back arrows in footer
    const links = document.querySelectorAll(`
      a:not(.nav-logo):not(.back-arrow):not([data-footnote-ref]):not([data-footnote-backref])
    `);

    links.forEach((link, index) => {
      // Create a deterministic seed from the link's content
      const textHash = hashString(link.textContent + link.href);
      const seed = textHash * 0.001 + index * 31.7;

      // Apply the generated SVG as a background image
      link.style.backgroundImage = generateHandDrawnUnderline(seed);
    });
  }

  // Apply underlines immediately
  applyHandDrawnUnderlines();

  /**
   * LINK PREFETCHING
   * To make navigation feel instant, we fetch the page content
   * the moment the user hovers over a link.
   */
  const prefetchCache = new Set();

  function prefetchLink(url) {
    if (prefetchCache.has(url)) return;

    // Only prefetch local links
    try {
      const targetUrl = new URL(url, window.location.href);
      if (targetUrl.origin !== window.location.origin) return;

      prefetchCache.add(url);

      // Low priority fetch to prime the browser cache
      fetch(url, { priority: 'low' }).catch(() => { });
    } catch (e) {
      // Ignore invalid URLs
    }
  }

  // Attach hover listeners to all links
  document.querySelectorAll('a').forEach(link => {
    link.addEventListener('mouseenter', () => {
      if (link.href) prefetchLink(link.href);
    });
  });

  // ==========================================
  // ROUGH NOTATION ANNOTATIONS
  // ==========================================
  if (typeof RoughNotation !== 'undefined') {
    const { annotate, annotationGroup } = RoughNotation;
    const annotations = [];

    // Helper to safely add annotation if element exists
    const addAnnotation = (id, options) => {
      const el = document.getElementById(id);
      if (el) {
        return annotate(el, options);
      }
      return null;
    };

    // Hero Section Annotations (Index Page)
    const highlightCraft = addAnnotation('highlight-craft', {
      type: 'highlight',
      color: 'rgba(0, 0, 0, 0.1)',
      strokeWidth: 2,
      iterations: 2,
      multiline: true
    });
    if (highlightCraft) annotations.push(highlightCraft);

    const circleOrganic = addAnnotation('underline-organic', {
      type: 'highlight',
      color: 'rgba(0, 0, 0, 0.12)',
      strokeWidth: 2,
      iterations: 2,
      multiline: true
    });
    if (circleOrganic) annotations.push(circleOrganic);

    const boxClarity = addAnnotation('underline-clarity', {
      type: 'box',
      color: '#444',
      strokeWidth: 1.8,
      padding: 4,
      iterations: 2,
      multiline: true
    });
    if (boxClarity) annotations.push(boxClarity);

    const quoteBox = addAnnotation('quote-text', {
      type: 'bracket',
      color: '#444',
      strokeWidth: 2,
      brackets: ['left', 'right'],
      padding: 8,
      iterations: 2
    });
    if (quoteBox) annotations.push(quoteBox);

    // Blog Page Annotations (if any specific ones are added)
    // For now, none specific, but the mechanism is extensible

    // Animate
    if (annotations.length > 0) {
      const group = annotationGroup(annotations);
      setTimeout(() => {
        group.show();
      }, 500);
    }
  }

  // ==========================================
  // ROUGH.JS CARD ICONS
  // ==========================================
  if (typeof rough !== 'undefined') {
    const drawRoughIcon = (canvasId, drawFn) => {
      const container = document.getElementById(canvasId);
      if (!container) return;

      const canvas = document.createElement('canvas');
      canvas.width = 70;
      canvas.height = 70;
      container.appendChild(canvas);

      const rc = rough.canvas(canvas);
      drawFn(rc);
    };

    // Icon 1
    drawRoughIcon('icon-1', (rc) => {
      rc.circle(35, 35, 48, {
        stroke: '#222',
        strokeWidth: 2.5,
        roughness: 2.8,
        bowing: 2,
        fill: 'rgba(0,0,0,0.04)',
        fillStyle: 'solid'
      });
      rc.circle(36, 34, 10, {
        stroke: '#222', strokeWidth: 2, roughness: 2, fill: '#222', fillStyle: 'solid'
      });
    });

    // Icon 2
    drawRoughIcon('icon-2', (rc) => {
      rc.rectangle(12, 12, 46, 46, {
        stroke: '#222', strokeWidth: 2.5, roughness: 3.2, bowing: 1.5
      });
      rc.line(12, 58, 58, 12, {
        stroke: '#222', strokeWidth: 2.5, roughness: 2.8, bowing: 2
      });
    });

    // Icon 3
    drawRoughIcon('icon-3', (rc) => {
      rc.path('M 5 40 Q 18 12, 32 35 Q 46 58, 62 28', {
        stroke: '#222', strokeWidth: 3, roughness: 3.5, bowing: 2.5, fill: 'none'
      });
      rc.circle(62, 28, 9, {
        stroke: '#222', strokeWidth: 2.5, roughness: 2.5, fill: '#222', fillStyle: 'solid'
      });
    });
  }

  // ==========================================
  // ROUGH.JS BACK ARROWS
  // ==========================================
  const drawBackArrow = (el) => {
    if (el.querySelector('canvas')) return;

    const canvas = document.createElement('canvas');
    canvas.width = 120;
    canvas.height = 60;
    el.prepend(canvas);

    const y = 30 + (Math.random() * 8 - 4);
    const x1 = 12 + (Math.random() * 6 - 3);
    const x2 = 108 + (Math.random() * 6 - 3);

    if (typeof rough !== 'undefined') {
      const rc = rough.canvas(canvas);
      rc.line(x2, y, x1, y, {
        stroke: '#222',
        strokeWidth: 2.6,
        roughness: 2.6,
        bowing: 1.8
      });
      rc.line(x1, y, x1 + 12, y - 10, {
        stroke: '#222',
        strokeWidth: 2.6,
        roughness: 2.6,
        bowing: 1.8
      });
      rc.line(x1, y, x1 + 12, y + 10, {
        stroke: '#222',
        strokeWidth: 2.6,
        roughness: 2.6,
        bowing: 1.8
      });
      return;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.strokeStyle = '#222';
    ctx.lineWidth = 2.4;
    ctx.lineCap = 'round';

    const jitter = () => (Math.random() * 2 - 1);
    ctx.beginPath();
    ctx.moveTo(x2 + jitter(), y + jitter());
    ctx.lineTo(x1 + jitter(), y + jitter());
    ctx.lineTo(x1 + 12 + jitter(), y - 10 + jitter());
    ctx.moveTo(x1 + jitter(), y + jitter());
    ctx.lineTo(x1 + 12 + jitter(), y + 10 + jitter());
    ctx.stroke();
  };

  document.querySelectorAll('.back-arrow').forEach(drawBackArrow);

});
