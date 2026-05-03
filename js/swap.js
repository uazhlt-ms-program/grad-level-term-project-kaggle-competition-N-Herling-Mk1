/* =============================================================
   CRUCIBLE — swap.js
   View-swap engine + Mermaid integration.
   ============================================================= */
(function () {
  'use strict';

  const NODE_TO_VIEW = {
    S1: 'stage1', S2: 'stage2', S3: 'stage3', S4: 'stage4',
    S5: 'stage5', S6: 'stage6', S7: 'stage7', S8: 'stage8'
  };

  function showView(name) {
    document.querySelectorAll('.view').forEach(v => v.classList.remove('active'));
    const target = document.getElementById('view-' + name);
    if (!target) {
      const tree = document.getElementById('view-tree');
      if (tree) tree.classList.add('active');
      return;
    }
    target.classList.add('active');
    if (name === 'tree') {
      history.replaceState(null, '', window.location.pathname);
    } else {
      history.replaceState(null, '', '#' + name);
    }
  }

  window.showView = showView;

  async function attachMermaidClicks() {
    if (typeof mermaid === 'undefined') {
      console.warn('[CRUCIBLE] Mermaid not loaded');
      return;
    }
    await mermaid.run({ querySelector: '.mermaid' });
    document.querySelectorAll('.mermaid g.node').forEach(g => {
      const m = g.id.match(/flowchart-(S\d+)-/);
      if (!m) return;
      const view = NODE_TO_VIEW[m[1]];
      if (!view) return;
      g.style.cursor = 'pointer';
      g.addEventListener('click', () => showView(view));
    });
  }

  window.addEventListener('DOMContentLoaded', () => {
    if (typeof mermaid !== 'undefined') {
      mermaid.initialize({
        startOnLoad: false,
        theme: 'default',
        flowchart: {
          curve: 'basis', padding: 10,
          nodeSpacing: 40, rankSpacing: 50,
          htmlLabels: true
        }
      });
      attachMermaidClicks().then(() => {
        const hash = window.location.hash.replace('#', '');
        if (hash && document.getElementById('view-' + hash)) showView(hash);
      });
    }
  });

  window.addEventListener('hashchange', () => {
    const hash = window.location.hash.replace('#', '');
    showView(hash || 'tree');
  });
})();
