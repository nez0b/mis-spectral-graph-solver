document$.subscribe(() => {
  renderMathInElement(document.body, {
    // Delimiters for inline and display math
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true }
    ],
    // KaTeX options
    throwOnError: false,
    errorColor: "#cc0000",
    strict: "warn",
    output: "html",
    trust: false,
    macros: {
      // Common mathematical notation shortcuts
      "\\RR": "\\mathbb{R}",
      "\\NN": "\\mathbb{N}",
      "\\ZZ": "\\mathbb{Z}",
      "\\QQ": "\\mathbb{Q}",
      "\\CC": "\\mathbb{C}",
      "\\argmax": "\\operatorname{argmax}",
      "\\argmin": "\\operatorname{argmin}",
      "\\simplex": "\\Delta_n",
      "\\clique": "\\omega(G)",
      "\\indep": "\\alpha(G)",
      "\\proj": "\\Pi_{\\Delta_n}"
    }
  });
});

// Optional: Add math rendering status indicator
window.addEventListener('load', function() {
  console.log('KaTeX math rendering initialized');
});