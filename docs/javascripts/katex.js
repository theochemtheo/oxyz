// Arithmatex preserves delimiters; KaTeX renders only marked mathematics.
document$.subscribe(({ body }) => {
  for (const element of body.querySelectorAll(".arithmatex")) {
    renderMathInElement(element, {
      delimiters: [
        { left: "\\[", right: "\\]", display: true },
        { left: "\\(", right: "\\)", display: false }
      ]
    });
  }
});
