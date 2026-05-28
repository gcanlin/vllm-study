(function () {
  const current = location.pathname.split("/").pop() || "index.html";
  for (const link of document.querySelectorAll(".study-topnav__link")) {
    const href = link.getAttribute("href") || "";
    const file = href === "./" ? "index.html" : href.split("/").pop();
    if (file === current) {
      link.setAttribute("aria-current", "page");
    } else {
      link.removeAttribute("aria-current");
    }
  }
})();
