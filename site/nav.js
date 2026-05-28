(function () {
  const pages = [
    { href: "./", file: "index.html", label: "DeepSeek V3 FusedMoE" },
    { href: "./qwen3_model_runner.html", file: "qwen3_model_runner.html", label: "Qwen3 Runner" },
    { href: "./qwen3_structure.html", file: "qwen3_structure.html", label: "Qwen3 Structure" },
    { href: "./vllm_cp_pcp_dcp.html", file: "vllm_cp_pcp_dcp.html", label: "CP / PCP / DCP" },
    { href: "./vllm_prefix_cache.html", file: "vllm_prefix_cache.html", label: "Prefix Cache" },
  ];

  const current = location.pathname.split("/").pop() || "index.html";
  const nav = document.createElement("nav");
  nav.className = "study-topnav";
  nav.setAttribute("aria-label", "Study note navigation");

  const brand = document.createElement("a");
  brand.className = "study-topnav__brand";
  brand.href = "./";
  brand.textContent = "vLLM Study";
  nav.appendChild(brand);

  const links = document.createElement("div");
  links.className = "study-topnav__links";

  for (const page of pages) {
    const link = document.createElement("a");
    link.className = "study-topnav__link";
    link.href = page.href;
    link.textContent = page.label;
    if (page.file === current) {
      link.setAttribute("aria-current", "page");
    }
    links.appendChild(link);
  }

  nav.appendChild(links);
  document.body.insertBefore(nav, document.body.firstChild);
})();
