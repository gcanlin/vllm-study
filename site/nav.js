(function () {
  const navItems = [
    { href: "./", file: "index.html", label: "DeepSeek V3 FusedMoE" },
    {
      href: "./qwen3_model_runner.html",
      file: "qwen3_model_runner.html",
      label: "Qwen3 Runner",
    },
    {
      href: "./qwen3_structure.html",
      file: "qwen3_structure.html",
      label: "Qwen3 Structure",
    },
    {
      href: "./vllm_cp_pcp_dcp.html",
      file: "vllm_cp_pcp_dcp.html",
      label: "CP / PCP / DCP",
    },
    {
      href: "./vllm_prefix_cache.html",
      file: "vllm_prefix_cache.html",
      label: "Prefix Cache",
    },
    { href: "./vllm_sampler.html", file: "vllm_sampler.html", label: "Sampler" },
    { href: "./vllm_vit_dp.html", file: "vllm_vit_dp.html", label: "ViT DP" },
    { href: "./vime_framework.html", file: "vime_framework.html", label: "Vime" },
  ];

  function currentFile() {
    const file = location.pathname.split("/").pop();
    return file || "index.html";
  }

  function renderLinks(links) {
    const current = currentFile();
    links.replaceChildren();

    for (const item of navItems) {
      const link = document.createElement("a");
      link.className = "study-topnav__link";
      link.href = item.href;
      link.textContent = item.label;
      if (item.file === current) {
        link.setAttribute("aria-current", "page");
      }
      links.appendChild(link);
    }
  }

  function initNav() {
    let nav = document.querySelector(".study-topnav");
    if (!nav) {
      nav = document.createElement("nav");
      nav.className = "study-topnav";
      nav.setAttribute("aria-label", "Study note navigation");
      nav.innerHTML =
        '<a class="study-topnav__brand" href="./">vLLM Study</a>' +
        '<div class="study-topnav__links"></div>';
      document.body.prepend(nav);
    }

    const links = nav.querySelector(".study-topnav__links");
    if (links) {
      renderLinks(links);
      return;
    }

    const current = currentFile();
    for (const link of nav.querySelectorAll(".study-topnav__link")) {
      const href = link.getAttribute("href") || "";
      const file = href === "./" ? "index.html" : href.split("/").pop();
      link.removeAttribute("aria-current");
      if (file === current) {
        link.setAttribute("aria-current", "page");
      }
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initNav, { once: true });
  } else {
    initNav();
  }
})();
