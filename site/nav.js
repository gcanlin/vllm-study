(function () {
  const links = document.querySelector(".study-topnav__links");
  if (links && !links.querySelector('a[href="./vllm_sampler.html"]')) {
    const samplerLink = document.createElement("a");
    samplerLink.className = "study-topnav__link";
    samplerLink.href = "./vllm_sampler.html";
    samplerLink.textContent = "Sampler";

    const vitLink = links.querySelector('a[href="./vllm_vit_dp.html"]');
    if (vitLink) {
      links.insertBefore(samplerLink, vitLink);
    } else {
      links.appendChild(samplerLink);
    }
  }

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
