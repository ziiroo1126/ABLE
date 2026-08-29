document.documentElement.classList.add("js");

const header = document.querySelector(".site-header");
const navToggle = document.querySelector(".nav-toggle");
const navLinks = document.querySelector(".nav-links");

function closeNavigation() {
  if (!navToggle || !navLinks) return;
  navLinks.classList.remove("is-open");
  navToggle.setAttribute("aria-expanded", "false");
}

if (navToggle && navLinks) {
  navToggle.addEventListener("click", () => {
    const isOpen = navLinks.classList.toggle("is-open");
    navToggle.setAttribute("aria-expanded", String(isOpen));
  });

  navLinks.querySelectorAll("a").forEach((link) => {
    link.addEventListener("click", closeNavigation);
  });
}

window.addEventListener(
  "scroll",
  () => header?.classList.toggle("is-scrolled", window.scrollY > 12),
  { passive: true },
);

document.querySelectorAll("[data-copy-target]").forEach((button) => {
  button.addEventListener("click", async () => {
    const target = document.querySelector(button.dataset.copyTarget);
    if (!target) return;

    const originalLabel = button.textContent;
    try {
      await navigator.clipboard.writeText(target.textContent.trim());
      button.textContent = "Copied";
    } catch (_error) {
      button.textContent = "Select to copy";
      const selection = window.getSelection();
      const range = document.createRange();
      range.selectNodeContents(target);
      selection.removeAllRanges();
      selection.addRange(range);
    }

    window.setTimeout(() => {
      button.textContent = originalLabel;
    }, 1800);
  });
});

const revealItems = document.querySelectorAll("[data-reveal]");
if ("IntersectionObserver" in window) {
  const revealObserver = new IntersectionObserver(
    (entries, observer) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    },
    { rootMargin: "0px 0px -8%", threshold: 0.08 },
  );
  revealItems.forEach((item) => revealObserver.observe(item));
} else {
  revealItems.forEach((item) => item.classList.add("is-visible"));
}

const lightbox = document.querySelector("[data-lightbox-dialog]");
const lightboxImage = lightbox?.querySelector("img");

document.querySelectorAll("[data-lightbox]").forEach((button) => {
  button.addEventListener("click", () => {
    if (!lightbox || !lightboxImage) return;
    lightboxImage.src = button.dataset.lightbox;
    lightboxImage.alt = button.dataset.lightboxAlt || "Expanded research figure";
    if (typeof lightbox.showModal === "function") {
      lightbox.showModal();
    } else {
      lightbox.setAttribute("open", "");
    }
  });
});

lightbox?.querySelector(".lightbox-close")?.addEventListener("click", () => {
  lightbox.close();
});

lightbox?.addEventListener("click", (event) => {
  if (event.target === lightbox) lightbox.close();
});

const currentYear = document.querySelector("[data-year]");
if (currentYear) currentYear.textContent = String(new Date().getFullYear());
