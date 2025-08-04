import { OverlayScrollbars } from "overlayscrollbars";

const overlay = () => {

  window.osInst = OverlayScrollbars(document.body, {
    overflow: {
      x: "hidden",
      y: "scroll",
    },
    scrollbars: {
      theme: "os-theme-body",
    },
  });
};

const obsIsnt = (entries, observer) => {

  entries.filter(index=> index.isIntersecting).forEach(index => {

    index.target.classList.add("scrolled");
    observer.unobserve(index.target);
  });
};

const scrolled = (obj, bool) => {

  obj.forEach(index => {

    new IntersectionObserver(obsIsnt, {
      rootMargin: bool ? `${index.offsetTop}px` : "0px",
    }).observe(index);
  });
};

export const init = () => {

  //scrolled(document.querySelectorAll(".scrolled-init"), false);
  overlay();
};