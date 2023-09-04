
/* variables */
const sign_in_btn = document.querySelector("#sign-in-btn");
const sign_up_btn = document.querySelector("#sign-up-btn");
const container = document.querySelector(".container");

/* allow the page to slide to right when sign up button is clicked */
sign_up_btn.addEventListener("click", () => {
  container.classList.add("sign-up-mode");
});

/* allow the page to slide to left when sign in button is clicked */
sign_in_btn.addEventListener("click", () => {
  container.classList.remove("sign-up-mode");
});