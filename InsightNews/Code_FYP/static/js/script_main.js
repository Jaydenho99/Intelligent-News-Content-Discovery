'use strict';

/* variables */
const navOpenBtn = document.querySelector("[data-menu-open-btn]");
const navCloseBtn = document.querySelector("[data-menu-close-btn]");
const navbar=document.querySelector("[data-navbar]");
const overlay=document.querySelector("[data-overlay]");
const navElemArr=[navOpenBtn, navCloseBtn, overlay];

for(let i = 0; i<navElemArr.length; i++){

    navElemArr[i].addEventListener("click",function() {

        navbar.classList.toggle("active");
        overlay.classList.toggle("active");
        document.classList.toggle("active");
    });
}

/* header sticky */
const header=document.querySelector("[data-header]");

window.addEventListener("scroll",function(){
    window.scrollY >= 10 ? header.classList.add("active") : header.classList.remove("active");
});

/* go top */
const goTopBtn = document.querySelector("[data-go-top]");
window.addEventListener("scroll",function() {

    window.scrollY >= 500 ? goTopBtn.classList.add("active") : goTopBtn.classList.remove("active");

});


/* pop-up cards elements when text is clicked */
let previewContainer=document.querySelector('.products-preview');
let previewBox=previewContainer.querySelectorAll('.preview');

document.querySelectorAll('.products-container .product').forEach(product =>{
    product.onclick=() =>{
        previewContainer.style.display='flex';
        let name=product.getAttribute('data-name');
        previewBox.forEach(preview => {
            let target = preview.getAttribute('data-target');
            if(name==target){
                preview.classList.add('active-card');
            }
        });

    };
});

/* pop-up cards elements for news contents in topic sentiment */
/* positive sentiment */
document.querySelectorAll('.title-sentiment-positive').forEach(product =>{
    product.onclick=() =>{
        previewContainer.style.display='flex';
        let name=product.getAttribute('data-name');
        previewBox.forEach(preview => {
            let target = preview.getAttribute('data-target');
            if(name==target){
                preview.classList.add('active-card');
            }
        });

    };
});

/* neutral sentiment */
document.querySelectorAll('.title-sentiment-neutral').forEach(product2 =>{
    product2.onclick=() =>{
        previewContainer.style.display='flex';
        let name=product2.getAttribute('data-name');
        previewBox.forEach(preview => {
            let target = preview.getAttribute('data-target');
            if(name==target){
                preview.classList.add('active-card');
            }
        });

    };
});

/* negative sentiment */
document.querySelectorAll('.title-sentiment-negative').forEach(product =>{
    product.onclick=() =>{
        previewContainer.style.display='flex';
        let name=product.getAttribute('data-name');
        previewBox.forEach(preview => {
            let target = preview.getAttribute('data-target');
            if(name==target){
                preview.classList.add('active-card');
            }
        });

    };
});

/* pop-up cards elements for news contents in keyword analysis */
document.querySelectorAll('.keyword-show').forEach(product =>{
    product.onclick=() =>{
        previewContainer.style.display='flex';
        let name=product.getAttribute('data-name');
        previewBox.forEach(preview => {
            let target = preview.getAttribute('data-target');
            if(name==target){
                preview.classList.add('active-card');
            }
        });

    };
});

document.querySelectorAll('.keyword-show-results').forEach(product =>{
    product.onclick=() =>{
        previewContainer.style.display='flex';
        let name=product.getAttribute('data-name');
        previewBox.forEach(preview => {
            let target = preview.getAttribute('data-target');
            if(name==target){
                preview.classList.add('active-card');
            }
        });

    };
});

previewBox.forEach(close =>{
    close.querySelector('.close-button-ion').onclick = () =>{
        close.classList.remove('active-card');
        previewContainer.style.display = 'none';
    };
});







