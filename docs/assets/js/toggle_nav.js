//document.querySelectorAll(".nav-item").forEach((ele) =>
//  ele.addEventListener("click", function (event) {
//    event.preventDefault();
//    document
//      .querySelectorAll(".nav-item")
//      .forEach((ele) => ele.classList.remove("active"));
//    this.classList.add("active")
//  })
//);

$(document).ready(function () {
  divId = $(location).attr('href').substring($(location).attr('href').indexOf("#"));
console.log(divId)
  $('html, body').animate({
    scrollTop: $(divId).offset().top - 55
  }, 'slow');
  $('.nav-link').removeClass('active');
  $('a[href$="'+divId+'"]').addClass('active');
});

$('.nav-link, a[href="#install"]').click(function(e) {
  divId = $(this).attr('href');
  divId = divId.substring(divId.indexOf("#"));
  $('html, body').animate({
    scrollTop: $(divId).offset().top - 55
  }, 100);
  $('.nav-link').removeClass('active');
  $(this).addClass('active');
});

$('.navbar-collapse a').click(function(){
  $(".navbar-collapse").collapse('hide');
});
