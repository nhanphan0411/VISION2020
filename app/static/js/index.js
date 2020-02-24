function readURL(input) {
    if (input.files && input.files[0]) {
        var reader = new FileReader();
        reader.onload = function (e) {
            $('#blah')
				.attr('src', e.target.result)
				.attr('style', 'text-align: center')
                .width(300);
        };
        reader.readAsDataURL(input.files[0]);
    }

}

var btn_submit = document.getElementById("myButton");
btn_submit.addEventListener("click", function(event){
  event.preventDefault();
  var content = document.getElementById("blah").src;
  // console.log(content);
  fetch('/upload', {
    method: 'post',
    contentType: 'application/json; charset=utf-8',
    body: JSON.stringify(content)
  }).then(res=>res.json())
    .then(res =>{
      // console.log(res);
      // var image = new Image();
      // image.src = 'data:image/png;base64,'+res;
      var image = document.getElementById("prediction")
      // .setAttribute("src", 'data:image/png;base64,'+res)
      // .setAttribute('text-align: center')
      // .width(1920);
      image.style.display = "block"
      image.style.width = "100vw"
      image.src = 'data:image/png;base64,'+res
      //document.body.appendChild(image);
    });
  console.log("sent it")
});


