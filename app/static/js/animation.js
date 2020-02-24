var btn =  document.getElementById("git-btn");

// btn.style.setProperty("top", "100px");
// btn.style.setProperty("left", "100px");


function makeNewPosition(){
    
    // Get viewport dimensions (remove the dimension of the div)
    // var h = $(window).height() - 50;
    // var w = $(window).width() - 50;
    var h = 685;
    var w = 365;
    
    var nh = Math.floor(Math.random() * h);
    var nw = Math.floor(Math.random() * w);
    
    return [nh,nw];    
    
}

function animateDiv(myclass){
    var newq = makeNewPosition();
    $(myclass).animate({ top: newq[0], left: newq[1] }, 3000, function(){
      animateDiv(myclass);        
    });
    
};

animateDiv('.git-btn');