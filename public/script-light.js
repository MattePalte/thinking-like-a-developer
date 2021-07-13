// LOGGING ACTIVITY ------------------------------------------------
// NEED JQUERY in the script
// LOG ALSO WHEN SCROLLING

// VARIABLE DECLARATION

const MOUSE_EVERY_MILLI = 100;
const MIN_DELAY_SUBMISSION = 1000;
const MIN_TOKEN_INTERACTION_BEFORE_FLUSH = 1000;
const CHECK_TO_FLUSH_EVERY_MILLI = 20000;
const CHECK_CHEATING_AND_STATUS_EVERY_MILLI = 5000;

var xMousePos = 0;
var yMousePos = 0;
var lastScrolledLeft = 0;
var lastScrolledTop = 0;

const allMouseEvents = [];
const allOptionsInteractionEvents = [];
const allClickEvents = [];
const optionsNames = [];
const traceVisibleTokens = [];

// time starting time
const d = new Date();
const openedPageTime = d.getTime();
var lastMouseMovement = openedPageTime;

var pendingXHR = false;
var voluntaryChangeOfPage = false;
var readyToSubmit = false;
var checkEntireJavascriptExecution = 0;

const height_single_option = 35;
const options = document.getElementsByClassName("send-log-option");
const liOptions = document.getElementsByClassName("li-option-element");

var platformName = "";
var platformVersion = "";
var platformOs = "";

// CHUNK MANAGEMENT
var startChunkMouse = 0;
var startChunkTokenInteractions = 0;
var startChunkOptions = 0;
var startChunkVizTokens = 0;

var endChunkMouse = 0;
var endChunkTokenInteractions = 0;
var endChunkOptions = 0;
var endChunkVizTokens = 0;

// OBJECT DECLARATION

class MouseLog {
    constructor(x, y, time) {
        this.x = x;
        this.y = y;
        this.t = time;
    }
}

class VisibleTokensLog {
    constructor(n, t) {
        this.n = n;
        this.t = t;
    }
}

class TokenClickLog {
    constructor(position, token, time, clicked, over, distance, idgroup) {
        this.position = position;
        this.token = token;
        this.t = time;
        this.clicked = clicked;
        this.over = over;
        this.distance = parseInt(distance);
        this.idgroup = idgroup;
    }
}

class OptionHoverLog {
    constructor(option, time, over) {
        this.option = option;
        this.t = time;
        this.over = over;
    }
}


// GENERAL FUNCTIONS

function htmlDecode(input) {
    var doc = new DOMParser().parseFromString(input, "text/html");
    return doc.documentElement.textContent;
}


function logMouse(x, y) {
    let noLogArea = document.getElementById('navbar').offsetHeight;
    var d = new Date();
    var currAbsTime = d.getTime();
    var currentTimeEvent = currAbsTime - openedPageTime;
    var sinceLastMouseMovement = currAbsTime - lastMouseMovement;
    if (sinceLastMouseMovement < MOUSE_EVERY_MILLI) return;
    if (y < noLogArea) return;
    y = y - noLogArea;
    var pageCoords = "" + x + ", " + y + ", " + currentTimeEvent +";";
    //console.log(pageCoords);
    allMouseEvents.push(new MouseLog(x, y, currentTimeEvent));
    lastMouseMovement = currAbsTime;
}

function captureMousePosition(event){
    xMousePos = event.pageX;
    yMousePos = event.pageY;
    logMouse(xMousePos, yMousePos);
}


// Cickable tokens

function logClick(clickedToken, clicked, over, distance) {
    const d = new Date();
    const currentTimeEvent = d.getTime() - openedPageTime;
    const position = parseInt(clickedToken.id.split("_")[1]);
    const idgroup = parseInt(clickedToken.getAttribute("id4identifier").split("_")[1]);
    const token = clickedToken.innerHTML;
    //console.log((clicked == 1 ? " clicked" : "") + (over == 1 ? " over" : "") + " : " + token + ", " + position + " (group " + idgroup + "), " + currentTimeEvent);
    allClickEvents.push(new TokenClickLog(position, htmlDecode(token), currentTimeEvent, clicked, over, distance, idgroup));
}

function tokenPress(clicked_id) {
    const className = document.getElementById(clicked_id).className;
    console.log("Class name: " + className)
    if (className == "token") {
        // clicked
        document.getElementById(clicked_id).className = "token permanent_token";
        document.getElementById(clicked_id).style.color = "red";
        document.getElementById(clicked_id).style.textShadow = "none";
        logClick(document.getElementById(clicked_id), 1, 1);
    } else {
        // unclicked
        document.getElementById(clicked_id).className = "token";
        document.getElementById(clicked_id).style.color = "black";
        document.getElementById(clicked_id).style.textShadow = "none";
        logClick(document.getElementById(clicked_id), 0, 1);
    }
    return false;
}

// SEND THE LOGS TO THE SERVER

// FLUSH CURRENT CHUNK
function flushChunkToServer(chosenOption = "") {
    // DECLARE VARIABLE
    const fileIndex = document.getElementById("file-index").innerHTML;
    const uuid = document.getElementById("uuid").innerHTML;
    const formattedCode =  htmlDecode(document.getElementById("codebox").innerHTML.replace(/<[^>]*>?/gm, ''));
    const clickedtokens = [];
    const tokensText = [];
    var data = null;
    // check if we can submit, otherwise show guidelines
    checkStatus();
    if (!readyToSubmit) {
        alert('Follow the TO DOs before submitting. You can read the them above the options.');
        return;
    }
    // prepare request and endpoint
    var xhr = new XMLHttpRequest();
    var url = "/result-receiver";


    if (pendingXHR == true) {
        return;
    } else {
        pendingXHR = true;
    }

    // fix the end of the chunk we are sending
    endChunkMouse = allMouseEvents.length;
    endChunkTokenInteractions = allClickEvents.length;
    endChunkOptions = allOptionsInteractionEvents.length;
    endChunkVizTokens = traceVisibleTokens.length;

    // SHOW LOADING
    document.getElementById('xhr_status').innerHTML = "Communication with server...";
    document.getElementById('navbar').style.backgroundColor = '#d3d3d3';
    document.getElementById("loader_placeholder").classList.add('loader');
    document.getElementById("loader_placeholder").innerHTML = "Synchronizing data with server...";

    // get clicked tokens
    //all_tokens = document.getElementsByClassName("token");
    let all_tokens = document.querySelectorAll(".t,.token");
    for (let i = 0; i < all_tokens.length; i++) {
        if (all_tokens[i].className.toString().includes("permanent_token")) {
            clickedtokens.push(1);
        } else {
            clickedtokens.push(0);
        }
        console.log(all_tokens[i]);
        console.log(all_tokens[i].childNodes[0]);
        let textToken = "";
        try {
            textToken = all_tokens[i].childNodes[0].nodeValue;
        } catch (error) {
            textToken = "";
            console.log(error);
        }
        tokensText.push({
            'text': textToken,
            'id': all_tokens[i].id.split("_")[1],
            'line': all_tokens[i].getAttribute('line'),
            'charStart': all_tokens[i].getAttribute('charStart')
        });
    }
    // SEND DATA TO SERVER
    // Sending and receiving data in JSON format using POST method

    const d = new Date();
    const totalTimeForTask = d.getTime() - openedPageTime;

    console.log("fileIndex: " + fileIndex);
    data =
        JSON.stringify({id: fileIndex.trim(),
                        uuid: uuid.trim(),
                        functionnamebyuser: chosenOption,
                        neighborssize: NEIGHBORS_SIZE,
                        options: optionsNames,
                        optionsinteraction: allOptionsInteractionEvents.slice(startChunkOptions, endChunkOptions),
                        mousetrace: allMouseEvents.slice(startChunkMouse, endChunkMouse),
                        tokeninteraction: allClickEvents.slice(startChunkTokenInteractions, endChunkTokenInteractions),
                        finalclickedtokens: clickedtokens,
                        tokens: tokensText,
                        tottimemilli: totalTimeForTask,
                        timeopenpage: openedPageTime,
                        formattedcode: formattedCode,
                        platformname: platformName,
                        platformversion: platformVersion,
                        platformos: platformOs,
                        tracevisibletokens: traceVisibleTokens.slice(startChunkVizTokens, endChunkVizTokens),
                        completejsexecution: checkEntireJavascriptExecution});
    //console.log("data: " + data);
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        console.log("Status: " + xhr.status);
        console.log("readyState: " + xhr.readyState);
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log("Response received 200");
            console.log('Data Stored on Server successfully...');
            //SIMULATE NETWORK DELAYING
            var delay = 0;
            if (chosenOption != "") {
                delay = MIN_DELAY_SUBMISSION;
            }
            setTimeout(function () {
                // back to normal state
                pendingXHR = false;
                document.getElementById('xhr_status').innerHTML = "";

                // update chunk
                startChunkMouse = endChunkMouse;
                startChunkTokenInteractions = endChunkTokenInteractions;
                startChunkOptions = endChunkOptions;
                startChunkVizTokens = endChunkVizTokens;

                if (chosenOption != "") {
                    voluntaryChangeOfPage = true;
                    // console.log(formattedCode)
                    window.location.href = 'rate-difficulty?positionOfFile=' + (parseInt(fileIndex)).toString();
                } else {
                    document.getElementById('navbar').style.backgroundColor = '#fa7d00';
                    document.getElementById("loader_placeholder").classList.remove('loader');
                    document.getElementById("loader_placeholder").innerHTML = "";
                }

            //SIMULATE NETWORK DELAYING
            }, delay);

        }
    };
    xhr.send(data);

    // LOG CLIENT SIDE

    console.log("fileIndex: " + fileIndex);
    console.log("allMouseEvents length: " + allMouseEvents.length);

}


// USER INTERFACE ----------------------------------------------

function deblurr(element, dist) {
    if (element.style.color != "red") {
        element.style.color = "black";
        element.style.textShadow = "none";
        element.style.backgroundColor = "white";
    }
    logClick(element, 0, 1, dist);
};

function blurr(element, dist) {
    if (element.style.color != "red") {
        element.style.color = "transparent";
        element.style.textShadow = "0 0 7px rgba(0,0,0,0.5)";
        element.style.backgroundColor = "transparent";
    }
    logClick(element, 0, 0, dist);
};

function getNeighbors(currentID, radiusNeighbors) {
    const selectedTokens = [];
    const distances = [];
    const currentLine = document.getElementById(currentID).getAttribute("line");
    // console.log(currentLine);
    const numerical_id = parseInt(currentID.split('_')[1]);
    for (let i = 1; i <= parseInt(radiusNeighbors); i++) {
        let prev_id = "token_" + (numerical_id - i);
        let succ_id = "token_" + (numerical_id + i);
        let prev_token = document.getElementById(prev_id);
        let succ_token = document.getElementById(succ_id);
        if (prev_token != null && prev_token.getAttribute("line") == currentLine) {
            selectedTokens.push(prev_token);
            distances.push(- parseInt(i))
        }
        if (succ_token != null && succ_token.getAttribute("line") == currentLine) {
            //console.log("show also " + succ_token.innerHTML);
            selectedTokens.push(succ_token);
            distances.push(parseInt(i))
        }
    };
    return { neighbors: selectedTokens, distances: distances}
};

function triggerDeblurr(event) {
    deblurr(event.target, 0);
    let { neighbors, distances } = getNeighbors(event.target.id, NEIGHBORS_SIZE);
    for(let i = 0; i < neighbors.length; i++) {
        neigh = neighbors[i];
        dist = distances[i];
        deblurr(neigh, dist);
    }
};

function triggerBlurr(event) {
    blurr(event.target, 0);
    let { neighbors, distances } = getNeighbors(event.target.id, NEIGHBORS_SIZE);
    for(let i = 0; i < neighbors.length; i++) {
        neigh = neighbors[i];
        dist = distances[i];
        blurr(neigh, dist);
    }
};


// STATUS CHECKER


function checkPlatform() {
    try {
        platformName = platform.name;
        platformVersion = platform.version;
        platformOs = platform.os;
        return "";
    } catch (error) {
        return "ERROR: Problem reading browser info (use another one): " + error;
    }
}

function checkMouseTrace() {
    try {
        n_events_mouse = allMouseEvents.length;
        if (n_events_mouse == 0) {
            return "Inspect the code with your mouse."
        }
        return "";
    } catch (error) {
        return error;
    }
}

function checkTokenInteraction() {
    try {
        var n_events_tokens = allClickEvents.length;
        if (n_events_tokens == 0) {
            return "Move the mouse over the tokens to see their text."
        }
        return "";
    } catch (error) {
        return error;
    }
}

function checkCheating() {
    var all_tokens_in_the_page = document.getElementsByClassName("token");
    var n_black_visible = 0;
    try {
        for (let i = 0; i < all_tokens_in_the_page.length; i++) {
            var c_token = all_tokens_in_the_page[i];
            // console.log(">", c_token.style.color, "<")
            if (c_token.style.color == "black") {
                n_black_visible++;
            }
            if (c_token.style.color != "red" && c_token.style.color != "black") {
                c_token.style.color = 'transparent';
            }
        }
        //console.log('n_black_visible: ', n_black_visible);
        if (n_black_visible > (NEIGHBORS_SIZE * 2 + 1)) {
            const d = new Date();
            const currentTimeEvent = d.getTime() - openedPageTime;
            traceVisibleTokens.push(new VisibleTokensLog(n_black_visible, currentTimeEvent));
            return "Too many tokens are visible. Impossible to submit."
        };
        return "";
    } catch (error) {
        return error;
    }
}

function checkStatus(){
    var err_msg = "";
    var status_msg = "STATUS: Ready to answer.";
    var experimentStatus = document.getElementById('experiment_status');
    err_msg = err_msg + " " + checkPlatform();
    // console.log('checkPlatform: ', err_msg)
    err_msg = err_msg + " " + checkMouseTrace();
    // console.log('checkMouseTrace: ', err_msg)
    err_msg = err_msg + " " + checkTokenInteraction();
    // console.log('checkTokenInteraction: ', err_msg)
    err_msg = err_msg + " " + checkCheating();
    // console.log('checkCheating: ', err_msg)
    if (err_msg.trim() != "") {
        status_msg = "TO DOs: " + err_msg;
        readyToSubmit = false;
    } else {
        readyToSubmit = true;
    }
    // console.log('Check Status: ', status_msg)
    experimentStatus.innerHTML = status_msg;
    setTimeout(checkStatus, CHECK_CHEATING_AND_STATUS_EVERY_MILLI);
}


// check flush

function checkFlush() {
    const n_new_interactions_before_flush = MIN_TOKEN_INTERACTION_BEFORE_FLUSH;
    if (allClickEvents.length >
        startChunkTokenInteractions + n_new_interactions_before_flush) {
        flushChunkToServer();
    } ;
    setTimeout(checkFlush, CHECK_TO_FLUSH_EVERY_MILLI);
}


// PREPARE OPTIONS SECTION AND INTERACTIONS

for (let i = 0; i < options.length; i++) {
    let opt = options[i];
    let optionName = (opt.textContent.trim());
    optionsNames.push(optionName);

    opt.addEventListener("click", function(clickedItem) {
        console.log("send to server");
        flushChunkToServer(optionName);
        //postSessionToServer(optionName);
    });

}

for (let i = 0; i < liOptions.length; i++) {
    let opt = liOptions[i];
    opt.addEventListener("mouseover", (event) => {
        var d = new Date();
        var currentTimeEvent = d.getTime() - openedPageTime;
        let interaction = new OptionHoverLog(
            event.target.parentNode.textContent.trim(),
            currentTimeEvent,
            1
        );
        //console.log(interaction);
        allOptionsInteractionEvents.push(interaction);
    });
    opt.addEventListener("mouseout", (event) => {
        var d = new Date();
        var currentTimeEvent = d.getTime() - openedPageTime;
        let interaction = new OptionHoverLog(
            event.target.parentNode.textContent.trim(),
            currentTimeEvent,
            0
        );
        //console.log(interaction);
        allOptionsInteractionEvents.push(interaction);
    });
}

// ADJUST NAVBAR HEIGHT
const nav_height = parseInt(175 + optionsNames.length * height_single_option);
document.getElementById('navbar').style.height = nav_height + 'px';

// INITIALIZE THE TOKEN DEBLURR LISTENER
let all_tokens_to_deblurr = document.getElementsByClassName("token");
const NEIGHBORS_SIZE = 3;
for (let i = 0; i < all_tokens_to_deblurr.length; i++) {
    // DEBUG console.log(all_tokens_to_deblurr[i]);
    all_tokens_to_deblurr[i].addEventListener("mouseover", triggerDeblurr);
    all_tokens_to_deblurr[i].addEventListener("mouseout", triggerBlurr);
}

// INITIALIZE MOUSE LISTENER

$(document).mousemove(function(event) {
    captureMousePosition(event);
})

$(window).scroll(function(event) {
    if(lastScrolledLeft != $(document).scrollLeft()){
        xMousePos -= lastScrolledLeft;
        lastScrolledLeft = $(document).scrollLeft();
        xMousePos += lastScrolledLeft;
    }
    if(lastScrolledTop != $(document).scrollTop()){
        yMousePos -= lastScrolledTop;
        lastScrolledTop = $(document).scrollTop();
        yMousePos += lastScrolledTop;
    }
    logMouse(xMousePos, yMousePos);
});

// prevent reload

window.onbeforeunload = function() {
    if (!voluntaryChangeOfPage) {
        return 'The experiment will be invalidated with a refresh!';
    }
};


// continuous check

checkStatus();

checkFlush();


checkEntireJavascriptExecution = 1;
