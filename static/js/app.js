//----------------------- JAVA SCRIPT FOR RECORDING THE AUDIO ----------------------------------------
URL = window.URL || window.webkitURL;

var gumStream;
var rec;
var input;
var AudioContext = window.AudioContext || window.webkitAudioContext;
var audioContext
var recordButton = document.getElementById("recordButton");
var stopButton = document.getElementById("stopButton");


recordButton.addEventListener("click", startRecording);
stopButton.addEventListener("click", stopRecording);

function remove() {
    document.getElementById("recordingsList").innerHTML = "";
    document.getElementById("downloadsample").innerHTML = "";
}

function startRecording() {
    console.log("recordButton clicked");
    remove()
    recordButton.disabled = true
    document.getElementById("stopButton").style.backgroundColor = "#ff0000";
    document.getElementById("recordButton").style.backgroundColor = "#595959";
    stopButton.setAttribute('disabled', '');
    var constraints = {
        audio: true,
        video: false
    }
    recordButton.disabled = true;
    stopButton.disabled = false;
    navigator.mediaDevices.getUserMedia(constraints).then(function(stream) {
        console.log("getUserMedia() success, stream created, initializing Recorder.js ...");
        audioContext = new AudioContext();
        gumStream = stream;
        input = audioContext.createMediaStreamSource(stream);
        rec = new Recorder(input, {
            numChannels: 1
        })
        rec.record()
        console.log("Recording started");
    }).catch(function(err) {
        recordButton.disabled = false;
        stopButton.disabled = true;
    });
}

function stopRecording() {
    console.log("stopButton clicked");
    document.getElementById("recordButton").style.visibility = "visible";
    document.getElementById("stopButton").style.visibility = "visible";
    document.getElementById("recordButton").style.backgroundColor = "#00cc66";
    document.getElementById("stopButton").style.backgroundColor = "#595959";
    stopButton.disabled = true;
    recordButton.disabled = false;
    rec.stop();
    gumStream.getAudioTracks()[0].stop();
    rec.exportWAV(createDownloadLink);
}

function createDownloadLink(blob) {
    var url = URL.createObjectURL(blob);
    var au = document.createElement('audio');
    var li = document.createElement('p');
    var link = document.createElement('a');
    var filename = new Date().toISOString();
    var k = document.getElementById("downloadsample")
    li.setAttribute('id', 'list1;')
    au.setAttribute('Style', "width: 350px;")
    au.controls = true;
    au.src = url;
    link.href = url;
    link.download = filename + ">Sample.wav";
    link.innerHTML = "Download Sample";
    li.appendChild(au);
    k.appendChild(link);
    e = document.getElementById('stopButton')
    e.onclick = function(event) {
        var xhr = new XMLHttpRequest();
        xhr.onload = function(e) {
            if (this.readyState === 4) {
                console.log("Server returned: ", e.target.responseText);
            }
        };
        var fd = new FormData();
        fd.append("audio_data", blob, filename);
        xhr.open("POST", "/", true);
        xhr.send(fd);
    }
    recordingsList.appendChild(li);
}
