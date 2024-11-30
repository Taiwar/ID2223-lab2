var SpeechRecognition = SpeechRecognition || webkitSpeechRecognition
var SpeechGrammarList = SpeechGrammarList || window.webkitSpeechGrammarList
var SpeechRecognitionEvent = SpeechRecognitionEvent || webkitSpeechRecognitionEvent

window.onload = (_) => {
  // Create the speech recognition classes.
  let recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.maxAlternatives = 1;

  // Control and diagnostics.
  const control = document.getElementById("control-button");
  const textField = document.getElementById("query-input");
  setSpeech(control, true);

  let talking = false;
  control.onclick = () => {
    if (!talking) {
      setAsk(false);
      recognition.start();
    } else {
      setAsk(true);
      recognition.stop();
    }
    setSpeech(talking);
    talking = !talking;
  }
  
  recognition.onresult = (event) => {
    setAsk(false);
    // Taken from MDN.
    let command = event.results[0][0].transcript;
    textField.value = command;
    query();
  }
  
  recognition.onerror = (event) => {
    console.error(event);
  }
};

async function query() {
  const textField = document.getElementById("query-input");
  const command = textField.value;
  if (!command) {
    return;
  }
  setAsk(false);
  const output = document.getElementById("output");
  const spinner = document.getElementById("spinner");
  const message = document.getElementById("response");
  spinner.classList.remove("is-hidden");
  message.classList.remove("is-hidden");
  output.innerText = command;
  spinner.classList.add("is-hidden");
  setAsk(true);
}

function setAsk(state) {
  const ask = document.getElementById("ask-button");
  ask.disabled = !state;
}

function setSpeech(state) {
  const control = document.getElementById("control-button");
  if (state) {
    control.classList = "button is-success";
    control.innerText = "Start speaking";
  } else {
    control.classList = "button is-danger";
    control.innerText = "Stop speaking"
  }
}
