// Main variables
let recognition;
let isListening = false;
let confidences = { sad: 0, fear: 0, anger: 0, anxiety: 0, neutral: 1, excitement: 0, joy: 0 };
let lastTranscript = '';
let eyeOpen = true;
let blinkTimer;

// Error handling
let recognitionTimeout = null;
let consecutiveErrors = 0;
let maxConsecutiveErrors = 3;
let backoffTime = 1000;
let textInputActive = false;

// Visualization variables
let eyeImages = {};
let emotionSequences = {};
let emotionStyles = {};
let currentEmotion = "neutral";
let lastEmotion = "neutral";
let transitionProgress = 1.0;
let transitionSpeed = 0.05;
let timelapseFrame = 0;
let lastFrameUpdate = 0;
let frameUpdateInterval = 150;
let useDrawnEyes = true;
let animateDrawings = true;
let interactionPoints = [];

function preload() {
  // style groups
  emotionStyles = {
    calm: ["neutral", "sad"],
    energetic: ["excitement", "anxiety", "fear"],
    positive: ["joy"],
    negative: ["anger"]
  };
  
  // sad sequence (skip missing 6)
  emotionSequences.sad = [];
  [1,2,3,4,5,7].forEach(i => {
    emotionSequences.sad.push(loadImage(`/static/sad/sad_${i}.png`));
  });
  
  // neutral
  emotionSequences.neutral = [];
  for (let i = 1; i <= 6; i++) {
    emotionSequences.neutral.push(loadImage(`/static/neutral/neutral_${i}.png`));
  }
  
  // energetic (indices match files)
  const energeticIndices = [1,2,3,4,5,6,7,8,10]; // exclude index 9 due to missing file
  emotionSequences.excitement = [];
  emotionSequences.anxiety    = [];
  emotionSequences.fear       = [];
  energeticIndices.forEach(n => {
    const img = loadImage(`/static/happy/energetic_${n}.png`);
    emotionSequences.excitement.push(img);
    emotionSequences.anxiety   .push(img);
    emotionSequences.fear      .push(img);
  });
  
  // joy static
  emotionSequences.joy = [ loadImage(`/static/happy/happy_static.png`) ];
  
  // anger reuse energetic
  emotionSequences.anger = emotionSequences.excitement.slice();
  
  // static fallbacks
  eyeImages.sad        = loadImage(`/static/sad/sad_static.png`);
  eyeImages.neutral    = loadImage(`/static/neutral/neutral_static.png`);
  eyeImages.excitement = loadImage(`/static/happy/energetic_static.png`);
  eyeImages.joy        = loadImage(`/static/happy/happy_static.png`);
  eyeImages.anxiety    = loadImage(`/static/sad/anxiety_static.png`);
  eyeImages.fear       = loadImage(`/static/sad/fear_static.png`);
  eyeImages.anger      = eyeImages.excitement;
  
  // Add styles for enhanced emotion display
  const style = document.createElement('style');
  style.textContent = `
    @keyframes pulse {
      0% { transform: scale(0.8); opacity: 0.7; }
      50% { transform: scale(1.2); opacity: 1; }
      100% { transform: scale(0.8); opacity: 0.7; }
    }
    
    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }
    
    #emotion-panel {
      animation: fadeIn 0.5s ease-out;
    }
  `;
  document.head.appendChild(style);
}

function setup() {
  // Add error handler to catch p5.AudioIn issues
  window.addEventListener('error', function(e) {
    console.log('Error caught:', e);
    console.trace();
  });
  
  let canvas = createCanvas(windowWidth, windowHeight);
  canvas.parent('visual-container');
  colorMode(HSB,360,100,100,1);
  noStroke();

  setupSpeechRecognition();
  // begin speech recognition immediately
  startListening();
  startBlinking();
  
  // Create the emotion display panel
  createEmotionPanel();
  
  console.log('setup done');
}

function createEmotionPanel() {
  // Remove existing panel if it exists
  const existingPanel = document.getElementById('emotion-panel');
  if (existingPanel) existingPanel.remove();
  
  const panel = document.createElement('div');
  panel.id = 'emotion-panel';
  Object.assign(panel.style, {
    position: 'fixed',
    bottom: '20px',
    left: '50%',
    transform: 'translateX(-50%)',
    background: 'rgba(0, 0, 0, 0.8)',
    color: '#fff',
    padding: '10px 20px',
    borderRadius: '10px',
    boxShadow: '0 0 15px rgba(255, 255, 255, 0.3)',
    zIndex: 1000,
    textAlign: 'center',
    fontFamily: 'Arial, sans-serif',
    display: 'none',
    width: 'auto',
    maxWidth: '250px'
  });
  
  const emotionLabel = document.createElement('div');
  emotionLabel.id = 'emotion-label';
  Object.assign(emotionLabel.style, {
    fontSize: '24px',
    fontWeight: 'bold',
    marginBottom: '6px',
    color: '#ffffff',
    textShadow: '0 0 10px rgba(255, 255, 255, 0.5)'
  });
  
  const intensityBar = document.createElement('div');
  Object.assign(intensityBar.style, {
    height: '12px',
    width: '100%',
    background: '#333',
    borderRadius: '6px',
    overflow: 'hidden',
    boxShadow: 'inset 0 0 5px rgba(0, 0, 0, 0.5)',
    marginTop: '8px'
  });
  
  const intensityFill = document.createElement('div');
  intensityFill.id = 'intensity-fill';
  Object.assign(intensityFill.style, {
    height: '100%',
    width: '50%',
    background: 'linear-gradient(90deg, #888, #fff)',
    transition: 'width 0.5s, background 0.5s'
  });
  
  intensityBar.appendChild(intensityFill);
  panel.appendChild(emotionLabel);
  panel.appendChild(intensityBar);
  document.body.appendChild(panel);
}

function setupSpeechRecognition() {
  if (!('SpeechRecognition' in window) && !('webkitSpeechRecognition' in window)) {
    document.getElementById('status').innerText = 'Speech not supported. Please type.';
    showTextInputFallback('Type your message below.');
    return;
  }
  recognition = new (window.SpeechRecognition || window.webkitSpeechRecognition)();
  recognition.continuous = true;
  recognition.interimResults = false;
  recognition.lang = 'en-US';

  recognition.onresult = e => {
    const transcript = e.results[e.results.length-1][0].transcript.trim();
    console.log('Transcript:',transcript);
    if (transcript && transcript !== lastTranscript) {
      lastTranscript = transcript;
      processText(transcript);
      consecutiveErrors = 0;
      backoffTime = 1000;
    }
  };
  recognition.onerror = e => {
    console.error('Speech error',e.error);
    consecutiveErrors++;
    clearTimeout(recognitionTimeout);
    if (consecutiveErrors>=maxConsecutiveErrors) showTextInputFallback('Speech failing — type instead.');
    recognitionTimeout = setTimeout(startListening, backoffTime);
  };
  recognition.onend = ()=>{
    console.log('Speech ended, restarting...');
    if (!recognitionTimeout) recognitionTimeout = setTimeout(startListening, backoffTime);
  };
  window.addEventListener('mousedown',startListening);
}

function startListening(){
  try { 
    recognition.start(); 
    isListening=true;
    document.getElementById('status').innerText = 'The eye is listening. Just start speaking...';
  } catch { 
    recognitionTimeout = setTimeout(startListening,1000); 
  }
}

function showTextInputFallback(msg){
  if (textInputActive) return;
  textInputActive=true;
  const c = document.createElement('div'); c.id='text-input-container';
  Object.assign(c.style,{position:'fixed',bottom:'20px',left:'50%',transform:'translateX(-50%)',background:'rgba(0,0,0,0.7)',padding:'10px',borderRadius:'8px',color:'#fff'});
  if (msg){const m=document.createElement('div');m.textContent=msg;c.appendChild(m);}  
  const inp = document.createElement('input'); inp.placeholder='Type here...'; inp.style.padding='8px'; inp.style.margin='5px';
  const btn = document.createElement('button'); btn.textContent='Go'; btn.onclick=()=>processText(inp.value);
  c.append(inp,btn);
  document.body.appendChild(c);
}

function getEmotionColor(emotion) {
  const colors = {
    sad: '#0000ff',      // Blue
    fear: '#800080',     // Purple
    anger: '#ff0000',    // Red
    anxiety: '#ffa500',  // Orange
    neutral: '#ffffff',  // White
    excitement: '#ffff00', // Yellow
    joy: '#00ff00'       // Green
  };
  return colors[emotion] || '#ffffff';
}

function processText(text){
  fetch('/classify',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})})
    .then(r=>r.json()).then(data=>{
      confidences = data.confidences;
      lastEmotion = currentEmotion;
      currentEmotion = Object.keys(confidences).reduce((a,b)=>confidences[a]>confidences[b]?a:b);
      transitionProgress=0;
      timelapseFrame=0;
      lastFrameUpdate=millis();
      interactionPoints=[];
      for(let i=0;i<15;i++) addInteractionPoints();
      
      // Update the status display at the top
      const statusEl = document.getElementById('status');
      const statusText = `Detected: ${data.emotion} (intensity: ${data.intensity})`;
      statusEl.innerText = statusText;
      Object.assign(statusEl.style, {
        background: 'rgba(0, 0, 0, 0.8)',
        color: getEmotionColor(data.emotion),
        padding: '6px 12px',
        borderRadius: '6px',
        fontSize: '14px',
        fontWeight: 'bold',
        position: 'fixed',
        top: '10px', 
        left: '50%',
        transform: 'translateX(-50%)',
        boxShadow: '0 0 10px rgba(255, 255, 255, 0.2)',
        zIndex: '1000',
        textShadow: '0 0 5px rgba(0, 0, 0, 0.8)',
        transition: 'background 0.3s, color 0.3s',
        display: 'inline-block',
        width: 'auto',
        maxWidth: '80%'
      });
      
      // Update the emotion panel
      const panel = document.getElementById('emotion-panel');
      panel.style.display = 'block';
      panel.style.borderColor = getEmotionColor(data.emotion);
      
      document.getElementById('emotion-label').textContent = data.emotion.toUpperCase();
      document.getElementById('emotion-label').style.color = getEmotionColor(data.emotion);
      
      const intensityValue = Math.abs(data.intensity);
      document.getElementById('intensity-fill').style.width = `${intensityValue}%`;
      document.getElementById('intensity-fill').style.background = `linear-gradient(90deg, ${getEmotionColor(data.emotion)}88, ${getEmotionColor(data.emotion)})`;
      
      // Create a visual flash for emotion change - reduced opacity
      const flashEl = document.createElement('div');
      Object.assign(flashEl.style, {
        position: 'fixed',
        top: '0',
        left: '0',
        width: '100%',
        height: '100%',
        backgroundColor: getEmotionColor(data.emotion),
        opacity: '0.2', // Reduced from 0.3
        pointerEvents: 'none',
        transition: 'opacity 0.5s',
        zIndex: '999'
      });
      document.body.appendChild(flashEl);
      setTimeout(() => {
        flashEl.style.opacity = '0';
        setTimeout(() => flashEl.remove(), 500);
      }, 100);
      
      eyeOpen=1.5; 
      setTimeout(()=>eyeOpen=1.0,300);
    }).catch(e=>{
      console.error(e);
      document.getElementById('status').innerText='Error analyzing.';
    });
}

function startBlinking(){setTimeout(doBlink,random(2000,7000));}
function doBlink(){eyeOpen=0.1;setTimeout(()=>eyeOpen=1.0,200);setTimeout(doBlink,random(2000,7000));}

function draw(){
  background(0,0,15);
  if(transitionProgress<1)transitionProgress=min(1,transitionProgress+transitionSpeed);
  if(animateDrawings && millis()-lastFrameUpdate>frameUpdateInterval){
    lastFrameUpdate=millis();
    const seq = emotionSequences[currentEmotion]||[];
    timelapseFrame=(timelapseFrame+1)%(seq.length||1);
  }
  if(useDrawnEyes) drawCustomEye(); else drawEmotionEye();
  drawInteractions();
}

function windowResized(){resizeCanvas(windowWidth,windowHeight);}

// Helper functions below
function drawCustomEye(){
  push(); translate(width/2, height/2);
  noFill(); stroke(0,0,80,0.3); strokeWeight(1);
  // horizontal oval
  ellipse(0,0,600,300);
  // iris image tinted by emotion
  const seq = emotionSequences[currentEmotion]||[];
  if(seq.length){
    tint(map(confidences[currentEmotion],0,1,0,360),80,100,transitionProgress);
    const img = seq[timelapseFrame];
    image(img,-img.width/2,-img.height/2);
    noTint();
  } else {
    drawEmotionIris();
  }
  noStroke(); fill(0);
  ellipse(0,0,50*eyeOpen,50*eyeOpen);
  pop();
}

function drawEmotionEye(){
  push(); translate(width/2, height/2);
  noFill(); stroke(0,0,80); strokeWeight(2);
  ellipse(0,0,600,300);
  drawEmotionIris();
  pop();
}

function drawEmotionIris(){
  const hueVal = map(confidences[currentEmotion]||0,0,1,0,360);
  fill(hueVal,80,100);
  ellipse(width/2, height/2, 200,200);
}

function drawInteractions(){
  noStroke(); fill(200,80,100);
  interactionPoints.forEach(p=>{
    ellipse(p.x, p.y, p.size);
    p.size *= 0.95;
    p.size<1 && (p.dead=true);
  });
  interactionPoints = interactionPoints.filter(p=>!p.dead);
}

function addInteractionPoints(){
  interactionPoints.push({
    x: random(width/2-200,width/2+200),
    y: random(height/2-100,height/2+100),
    size: random(5,20),
    dead: false
  });
}