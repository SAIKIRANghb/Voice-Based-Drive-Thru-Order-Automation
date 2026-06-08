// Connect to Socket.IO Server
const socket = io();

// DOM Elements
const elStatusDot = document.querySelector('.status-dot');
const elStatusText = document.getElementById('status-text');
const elBtnDriveCar = document.getElementById('btn-drive-car');
const elBtnResetSession = document.getElementById('btn-reset-session');
const elBtnMicToggle = document.getElementById('btn-mic-toggle');
const elBtnSimulateVoice = document.getElementById('btn-simulate-voice');

const elRangeMu = document.getElementById('range-mu');
const elValMu = document.getElementById('val-mu');
const elSelectTaps = document.getElementById('select-taps');
const elRangeVad = document.getElementById('range-vad');
const elValVad = document.getElementById('val-vad');
const elToggleEcho = document.getElementById('toggle-echo');
const elRangeScale = document.getElementById('range-scale');
const elValScale = document.getElementById('val-scale');

const elSelectAsr = document.getElementById('select-asr');
const elSelectLlm = document.getElementById('select-llm');
const elSelectTts = document.getElementById('select-tts');

const elCartList = document.getElementById('cart-list');
const elSubtotalVal = document.getElementById('subtotal-val');
const elPromoRow = document.getElementById('promo-row');
const elDiscountVal = document.getElementById('discount-val');
const elTotalVal = document.getElementById('total-val');
const elLogsContainer = document.getElementById('logs-container');
const elGuardrailAlert = document.getElementById('guardrail-alert');
const elGuardrailMsg = document.getElementById('guardrail-msg');

const elTextInput = document.getElementById('text-input');
const elBtnSendText = document.getElementById('btn-send-text');
const elBtnSpeakerAnalytics = document.getElementById('btn-speaker-analytics');
const MENU_PRICES = {
    burger: 5.99,
    fries: 2.49,
    soda: 1.99,
    nuggets: 4.49,
    shake: 2.99,
    'onion rings': 2.99,
    'apple pie': 1.79
};

// Audio Context Variables
let audioContext = null;
let micStream = null;
let micNode = null;
let scriptProcessor = null;
let isStreaming = false;
let audioStreamId = 0;
let micSource = null;
let muteGain = null;

// Audio Playback Variables
let playContext = null;

// Oscilloscope Canvas Variables
const canvasRaw = document.getElementById('canvas-raw');
const canvasRef = document.getElementById('canvas-ref');
const canvasClean = document.getElementById('canvas-clean');

const ctxRaw = canvasRaw.getContext('2d');
const ctxRef = canvasRef.getContext('2d');
const ctxClean = canvasClean.getContext('2d');

// Wave History Buffers
const historyLen = 200;
const historyRaw = new Array(historyLen).fill(0);
const historyRef = new Array(historyLen).fill(0);
const historyClean = new Array(historyLen).fill(0);

// Initialize canvases
function initCanvases() {
    [canvasRaw, canvasRef, canvasClean].forEach(canvas => {
        canvas.width = canvas.parentElement.clientWidth;
        canvas.height = 60;
    });
}
window.addEventListener('resize', initCanvases);
initCanvases();

// Socket IO Event Listeners
socket.on('connect', () => {
    elStatusDot.className = 'status-dot connected';
    elStatusText.innerText = 'Connected';
    elBtnDriveCar.disabled = false;
    logSystem("Connected to backend media worker.");
});

socket.on('disconnect', () => {
    elStatusDot.className = 'status-dot disconnected';
    elStatusText.innerText = 'Disconnected';
    elBtnDriveCar.disabled = true;
    elBtnMicToggle.disabled = true;
    elBtnSimulateVoice.disabled = true;
    elTextInput.disabled = true;
    elBtnSendText.disabled = true;
    elBtnSpeakerAnalytics.disabled = true;
    logSystem("Connection lost. Retrying...");
});

socket.on('agent_response', (data) => {
    // 1. Log response
    logMessage('agent', data.text);
    
    // 2. Playback generated TTS audio
    const audioBytes = normalizeAudioPayload(data.audio);
    let playbackPromise = Promise.resolve();
    if (audioBytes && audioBytes.byteLength > 0) {
        playbackPromise = playPCMAudio(audioBytes, data.audio_sample_rate || 16000);
    }
    
    // 3. Highlight current LangGraph FSM Node
    updateFsmNode(data.node);
    
    // 4. Update Order Cart
    updateCart(data.cart, data.total);
    
    // 5. Check Hallucination Warning
    if (data.hallucination_warning) {
        elGuardrailAlert.style.display = 'block';
        elGuardrailMsg.innerText = `Off-menu item detected. Intercepted and corrected to: "${data.text}"`;
    } else {
        elGuardrailAlert.style.display = 'none';
    }

    if (data.node === 'closing' && isStreaming) {
        playbackPromise.finally(() => {
            stopMicStream({ finalize: false, reason: 'order_closed' });
            elBtnMicToggle.disabled = true;
            logSystem('Order closed. Microphone pipeline stopped.');
        });
    }
});

socket.on('agent_error', (data) => {
    logSystem(`Agent error: ${data.message || 'Unknown error'}`);
});

socket.on('asr_status', (data) => {
    logSystem(data.message || 'Audio was received, but no transcript was detected.');
});

socket.on('customer_speech', (data) => {
    logMessage('user', data.text, data.speaker);
});

socket.on('speaker_analytics_result', (data) => {
    const results = data.results || [];
    if (results.length === 0) {
        logSystem("Speaker analytics has no captured utterances yet.");
        return;
    }

    const summary = results.map(result => {
        const score = Number(result.speaker_score || 0).toFixed(2);
        return `#${result.utterance_index + 1}: ${result.speaker} (${score}, ${result.duration_ms} ms)`;
    }).join('; ');
    logSystem(`Speaker analytics: ${summary}`);
});

socket.on('session_reset', () => {
    logSystem('Session reset on backend. UI state cleared.');
    updateFsmNode(null);
    updateCart({}, 0.0);
    elGuardrailAlert.style.display = 'none';
    elBtnMicToggle.disabled = true;
    elBtnSimulateVoice.disabled = true;
    elTextInput.disabled = true;
    elBtnSendText.disabled = true;
    elBtnSpeakerAnalytics.disabled = true;
});

socket.on('dsp_levels', (data) => {
    // Log energy levels in dB approximation for display
    document.getElementById('raw-db').innerText = `${Math.round(20 * Math.log10((data.raw || 1) / 32768))} dB`;
    document.getElementById('ref-db').innerText = `${Math.round(20 * Math.log10((data.ref || 1) / 32768))} dB`;
    document.getElementById('clean-db').innerText = `${Math.round(20 * Math.log10((data.clean || 1) / 32768))} dB`;
    
    // Push new heights to waveform history
    historyRaw.push(data.raw / 32768.0);
    historyRef.push(data.ref / 32768.0);
    historyClean.push(data.clean / 32768.0);
    
    // Cap buffer lengths
    if (historyRaw.length > historyLen) historyRaw.shift();
    if (historyRef.length > historyLen) historyRef.shift();
    if (historyClean.length > historyLen) historyClean.shift();
});

// Animation Loop for Canvases
function drawOscilloscopes() {
    requestAnimationFrame(drawOscilloscopes);
    
    drawWaveform(ctxRaw, historyRaw, '#ef4444');
    drawWaveform(ctxRef, historyRef, '#f59e0b');
    drawWaveform(ctxClean, historyClean, '#10b981');
}
drawOscilloscopes();

function drawWaveform(ctx, history, color) {
    const w = ctx.canvas.width;
    const h = ctx.canvas.height;
    ctx.clearRect(0, 0, w, h);
    
    // Draw background grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x < w; x += 30) {
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
    }
    ctx.moveTo(0, h/2);
    ctx.lineTo(w, h/2);
    ctx.stroke();
    
    // Draw wave
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    
    const sliceWidth = w / historyLen;
    let x = 0;
    
    for (let i = 0; i < historyLen; i++) {
        const amp = history[i] * (h / 2) * 1.5; // Scale for visual visibility
        const y = (h / 2) - amp;
        
        if (i === 0) {
            ctx.moveTo(x, y);
        } else {
            ctx.lineTo(x, y);
        }
        x += sliceWidth;
    }
    
    ctx.stroke();
}

// Start Session Action
elBtnDriveCar.addEventListener('click', () => {
    elBtnDriveCar.disabled = true;
    logSystem("Starting voice session...");

    socket.emit('start_session', {});

    elBtnMicToggle.disabled = false;
    elBtnSimulateVoice.disabled = false;
    elTextInput.disabled = false;
    elBtnSendText.disabled = false;
    elBtnSpeakerAnalytics.disabled = false;
    elBtnResetSession.disabled = false;
});

// Reset Session Action
elBtnResetSession.addEventListener('click', () => {
    elBtnDriveCar.disabled = false;
    elBtnMicToggle.disabled = true;
    elBtnSimulateVoice.disabled = true;
    elTextInput.disabled = true;
    elBtnSendText.disabled = true;
    elBtnSpeakerAnalytics.disabled = true;
    
    if (isStreaming) {
        stopMicStream({ finalize: false, reason: 'session_reset' });
    }
    
    socket.emit('reset_session');
    
    // Clear FSM states
    updateFsmNode(null);
    updateCart({}, 0.0);
    elGuardrailAlert.style.display = 'none';
    
    logSystem("Session reset. Lane is ready.");
});

// Microphone Capture toggle
elBtnMicToggle.addEventListener('click', () => {
    if (isStreaming) {
        logSystem('Microphone is already listening. Pause after each question to send it.');
    } else {
        startMicStream();
    }
});

async function startMicStream() {
    try {
        if (!window.isSecureContext) {
            logSystem("Microphone requires a secure browser context. Open this app as http://localhost:5000 or http://127.0.0.1:5000, not 0.0.0.0 or a LAN IP unless HTTPS is enabled.");
            return;
        }
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            logSystem("Microphone capture is not available in this browser/context.");
            return;
        }

        const streamId = audioStreamId + 1;
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: {
                channelCount: 1,
                echoCancellation: true,
                noiseSuppression: true,
                autoGainControl: true
            }
        });
        
        audioContext = new (window.AudioContext || window.webkitAudioContext)();
        micSource = audioContext.createMediaStreamSource(micStream);
        
        const targetSampleRate = 16000;
        // Silero VAD requires at least 512 samples at 16 kHz.
        const frameSize = 512;
        const inputSampleRate = audioContext.sampleRate;
        const ratio = inputSampleRate / targetSampleRate;
        let inputBuffer = [];
        let readPosition = 0;
        let sampleBuffer = [];
        audioStreamId = streamId;
        
        scriptProcessor = audioContext.createScriptProcessor(1024, 1, 1);
        scriptProcessor.onaudioprocess = (e) => {
            if (!isStreaming || streamId !== audioStreamId) {
                return;
            }
            const inputData = e.inputBuffer.getChannelData(0);

            for (let i = 0; i < inputData.length; i++) {
                inputBuffer.push(inputData[i]);
            }

            while (readPosition + 1 < inputBuffer.length) {
                const baseIndex = Math.floor(readPosition);
                const fraction = readPosition - baseIndex;
                const current = inputBuffer[baseIndex] || 0;
                const next = inputBuffer[baseIndex + 1] || current;
                const resampled = current + (next - current) * fraction;
                const s = Math.max(-1, Math.min(1, resampled));
                const intSample = s < 0 ? s * 0x8000 : s * 0x7FFF;
                sampleBuffer.push(Math.round(intSample));

                if (sampleBuffer.length === frameSize) {
                    const int16Array = new Int16Array(sampleBuffer);
                    if (isStreaming && streamId === audioStreamId) {
                        socket.emit('audio_frame', {
                            stream_id: streamId,
                            audio: int16Array.buffer
                        });
                    }
                    sampleBuffer = [];
                }

                readPosition += ratio;
            }

            const consumed = Math.floor(readPosition);
            if (consumed > 0) {
                inputBuffer = inputBuffer.slice(consumed);
                readPosition -= consumed;
            }
        };
        
        muteGain = audioContext.createGain();
        muteGain.gain.value = 0;
        micSource.connect(scriptProcessor);
        scriptProcessor.connect(muteGain);
        muteGain.connect(audioContext.destination);
        
        isStreaming = true;
        socket.emit('audio_stream_started', { stream_id: streamId });
        elBtnMicToggle.classList.add('streaming');
        elBtnMicToggle.innerHTML = '<i class="fa-solid fa-microphone-lines"></i> Listening';
        logSystem(`Microphone streaming active (${Math.round(inputSampleRate)} Hz captured, PCM-16 @ 16 kHz sent).`);
        logSystem('Keep speaking naturally; each pause sends the current question.');
    } catch (e) {
        console.error("Microphone capture failed:", e);
        const permissionName = e.name || "MicrophoneError";
        let help = e.message || "Unknown microphone error";
        if (permissionName === "NotAllowedError" || permissionName === "SecurityError") {
            help = "Permission denied by the browser or OS. Allow microphone access for this site, then reload the page.";
        } else if (permissionName === "NotFoundError" || permissionName === "DevicesNotFoundError") {
            help = "No microphone device was found.";
        } else if (permissionName === "NotReadableError" || permissionName === "TrackStartError") {
            help = "The microphone is busy in another app or blocked by the OS.";
        }
        logSystem(`Microphone access failed (${permissionName}): ${help}`);
    }
}

function stopMicStream(options = {}) {
    const finalize = options.finalize !== false;
    const reason = options.reason || 'manual_stop';
    const streamId = audioStreamId;
    isStreaming = false;
    if (socket.connected) {
        socket.emit('audio_stream_stopped', {
            stream_id: streamId,
            finalize,
            reason
        });
    }
    elBtnMicToggle.classList.remove('streaming');
    elBtnMicToggle.innerHTML = '<i class="fa-solid fa-microphone"></i> Start Mic Stream';
    
    if (scriptProcessor) {
        scriptProcessor.disconnect();
        scriptProcessor = null;
    }
    if (muteGain) {
        muteGain.disconnect();
        muteGain = null;
    }
    if (micSource) {
        micSource.disconnect();
        micSource = null;
    }
    if (micStream) {
        micStream.getTracks().forEach(track => track.stop());
        micStream = null;
    }
    if (audioContext) {
        audioContext.close();
        audioContext = null;
    }
    logSystem("Microphone streaming stopped.");
}

function min(a, b) { return a < b ? a : b; }

// Direct Text Simulator send
elBtnSendText.addEventListener('click', () => {
    sendSimulatedText();
});

elBtnSpeakerAnalytics.addEventListener('click', () => {
    logSystem("Running offline speaker analytics...");
    socket.emit('run_speaker_analytics');
});

elTextInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter') {
        sendSimulatedText();
    }
});

function sendSimulatedText() {
    const text = elTextInput.value.trim();
    if (!text) return;
    
    logMessage('user', text);
    socket.emit('send_text', { text: text });
    elTextInput.value = '';
}

// Simulate Customer Speaking Button (Streams pre-programmed phrase)
elBtnSimulateVoice.addEventListener('click', () => {
    const phrase = "I want one burger and a cup of fries, and wait, do you have any apple pie?";
    logSystem(`Simulating voice order: "${phrase}"`);
    logMessage('user', phrase);
    
    // Instead of streaming microphone binary packets, send the text directly
    // to simulate the exact transcription result of this phrase.
    socket.emit('send_text', { text: phrase });
});

function normalizeAudioPayload(audio) {
    if (!audio) return null;

    if (typeof audio === 'string') {
        const binary = atob(audio);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    }

    if (audio instanceof ArrayBuffer) {
        return new Uint8Array(audio);
    }

    if (ArrayBuffer.isView(audio)) {
        return new Uint8Array(audio.buffer, audio.byteOffset, audio.byteLength);
    }

    if (Array.isArray(audio)) {
        return new Uint8Array(audio);
    }

    if (audio.type === 'Buffer' && Array.isArray(audio.data)) {
        return new Uint8Array(audio.data);
    }

    return null;
}

// Playback Agent Speech PCM
function playPCMAudio(audioBytes, sampleRate = 16000) {
    if (!playContext) {
        playContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (playContext.state === 'suspended') {
        playContext.resume();
    }
    
    // Convert int16 bytes array back to float32
    const int16View = new Int16Array(audioBytes.buffer, audioBytes.byteOffset, Math.floor(audioBytes.byteLength / 2));
    const float32Array = new Float32Array(int16View.length);
    for (let i = 0; i < int16View.length; i++) {
        float32Array[i] = int16View[i] / 32768.0;
    }
    
    // Create audio buffer (1 channel, PCM sample rate from backend)
    const audioBuffer = playContext.createBuffer(1, float32Array.length, sampleRate);
    audioBuffer.copyToChannel(float32Array, 0);
    
    // Source node
    const sourceNode = playContext.createBufferSource();
    sourceNode.buffer = audioBuffer;
    sourceNode.connect(playContext.destination);
    
    return new Promise((resolve) => {
        sourceNode.onended = () => {
            resolve();
        };

        sourceNode.start(0);
    });
}

// GUI Updates
function updateFsmNode(nodeName) {
    // Remove active class from all nodes
    document.querySelectorAll('.node').forEach(node => {
        node.classList.remove('active');
    });
    
    if (nodeName) {
        const elNode = document.getElementById(`node-${nodeName}`);
        if (elNode) {
            elNode.classList.add('active');
            logSystem(`State Machine transitioned to: [${nodeName.toUpperCase()}]`);
        }
    }
}

function updateCart(cart, total) {
    elCartList.innerHTML = '';
    
    const items = Object.entries(cart);
    if (items.length === 0) {
        elCartList.innerHTML = '<li class="empty-cart">No items added yet.</li>';
        elSubtotalVal.innerText = '$0.00';
        elTotalVal.innerText = '$0.00';
        elPromoRow.style.display = 'none';
        return;
    }
    
    let subtotal = 0;
    items.forEach(([item, qty]) => {
        const price = MENU_PRICES[item] || 0;
        const itemTotal = price * qty;
        subtotal += itemTotal;
        
        const li = document.createElement('li');
        li.className = 'cart-item';
        li.innerHTML = `
            <span class="cart-item-name">${qty}x ${item}</span>
            <span>$${itemTotal.toFixed(2)}</span>
        `;
        elCartList.appendChild(li);
    });
    
    elSubtotalVal.innerText = `$${subtotal.toFixed(2)}`;
    
    // Apply discount calculation display
    const discount = subtotal - total;
    if (discount > 0.01) {
        elPromoRow.style.display = 'flex';
        elDiscountVal.innerText = `-$${discount.toFixed(2)}`;
    } else {
        elPromoRow.style.display = 'none';
    }
    
    elTotalVal.innerText = `$${total.toFixed(2)}`;
}

// Logger Helpers
function logSystem(text) {
    const timeStr = new Date().toLocaleTimeString();
    const logDiv = document.createElement('div');
    logDiv.className = 'log-message system';
    logDiv.innerHTML = `<span class="time">[${timeStr}]</span> <i class="fa-solid fa-gears"></i> ${text}`;
    elLogsContainer.appendChild(logDiv);
    elLogsContainer.scrollTop = elLogsContainer.scrollHeight;
}

function logMessage(sender, text, label = null) {
    const timeStr = new Date().toLocaleTimeString();
    const logDiv = document.createElement('div');
    logDiv.className = `log-message ${sender}`;
    const senderLabel = label || (sender === 'user' ? 'Customer' : 'Agent');
    const icon = sender === 'user'
        ? `<i class="fa-solid fa-user"></i> ${senderLabel}:`
        : `<i class="fa-solid fa-robot"></i> ${senderLabel}:`;
    logDiv.innerHTML = `<span class="time">[${timeStr}]</span> ${icon} ${text}`;
    elLogsContainer.appendChild(logDiv);
    elLogsContainer.scrollTop = elLogsContainer.scrollHeight;
}

// Parameter Update listeners
function updateAecSettings() {
    const mu = parseFloat(elRangeMu.value);
    const taps = parseInt(elSelectTaps.value);
    const vadThreshold = parseFloat(elRangeVad.value);
    const scale = parseFloat(elRangeScale.value);
    const inject = elToggleEcho.checked;
    
    elValMu.innerText = mu.toFixed(3);
    elValVad.innerText = vadThreshold.toFixed(2);
    elValScale.innerText = scale.toFixed(1);
    
    socket.emit('update_settings', {
        taps: taps,
        mu: mu,
        vad_threshold: vadThreshold,
        inject_echo: inject,
        echo_scale: scale
    });
}

[elRangeMu, elSelectTaps, elRangeVad, elToggleEcho, elRangeScale, elSelectAsr, elSelectLlm, elSelectTts].forEach(el => {
    el.addEventListener('change', updateAecSettings);
    el.addEventListener('input', updateAecSettings);
});
