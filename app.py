import os
import re
import json
from datetime import datetime
from pathlib import Path
import requests
from flask import Flask, request, jsonify, render_template_string
from duckduckgo_search import DDGS

app = Flask(__name__)
STATIC_DIR = Path("static/generated")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ========== MEMORIA ==========
MEMORIA_FILE = "memoria.json"

def cargar_memoria():
    if os.path.exists(MEMORIA_FILE):
        with open(MEMORIA_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {"historial": [], "hechos": {}}

def guardar_memoria(data):
    with open(MEMORIA_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

# ========== FUNCIONES DE IA (Groq opcional) ==========

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Lee desde variable de entorno

def consultar_groq(mensajes):
    if not GROQ_API_KEY:
        return "⚙️ Modo local: no hay API key de Groq. Las conversaciones simuladas funcionan, pero para respuestas avanzadas configura GROQ_API_KEY."
    
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": mensajes,
        "max_tokens": 500
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=20)
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"]
        else:
            return f"Error Groq: {r.status_code} - {r.text}"
    except Exception as e:
        return f"Error: {str(e)}"

# ========== BUSCAR WEB (DuckDuckGo) ==========
def buscar_web(query):
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, region='es-es', max_results=3)
            snippets = [f"{r.get('title', '')}: {r.get('body', '')}" for r in results if r.get('body')]
            if snippets:
                return "\n\n".join(snippets)
            else:
                return None
    except Exception as e:
        return f"Error en búsqueda: {e}"

# ========== GENERAR IMAGEN (Pollinations) ==========
def generar_imagen(prompt):
    try:
        from pollinations_api import PollinationsAPI
        api = PollinationsAPI()
        img_filename = f"img_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        img_path = STATIC_DIR / img_filename
        api.generate_image(prompt=prompt, width=512, height=512, nologo=True, save_to=str(img_path))
        return f"/static/generated/{img_filename}"
    except:
        return None

# ========== DETECTOR DE INTENCIÓN ==========
def detectar_intencion(texto):
    t = texto.lower()
    if any(p in t for p in ["investigar", "buscar"]):
        return "buscar"
    if any(p in t for p in ["imagen de", "dibuja", "genera imagen"]):
        return "imagen"
    if any(p in t for p in ["documento sobre", "crea documento"]):
        return "documento"
    if any(p in t for p in ["recuerda que", "guarda que"]):
        return "recordar"
    if any(p in t for p in ["qué recuerdas", "mis datos"]):
        return "recuerdos"
    return "chat"

# ========== PROCESAR MENSAJE ==========
def procesar(mensaje, memoria):
    intencion = detectar_intencion(mensaje)
    
    # Recordar
    if intencion == "recordar":
        texto = re.sub(r'(recuerda que|guarda que)', '', mensaje, flags=re.IGNORECASE).strip()
        if "mi" in texto.lower() and " es " in texto.lower():
            partes = texto.lower().split(" es ", 1)
            clave = partes[0].replace("mi", "").strip()
            valor = partes[1].strip()
            memoria["hechos"][clave] = {"valor": valor, "fecha": datetime.now().isoformat()}
            guardar_memoria(memoria)
            return f"🧠 Recordado: {clave} = {valor}"
        else:
            return "Usa el formato: 'recuerda que mi [dato] es [valor]'"
    
    # Mostrar recuerdos
    if intencion == "recuerdos":
        if not memoria["hechos"]:
            return "No tengo información guardada."
        lines = [f"- {k}: {v['valor']}" for k, v in memoria["hechos"].items()]
        return "🧠 Mis recuerdos:\n" + "\n".join(lines)
    
    # Buscar web
    if intencion == "buscar":
        query = re.sub(r'(investigar|buscar)', '', mensaje, flags=re.IGNORECASE).strip()
        if not query:
            return "¿Qué quieres buscar?"
        resultado = buscar_web(query)
        if resultado:
            return f"🔍 Resultados para '{query}':\n{resultado}"
        else:
            return f"No encontré resultados para '{query}'."
    
    # Generar imagen
    if intencion == "imagen":
        prompt = re.sub(r'(imagen de|dibuja|genera imagen)', '', mensaje, flags=re.IGNORECASE).strip()
        if len(prompt) < 3:
            return "Describe mejor la imagen."
        img_path = generar_imagen(prompt)
        if img_path:
            return img_path  # devuelve ruta, el frontend la mostrará
        else:
            return "No pude generar la imagen. Intenta con otro prompt."
    
    # Documento
    if intencion == "documento":
        tema = re.sub(r'(documento sobre|crea documento)', '', mensaje, flags=re.IGNORECASE).strip()
        if not tema:
            tema = "sin_titulo"
        filename = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = STATIC_DIR / filename
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"Documento: {tema}\nFecha: {datetime.now()}\n\n{tema}")
        return f"📄 Documento guardado: /static/generated/{filename}"
    
    # Chat normal (usar Groq si hay clave)
    if GROQ_API_KEY:
        contexto = memoria["historial"][-5:]  # últimos 5 mensajes
        messages = [{"role": "system", "content": "Eres NYXIA, asistente amable en español."}]
        for h in contexto:
            messages.append({"role": "user", "content": h["user"]})
            messages.append({"role": "assistant", "content": h["assistant"]})
        messages.append({"role": "user", "content": mensaje})
        respuesta = consultar_groq(messages)
    else:
        respuesta = f"🌌 (Modo demo) Recibí: '{mensaje}'. Configura GROQ_API_KEY para respuestas inteligentes."
    
    # Guardar en historial
    memoria["historial"].append({"user": mensaje, "assistant": respuesta, "timestamp": datetime.now().isoformat()})
    if len(memoria["historial"]) > 50:
        memoria["historial"] = memoria["historial"][-50:]
    guardar_memoria(memoria)
    return respuesta

# ========== INTERFAZ HTML (simple, con voz si el navegador soporta) ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYXIA - Asistente IA</title>
    <style>
        body {
            background: #0a0a2a;
            color: #eef;
            font-family: 'Segoe UI', system-ui;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 700px;
            margin: 0 auto;
            background: rgba(0,0,0,0.5);
            border-radius: 20px;
            padding: 20px;
            backdrop-filter: blur(5px);
        }
        h1 {
            text-align: center;
            color: #8ab2ff;
        }
        .messages {
            height: 400px;
            overflow-y: auto;
            padding: 10px;
            background: rgba(0,0,0,0.3);
            border-radius: 15px;
            margin-bottom: 20px;
        }
        .user {
            text-align: right;
            color: #aaffdd;
            margin: 10px;
        }
        .bot {
            text-align: left;
            color: #ffccaa;
            margin: 10px;
            white-space: pre-wrap;
        }
        img {
            max-width: 100%;
            border-radius: 10px;
            margin-top: 10px;
        }
        .input-area {
            display: flex;
            gap: 10px;
        }
        input {
            flex: 1;
            padding: 12px;
            border-radius: 30px;
            border: none;
            background: #1e1e3e;
            color: white;
            font-size: 16px;
        }
        button {
            padding: 12px 20px;
            border-radius: 30px;
            border: none;
            background: #8ab2ff;
            color: black;
            font-weight: bold;
            cursor: pointer;
        }
        .mic-btn {
            background: #ff6680;
        }
        button:hover {
            opacity: 0.8;
        }
        .typing {
            color: #aaa;
            font-style: italic;
        }
        hr {
            border-color: #334;
        }
    </style>
</head>
<body>
<div class="container">
    <h1>🌌 NYXIA - Asistente IA</h1>
    <div class="messages" id="messages">
        <div class="bot">🌌 Hola, soy NYXIA. Pregúntame cualquier cosa. También puedo investigar, crear imágenes y guardar información.</div>
    </div>
    <div class="input-area">
        <input type="text" id="input" placeholder="Escribe o usa el micrófono...">
        <button id="sendBtn">➤</button>
        <button id="micBtn" class="mic-btn">🎤</button>
    </div>
</div>

<script>
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');
    const micBtn = document.getElementById('micBtn');
    const messagesDiv = document.getElementById('messages');
    let recognition = null;
    let isListening = false;
    
    function addMessage(text, isUser) {
        const div = document.createElement('div');
        div.className = isUser ? 'user' : 'bot';
        // Si el texto es una ruta de imagen
        if (!isUser && text.startsWith('/static/generated/') && (text.endsWith('.png') || text.endsWith('.jpg'))) {
            const img = document.createElement('img');
            img.src = text;
            div.appendChild(img);
        } else {
            div.innerText = text;
        }
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
    
    async function sendMessage() {
        const text = input.value.trim();
        if (!text) return;
        addMessage(text, true);
        input.value = '';
        // Indicador de escritura
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing';
        typingDiv.innerText = 'NYXIA está pensando...';
        messagesDiv.appendChild(typingDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mensaje: text})
            });
            const data = await response.json();
            typingDiv.remove();
            addMessage(data.respuesta, false);
            if (data.imagen) {
                addMessage(data.imagen, false);
            }
        } catch(e) {
            typingDiv.remove();
            addMessage('❌ Error de conexión', false);
        }
    }
    
    sendBtn.onclick = sendMessage;
    input.addEventListener('keypress', (e) => { if (e.key === 'Enter') sendMessage(); });
    
    // Reconocimiento de voz (opcional)
    if ('webkitSpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'es-ES';
        recognition.continuous = false;
        recognition.interimResults = false;
        
        recognition.onstart = () => {
            isListening = true;
            micBtn.style.background = '#ff3366';
        };
        recognition.onend = () => {
            isListening = false;
            micBtn.style.background = '#ff6680';
        };
        recognition.onresult = (event) => {
            const texto = event.results[0][0].transcript;
            input.value = texto;
            sendMessage();
        };
        micBtn.onclick = () => {
            if (recognition && !isListening) {
                try { recognition.start(); } catch(e) {}
            }
        };
    } else {
        micBtn.disabled = true;
        micBtn.title = "Voz no soportada";
    }
</script>
</body>
</html>
"""

# ========== FLASK RUTAS ==========
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    mensaje = data.get('mensaje', '').strip()
    if not mensaje:
        return jsonify({'respuesta': 'No recibí mensaje.'})
    
    memoria = cargar_memoria()
    respuesta = procesar(mensaje, memoria)
    
    # Si la respuesta es una ruta de imagen
    if isinstance(respuesta, str) and respuesta.startswith('/static/generated/'):
        return jsonify({'respuesta': '✅ Imagen generada', 'imagen': respuesta})
    else:
        return jsonify({'respuesta': respuesta})

@app.route('/static/generated/<path:filename>')
def serve_image(filename):
    return send_from_directory(STATIC_DIR, filename)

# Necesario para servir archivos estáticos
from flask import send_from_directory

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860, debug=False)
