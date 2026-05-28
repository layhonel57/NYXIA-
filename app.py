import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# Configurar logging para ver errores en la consola de Replit
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
STATIC_DIR = Path("static/generated")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ========== CONFIGURACIÓN SEGURA ==========
# En Replit, ve a Tools > Secrets y agrega GROQ_API_KEY
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY = 50

# ========== MEMORIA ==========
MEMORIA_FILE = "memoria.json"

def cargar_memoria() -> Dict:
    try:
        if os.path.exists(MEMORIA_FILE):
            with open(MEMORIA_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception:
        pass
    return {"historial": [], "hechos": {}}

def guardar_memoria(data: Dict) -> bool:
    try:
        with open(MEMORIA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        logger.error(f"Error guardando memoria: {e}")
        return False

# ========== GROQ API (CORREGIDO) ==========
def consultar_groq(mensajes: List[Dict[str, str]]) -> str:
    if not GROQ_API_KEY:
        return "NO_KEY"
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # ¡EL ERROR 400 ESTABA AQUÍ! Antes decía "messages": GROQ_API_KEY
    payload = {
        "model": GROQ_MODEL,
        "messages": mensajes, 
        "max_tokens": 800,
        "temperature": 0.7
    }
    
    try:
        response = requests.post(GROQ_API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        logger.error(f"Error HTTP Groq: {e.response.status_code} - {e.response.text}")
        return "ERROR_API"
    except Exception as e:
        logger.error(f"Error conexión Groq: {e}")
        return "ERROR_API"

# ========== BÚSQUEDA WEB (MEJORADA) ==========
def buscar_web(query: str) -> Optional[str]:
    try:
        from duckduckgo_search import DDGS
        
        with DDGS() as ddgs:
            # Añadimos más parámetros para evitar resultados basura
            results = list(ddgs.text(query, region="wt-wt", safesearch="moderate", max_results=4))
        
        if not results:
            return None
            
        snippets = []
        for i, r in enumerate(results, 1):
            title = r.get('title', '')
            body = r.get('body', '')
            href = r.get('href', '')
            # Filtro de seguridad para no mostrar basura
            if body and len(body) > 20 and query.lower().split()[0] in body.lower()+title.lower():
                snippet = f"**{i}. {title}**\n{body}"
                if href:
                    snippet += f"\n🔗 {href}"
                snippets.append(snippet)
                
        return "\n\n".join(snippets) if snippets else None
        
    except Exception as e:
        logger.error(f"Error búsqueda: {e}")
        return None

# ========== GENERAR IMAGEN ==========
def generar_imagen(prompt: str) -> Optional[str]:
    try:
        encoded_prompt = requests.utils.quote(prompt)
        seed = int(datetime.now().timestamp())
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&seed={seed}&nologo=true"
        
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        img_filename = f"img_{seed}.png"
        img_path = STATIC_DIR / img_filename
        with open(img_path, 'wb') as f:
            f.write(response.content)
        return f"/static/generated/{img_filename}"
    except Exception as e:
        logger.error(f"Error imagen: {e}")
        return None

# ========== INTENCIONES ==========
INTENT_PATTERNS = {
    "buscar": ["investiga", "buscar", "busca", "qué es", "quién es", "dime sobre", "infórmate"],
    "imagen": ["imagen de", "dibuja", "genera imagen", "crea imagen", "generar gato", "haz una imagen"],
    "documento": ["documento sobre", "crea documento", "genera documento"],
    "recordar": ["recuerda que", "guarda que", "anota que"],
    "recuerdos": ["qué recuerdas", "mis datos", "qué sabes"],
    "borrar": ["olvida", "borra mi"],
    "ayuda": ["ayuda", "qué puedes hacer", "funciones"]
}

def detectar_intencion(texto: str) -> Tuple[str, str]:
    t = texto.lower().strip()
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in t:
                contenido = re.sub(re.escape(pattern), '', t, flags=re.IGNORECASE).strip()
                contenido = re.sub(r'^[\s,:;]+', '', contenido).strip()
                return intent, contenido
    return "chat", texto

# ========== PROCESAR MENSAJE ==========
def procesar(mensaje: str, memoria: Dict) -> Dict:
    intencion, contenido = detectar_intencion(mensaje)
    resultado = {"respuesta": "", "imagen": None, "documento": None}
    
    if intencion == "ayuda":
        resultado["respuesta"] = "🤖 **Comandos:**\n- 🔍 *Investiga sobre [tema]*\n- 🎨 *Genera imagen de [algo]*\n- 📄 *Crea documento sobre [tema]*\n- 🧠 *Recuerda que mi [dato] es [valor]*\n- 📋 *Qué recuerdas?*\n- 🗑️ *Olvida [dato]*"
        return resultado
    
    if intencion == "recordar":
        match = re.search(r'(?:mi\s+)?(\w[\w\s]*?)\s+es\s+(.+)', contenido, re.IGNORECASE)
        if match:
            clave, valor = match.group(1).strip().lower(), match.group(2).strip()
            memoria["hechos"][clave] = {"valor": valor}
            guardar_memoria(memoria)
            resultado["respuesta"] = f"🧠 ¡Guardado! Sé que tu/a {clave} es {valor}"
        else:
            resultado["respuesta"] = "💡 Usa el formato: *recuerda que mi nombre es Juan*"
        return resultado
        
    if intencion == "recuerdos":
        if not memoria["hechos"]:
            resultado["respuesta"] = "🧠 Aún no guardo nada sobre ti."
        else:
            lines = [f"• **{k}**: {v['valor']}" for k, v in memoria["hechos"].items()]
            resultado["respuesta"] = "🧠 **Lo que sé:**\n" + "\n".join(lines)
        return resultado
        
    if intencion == "borrar":
        clave = contenido.lower().strip()
        if clave in memoria["hechos"]:
            del memoria["hechos"][clave]
            guardar_memoria(memoria)
            resultado["respuesta"] = f"🗑️ Olvidé: {clave}"
        else:
            resultado["respuesta"] = f"❌ No tengo guardado '{clave}'"
        return resultado
    
    if intencion == "buscar":
        query = contenido or mensaje
        resultado["respuesta"] = f"🔍 Buscando: *{query}*...\n\n"
        info = buscar_web(query)
        if info:
            resultado["respuesta"] += info
        else:
            resultado["respuesta"] += "❌ No encontré información útil. Intenta reformular."
        return resultado
        
    if intencion == "imagen":
        prompt = contenido or mensaje
        if len(prompt) < 4:
            resultado["respuesta"] = "💡 Describe la imagen mejor."
            return resultado
        img = generar_imagen(prompt)
        if img:
            resultado["imagen"] = img
            resultado["respuesta"] = f"✅ Imagen generada"
        else:
            resultado["respuesta"] = "❌ Error al generar la imagen."
        return resultado
        
    if intencion == "documento":
        tema = contenido or "sin_titulo"
        filename = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = STATIC_DIR / filename
        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"DOCUMENTO: {tema}\nFecha: {datetime.now()}\n\n{tema}")
        resultado["documento"] = f"/static/generated/{filename}"
        resultado["respuesta"] = f"📄 Documento creado: {tema}"
        return resultado
    
    # === CHAT CON IA ===
    if GROQ_API_KEY:
        sistema = "Eres NYXIA, asistente amable y concisa en español."
        if memoria["hechos"]:
            sistema += "\nInfo del usuario: " + ", ".join([f"{k}={v['valor']}" for k,v in memoria["hechos"].items()])
            
        messages = [{"role": "system", "content": sistema}]
        for h in memoria.get("historial", [])[-6:]:
            messages.append({"role": "user", "content": h.get("user", "")})
            messages.append({"role": "assistant", "content": h.get("assistant", "")})
        messages.append({"role": "user", "content": mensaje})
        
        respuesta = consultar_groq(messages)
        
        if respuesta == "NO_KEY":
            resultado["respuesta"] = "🌌 Modo demo (Sin API Key)"
        elif respuesta == "ERROR_API":
            resultado["respuesta"] = "⚠️ Tu API Key de Groq parece inválida o hay un error de conexión. Revisa el Secret en Replit."
        else:
            resultado["respuesta"] = respuesta
    else:
        resultado["respuesta"] = "🌌 **Modo Demo**\n\nRecibí: *" + mensaje + "*\n\n*(Para respuestas inteligentes, agrega tu `GROQ_API_KEY` en **Tools > Secrets** en Replit)*"
    
    # Guardar historial
    memoria["historial"].append({"user": mensaje, "assistant": resultado["respuesta"]})
    if len(memoria["historial"]) > MAX_HISTORY:
        memoria["historial"] = memoria["historial"][-MAX_HISTORY:]
    guardar_memoria(memoria)
    
    return resultado

# ========== HTML INTERFAZ ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYXIA - Asistente IA</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { background: #0a0a2a; color: #eef; font-family: 'Segoe UI', sans-serif; min-height: 100vh; padding: 20px; }
        .container { max-width: 800px; margin: 0 auto; background: rgba(15,15,40,0.8); border-radius: 24px; padding: 24px; border: 1px solid rgba(138,178,255,0.1); }
        h1 { text-align: center; color: #8ab2ff; font-weight: 300; letter-spacing: 4px; }
        .sub { text-align: center; color: rgba(138,178,255,0.5); font-size: 0.85em; margin-bottom: 20px; }
        .messages { height: 450px; overflow-y: auto; padding: 16px; background: rgba(0,0,0,0.3); border-radius: 16px; margin-bottom: 20px; }
        .messages::-webkit-scrollbar { width: 6px; }
        .messages::-webkit-scrollbar-thumb { background: rgba(138,178,255,0.3); border-radius: 3px; }
        .msg { max-width: 85%; padding: 12px 16px; border-radius: 18px; margin-bottom: 12px; line-height: 1.5; animation: fadeIn 0.3s ease; white-space: pre-wrap; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        .msg.user { margin-left: auto; background: linear-gradient(135deg, #4a6fa5, #3a5f95); border-bottom-right-radius: 4px; }
        .msg.bot { margin-right: auto; background: rgba(255,255,255,0.08); border-bottom-left-radius: 4px; }
        .msg img { max-width: 100%; border-radius: 12px; margin-top: 8px; }
        .msg a { color: #7af; }
        .typing { color: #667; font-style: italic; padding: 10px; }
        .typing::after { content: '...'; animation: dots 1s infinite; }
        @keyframes dots { 0%{opacity:0} 50%{opacity:1} 100%{opacity:0} }
        .input-area { display: flex; gap: 10px; }
        input { flex: 1; padding: 14px 20px; border-radius: 30px; border: 2px solid rgba(138,178,255,0.2); background: rgba(0,0,0,0.4); color: white; font-size: 16px; outline: none; }
        input:focus { border-color: rgba(138,178,255,0.6); }
        .btn { width: 48px; height: 48px; border-radius: 50%; border: none; cursor: pointer; font-size: 1.2em; display: flex; align-items: center; justify-content: center; transition: 0.2s; }
        .btn:hover { transform: scale(1.05); }
        .btn-send { background: #8ab2ff; color: #0a0a2a; }
        .btn-mic { background: #ff6680; color: white; }
        .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
    </style>
</head>
<body>
<div class="container">
    <h1>🌌 NYXIA</h1>
    <div class="sub">Versión 2.0 Estable</div>
    <div class="messages" id="messages">
        <div class="msg bot">🚀 Versión actualizada cargada correctamente. Puedes escribir "ayuda" para ver mis funciones.</div>
    </div>
    <div class="input-area">
        <input type="text" id="input" placeholder="Escribe algo..." autocomplete="off">
        <button class="btn btn-send" id="sendBtn">➤</button>
        <button class="btn btn-mic" id="micBtn">🎤</button>
    </div>
</div>

<script>
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');
    const micBtn = document.getElementById('micBtn');
    const msgs = document.getElementById('messages');
    let isProcessing = false;

    function addMsg(text, isUser, img=null) {
        const d = document.createElement('div');
        d.className = `msg ${isUser ? 'user' : 'bot'}`;
        d.innerHTML = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>').replace(/\*(.+?)\*/g, '<em>$1</em>').replace(/\n/g, '<br>');
        if (img) { const i = document.createElement('img'); i.src = img; d.appendChild(i); }
        msgs.appendChild(d);
        msgs.scrollTop = msgs.scrollHeight;
    }

    async function send(text) {
        const t = text || input.value.trim();
        if (!t || isProcessing) return;
        isProcessing = true; sendBtn.disabled = true; input.value = '';
        addMsg(t, true);
        const typ = document.createElement('div'); typ.className = 'typing'; typ.innerText = 'NYXIA está pensando'; msgs.appendChild(typ); msgs.scrollTop = msgs.scrollHeight;
        try {
            const r = await fetch('/chat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({mensaje: t}) });
            const d = await r.json(); typ.remove();
            addMsg(d.respuesta, false, d.imagen);
        } catch(e) { typ.remove(); addMsg('Error de conexión', false); }
        isProcessing = false; sendBtn.disabled = false; input.focus();
    }

    sendBtn.onclick = () => send();
    input.addEventListener('keypress', e => { if (e.key === 'Enter') send(); });

    if ('webkitSpeechRecognition' in window) {
        const rec = new webkitSpeechRecognition(); rec.lang = 'es-ES'; rec.continuous = false;
        rec.onresult = e => { input.value = e.results[0][0].transcript; send(); };
        micBtn.onclick = () => rec.start();
    } else { micBtn.disabled = true; }
</script>
</body>
</html>"""

# ========== RUTAS FLASK ==========
@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        mensaje = data.get('mensaje', '').strip()
        if not mensaje: return jsonify({'respuesta': 'Vacío'}), 400
        
        memoria = cargar_memoria()
        resultado = procesar(mensaje, memoria)
        return jsonify(resultado)
    except Exception as e:
        logger.error(f"Error general: {e}", exc_info=True)
        return jsonify({'respuesta': f'Error interno: {str(e)}'}), 500

@app.route('/static/generated/<path:filename>')
def serve_static(filename):
    return send_from_directory(STATIC_DIR, filename)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🚀 NYXIA v2.0 iniciando en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
