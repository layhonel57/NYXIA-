import os
import re
import json
import logging
from datetime import datetime
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, List, Tuple

import requests
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
STATIC_DIR = Path("static/generated")
STATIC_DIR.mkdir(parents=True, exist_ok=True)

# ========== CONFIGURACIÓN ==========
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")  # Configurar como variable de entorno
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.3-70b-versatile"
MAX_HISTORY = 50
MAX_CONTEXT_MESSAGES = 6
REQUEST_TIMEOUT = 30

# ========== MEMORIA CON CACHÉ ==========
MEMORIA_FILE = "memoria.json"
_memoria_cache: Optional[Dict] = None
_memoria_cache_time: Optional[datetime] = None
CACHE_TTL_SECONDS = 5

def cargar_memoria() -> Dict:
    """Carga memoria con caché para evitar lecturas frecuentes del disco."""
    global _memoria_cache, _memoria_cache_time
    
    now = datetime.now()
    if (_memoria_cache is not None and 
        _memoria_cache_time and 
        (now - _memoria_cache_time).total_seconds() < CACHE_TTL_SECONDS):
        return _memoria_cache
    
    try:
        if os.path.exists(MEMORIA_FILE):
            with open(MEMORIA_FILE, 'r', encoding='utf-8') as f:
                _memoria_cache = json.load(f)
        else:
            _memoria_cache = {"historial": [], "hechos": {}, "preferencias": {}}
    except (json.JSONDecodeError, IOError) as e:
        logger.warning(f"Error cargando memoria: {e}")
        _memoria_cache = {"historial": [], "hechos": {}, "preferencias": {}}
    
    _memoria_cache_time = now
    return _memoria_cache

def guardar_memoria(data: Dict) -> bool:
    """Guarda memoria e invalida caché."""
    global _memoria_cache, _memoria_cache_time
    try:
        with open(MEMORIA_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        _memoria_cache = data
        _memoria_cache_time = datetime.now()
        return True
    except IOError as e:
        logger.error(f"Error guardando memoria: {e}")
        return False

# ========== CONSULTAR GROQ (SÍNCRONA) ==========
def consultar_groq(mensajes: List[Dict[str, str]]) -> str:
    """Consulta la API de Groq de forma síncrona."""
    if not GROQ_API_KEY:
        return None
    
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": GROQ_MODEL,
        "messages": mensajes,
        "max_tokens": 800,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    try:
        response = requests.post(
            GROQ_API_URL, 
            headers=headers, 
            json=payload, 
            timeout=REQUEST_TIMEOUT
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.error("Timeout en consulta Groq")
        return "⏱️ La consulta tardó demasiado. Intenta de nuevo."
    except requests.exceptions.HTTPError as e:
        logger.error(f"Error HTTP Groq: {e.response.status_code}")
        if e.response.status_code == 401:
            return "🔑 API key inválida. Verifica la configuración."
        if e.response.status_code == 429:
            return "⏳ Demasiadas peticiones. Espera un momento."
        return f"❌ Error del servidor: {e.response.status_code}"
    except requests.exceptions.RequestException as e:
        logger.error(f"Error de conexión Groq: {e}")
        return "❌ No pude conectar con el servicio de IA."

# ========== BUSCAR WEB (DuckDuckGo) ==========
def buscar_web(query: str) -> Optional[str]:
    """Busca en la web usando DuckDuckGo."""
    try:
        from duckduckgo_search import DDGS
        
        if not query or len(query) < 3:
            return None
            
        with DDGS() as ddgs:
            results = list(ddgs.text(query, region='es-es', max_results=3))
            
        if not results:
            return None
            
        snippets = []
        for i, r in enumerate(results, 1):
            title = r.get('title', 'Sin título')
            body = r.get('body', '')
            href = r.get('href', '')
            if body:
                snippet = f"**{i}. {title}**\n{body}"
                if href:
                    snippet += f"\n🔗 {href}"
                snippets.append(snippet)
        
        return "\n\n".join(snippets) if snippets else None
        
    except ImportError:
        logger.warning("duckduckgo_search no instalado")
        return "⚠️ Módulo de búsqueda no disponible. Instala: pip install duckduckgo-search"
    except Exception as e:
        logger.error(f"Error en búsqueda: {e}")
        return f"❌ Error en búsqueda: {str(e)}"

# ========== GENERAR IMAGEN (Pollinations) ==========
def generar_imagen(prompt: str) -> Optional[str]:
    """Genera una imagen usando Pollinations API."""
    try:
        # Usar la API directa sin dependencia adicional
        encoded_prompt = requests.utils.quote(prompt)
        seed = datetime.now().timestamp()
        url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=768&height=768&seed={seed}&nologo=true"
        
        # Descargar y guardar localmente
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        img_filename = f"img_{int(seed)}.png"
        img_path = STATIC_DIR / img_filename
        
        with open(img_path, 'wb') as f:
            f.write(response.content)
            
        return f"/static/generated/{img_filename}"
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error generando imagen: {e}")
        return None

# ========== DETECTOR DE INTENCIÓN MEJORADO ==========
INTENT_PATTERNS: Dict[str, List[str]] = {
    "buscar": ["investigar", "buscar", "busca", "buscar información", "qué es", "quién es", 
               "cuándo fue", "dime sobre", "infórmate sobre", "busca en internet"],
    "imagen": ["imagen de", "dibuja", "genera imagen", "crea imagen", "dibujar", "pinta",
               "generar imagen", "crea una imagen", "haz una imagen", "foto de"],
    "documento": ["documento sobre", "crea documento", "genera documento", "escribe un",
                  "redacta", "crear texto", "escribe sobre"],
    "recordar": ["recuerda que", "guarda que", "anota que", "apunta que", "no olvides que",
                 "memoriza que", "registra que"],
    "recuerdos": ["qué recuerdas", "mis datos", "qué sabes de mí", "mi información",
                  "lo que sabes", "qué tienes guardado", "ver recuerdos", "mis notas"],
    "borrar": ["olvida", "borra", "elimina recuerdo", "borrar recuerdo"],
    "ayuda": ["ayuda", "qué puedes hacer", "funciones", "comandos", "help", "cómo usarte"]
}

def detectar_intencion(texto: str) -> Tuple[str, str]:
    """Detecta la intención y extrae el contenido relevante."""
    t = texto.lower().strip()
    
    for intent, patterns in INTENT_PATTERNS.items():
        for pattern in patterns:
            if pattern in t:
                # Extraer contenido removiendo el patrón
                contenido = re.sub(re.escape(pattern), '', t, flags=re.IGNORECASE).strip()
                # Limpiar espacios múltiples y caracteres iniciales
                contenido = re.sub(r'^[\s,:;]+', '', contenido).strip()
                return intent, contenido
    
    return "chat", texto

# ========== RESPUESTA DE AYUDA ==========
def obtener_ayuda() -> str:
    """Genera mensaje de ayuda con todos los comandos disponibles."""
    return """🤖 **¿Qué puedo hacer?**

🔍 **Investigar**: "Investiga sobre [tema]" o "Busca [consulta]"
🎨 **Imágenes**: "Genera imagen de [descripción]" o "Dibuja [algo]"
📄 **Documentos**: "Crea documento sobre [tema]"
🧠 **Recordar**: "Recuerda que mi [dato] es [valor]"
📋 **Ver recuerdos**: "Qué recuerdas" o "Mis datos"
🗑️ **Olvidar**: "Olvida [dato]"
💬 **Charlar**: Simplemente escribe cualquier cosa

💡 **Ejemplos**:
• "Investiga sobre inteligencia artificial"
• "Genera imagen de un gato astronauta"
• "Recuerda que mi color favorito es el azul"
• "Qué recuerdas sobre mí?"
"""

# ========== PROCESAR MENSAJE ==========
def procesar(mensaje: str, memoria: Dict, session_id: str = "default") -> Dict:
    """Procesa un mensaje y retorna respuesta estructurada."""
    intencion, contenido = detectar_intencion(mensaje)
    
    resultado = {
        "respuesta": "",
        "imagen": None,
        "documento": None,
        "tipo": intencion
    }
    
    # --- AYUDA ---
    if intencion == "ayuda":
        resultado["respuesta"] = obtener_ayuda()
        return resultado
    
    # --- RECORDAR ---
    if intencion == "recordar":
        if not contenido:
            resultado["respuesta"] = "💡 Usa el formato: 'recuerda que mi [dato] es [valor]'"
            return resultado
            
        # Patrones: "mi X es Y" o "X es Y"
        match = re.search(r'(?:mi\s+)?(\w[\w\s]*?)\s+es\s+(.+)', contenido, re.IGNORECASE)
        if match:
            clave = match.group(1).strip().lower()
            valor = match.group(2).strip()
            memoria["hechos"][clave] = {
                "valor": valor, 
                "fecha": datetime.now().isoformat()
            }
            guardar_memoria(memoria)
            resultado["respuesta"] = f"🧠 ¡Guardado! Ahora sé que tu/a {clave} es {valor}"
        else:
            # Guardar como nota libre
            nota_key = f"nota_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            memoria["hechos"][nota_key] = {
                "valor": contenido,
                "fecha": datetime.now().isoformat(),
                "tipo": "nota"
            }
            guardar_memoria(memoria)
            resultado["respuesta"] = f"📝 Nota guardada: {contenido}"
        return resultado
    
    # --- BORRAR RECUERDO ---
    if intencion == "borrar":
        if not contenido:
            resultado["respuesta"] = "💡 ¿Qué quieres que olvide? Ej: 'Olvida mi nombre'"
            return resultado
            
        clave = contenido.lower().strip()
        # Buscar coincidencias parciales
        encontrados = [k for k in memoria["hechos"] if clave in k.lower()]
        
        if encontrados:
            for k in encontrados:
                del memoria["hechos"][k]
            guardar_memoria(memoria)
            resultado["respuesta"] = f"🗑️ Borrado: {', '.join(encontrados)}"
        else:
            resultado["respuesta"] = f"🔍 No encontré nada sobre '{contenido}' para borrar."
        return resultado
    
    # --- MOSTRAR RECUERDOS ---
    if intencion == "recuerdos":
        if not memoria["hechos"]:
            resultado["respuesta"] = "🧠 No tengo información guardada sobre ti aún.\n\nUsa 'recuerda que mi [dato] es [valor]' para guardar algo."
            return resultado
            
        lines = []
        for k, v in memoria["hechos"].items():
            if v.get("tipo") == "nota":
                lines.append(f"📝 {v['valor']}")
            else:
                lines.append(f"• **{k}**: {v['valor']}")
        
        resultado["respuesta"] = "🧠 **Lo que sé de ti:**\n\n" + "\n".join(lines)
        return resultado
    
    # --- BUSCAR WEB ---
    if intencion == "buscar":
        query = contenido or mensaje
        resultado["respuesta"] = f"🔍 Buscando: *{query}*...\n\n"
        
        resultado_busqueda = buscar_web(query)
        if resultado_busqueda:
            resultado["respuesta"] += resultado_busqueda
        else:
            resultado["respuesta"] += "❌ No encontré resultados relevantes."
        return resultado
    
    # --- GENERAR IMAGEN ---
    if intencion == "imagen":
        prompt = contenido or mensaje
        if len(prompt) < 5:
            resultado["respuesta"] = "💡 Describe mejor la imagen que quieres generar."
            return resultado
            
        resultado["respuesta"] = f"🎨 Generando imagen: *{prompt}*..."
        img_path = generar_imagen(prompt)
        
        if img_path:
            resultado["imagen"] = img_path
            resultado["respuesta"] = f"✅ **Imagen generada:** {prompt}"
        else:
            resultado["respuesta"] = "❌ No pude generar la imagen. Intenta con otra descripción."
        return resultado
    
    # --- DOCUMENTO ---
    if intencion == "documento":
        tema = contenido or "sin_titulo"
        filename = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        path = STATIC_DIR / filename
        
        contenido_doc = f"""{'='*50}
DOCUMENTO: {tema.upper()}
{'='*50}
Fecha de creación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Generado por: NYXIA IA

{'~'*50}
{tema}

{'~'*50}
NOTAS:
- Este documento fue generado automáticamente
- Puedes editarlo según tus necesidades
{'='*50}
"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(contenido_doc)
            
        resultado["documento"] = f"/static/generated/{filename}"
        resultado["respuesta"] = f"📄 **Documento creado**: {tema}\n📥 [Descargar]({resultado['documento']})"
        return resultado
    
    # --- CHAT CON GROQ ---
    if GROQ_API_KEY:
        # Construir contexto con recuerdos
        sistema = "Eres NYXIA, un asistente de IA amable, inteligente y útil que responde en español."
        
        if memoria["hechos"]:
            hechos_str = "\n".join([f"- {k}: {v['valor']}" for k, v in memoria["hechos"].items() 
                                   if v.get("tipo") != "nota"])
            if hechos_str:
                sistema += f"\n\nInformación que sabes del usuario:\n{hechos_str}"
        
        messages = [{"role": "system", "content": sistema}]
        
        # Agregar historial reciente
        historial = memoria.get("historial", [])[-MAX_CONTEXT_MESSAGES:]
        for h in historial:
            messages.append({"role": "user", "content": h.get("user", "")})
            if h.get("assistant"):
                messages.append({"role": "assistant", "content": h["assistant"]})
        
        messages.append({"role": "user", "content": mensaje})
        
        respuesta = consultar_groq(messages)
        resultado["respuesta"] = respuesta or "⚙️ No obtuve respuesta del servicio de IA."
    else:
        resultado["respuesta"] = (
            f"🌌 **Modo demo** - Recibí: *'{mensaje}'*\n\n"
            "🔑 Para respuestas inteligentes, configura la variable de entorno:\n"
            "`export GROQ_API_KEY=tu_api_key_aqui`"
        )
    
    # Guardar en historial
    memoria.setdefault("historial", []).append({
        "user": mensaje, 
        "assistant": resultado["respuesta"],
        "timestamp": datetime.now().isoformat()
    })
    
    # Limitar historial
    if len(memoria["historial"]) > MAX_HISTORY:
        memoria["historial"] = memoria["historial"][-MAX_HISTORY:]
    
    guardar_memoria(memoria)
    return resultado

# ========== INTERFAZ HTML MEJORADA ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NYXIA - Asistente IA</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        
        body {
            background: linear-gradient(135deg, #0a0a2a 0%, #1a1a3a 50%, #0a0a2a 100%);
            color: #eef;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 800px;
            margin: 0 auto;
            background: rgba(15, 15, 40, 0.8);
            border-radius: 24px;
            padding: 24px;
            backdrop-filter: blur(20px);
            border: 1px solid rgba(138, 178, 255, 0.1);
            box-shadow: 0 20px 60px rgba(0,0,0,0.5);
        }
        
        .header {
            text-align: center;
            padding: 16px 0 24px;
            border-bottom: 1px solid rgba(138, 178, 255, 0.15);
            margin-bottom: 20px;
        }
        
        .header h1 {
            color: #8ab2ff;
            font-size: 2em;
            font-weight: 300;
            letter-spacing: 4px;
        }
        
        .header .subtitle {
            color: rgba(138, 178, 255, 0.5);
            font-size: 0.85em;
            margin-top: 8px;
        }
        
        .messages {
            height: 450px;
            overflow-y: auto;
            padding: 16px;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 16px;
            margin-bottom: 20px;
            scroll-behavior: smooth;
        }
        
        .messages::-webkit-scrollbar { width: 6px; }
        .messages::-webkit-scrollbar-track { background: transparent; }
        .messages::-webkit-scrollbar-thumb { 
            background: rgba(138, 178, 255, 0.3); 
            border-radius: 3px; 
        }
        
        .message {
            max-width: 85%;
            padding: 12px 16px;
            border-radius: 18px;
            margin-bottom: 12px;
            line-height: 1.5;
            word-wrap: break-word;
            animation: fadeIn 0.3s ease;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .message.user {
            margin-left: auto;
            background: linear-gradient(135deg, #4a6fa5, #3a5f95);
            border-bottom-right-radius: 4px;
        }
        
        .message.bot {
            margin-right: auto;
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-bottom-left-radius: 4px;
        }
        
        .message.bot strong { color: #8ab2ff; }
        .message.bot em { color: #aab2ff; }
        .message.bot a { color: #7af; text-decoration: underline; }
        
        .message img {
            max-width: 100%;
            border-radius: 12px;
            margin-top: 8px;
            cursor: pointer;
            transition: transform 0.2s;
        }
        
        .message img:hover { transform: scale(1.02); }
        
        .message .doc-link {
            display: inline-block;
            margin-top: 8px;
            padding: 8px 16px;
            background: rgba(138, 178, 255, 0.2);
            border-radius: 8px;
            color: #8ab2ff;
            text-decoration: none;
            font-size: 0.9em;
        }
        
        .message .doc-link:hover { background: rgba(138, 178, 255, 0.3); }
        
        .typing {
            color: #667;
            font-style: italic;
            padding: 12px 16px;
        }
        
        .typing::after {
            content: '';
            animation: dots 1.5s infinite;
        }
        
        @keyframes dots {
            0%, 20% { content: '.'; }
            40% { content: '..'; }
            60%, 100% { content: '...'; }
        }
        
        .input-area {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .input-wrapper {
            flex: 1;
            position: relative;
        }
        
        input {
            width: 100%;
            padding: 14px 20px;
            border-radius: 30px;
            border: 2px solid rgba(138, 178, 255, 0.2);
            background: rgba(0, 0, 0, 0.4);
            color: white;
            font-size: 16px;
            transition: border-color 0.3s;
            outline: none;
        }
        
        input:focus { border-color: rgba(138, 178, 255, 0.5); }
        input::placeholder { color: rgba(255, 255, 255, 0.3); }
        
        .btn {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            border: none;
            cursor: pointer;
            font-size: 1.2em;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
        }
        
        .btn:hover { transform: scale(1.05); }
        .btn:active { transform: scale(0.95); }
        
        .btn-send {
            background: linear-gradient(135deg, #8ab2ff, #6a92df);
            color: #0a0a2a;
        }
        
        .btn-mic {
            background: linear-gradient(135deg, #ff6680, #ff4466);
            color: white;
        }
        
        .btn-mic.listening {
            background: linear-gradient(135deg, #ff3366, #ff1144);
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { box-shadow: 0 0 0 0 rgba(255, 51, 102, 0.4); }
            50% { box-shadow: 0 0 0 12px rgba(255, 51, 102, 0); }
        }
        
        .btn-help {
            background: rgba(255, 255, 255, 0.1);
            color: #8ab2ff;
            font-size: 1em;
        }
        
        .btn:disabled {
            opacity: 0.4;
            cursor: not-allowed;
            transform: none;
        }
        
        .quick-actions {
            display: flex;
            gap: 8px;
            margin-bottom: 16px;
            flex-wrap: wrap;
        }
        
        .quick-btn {
            padding: 6px 14px;
            border-radius: 20px;
            border: 1px solid rgba(138, 178, 255, 0.2);
            background: rgba(138, 178, 255, 0.08);
            color: #8ab2ff;
            font-size: 0.8em;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-btn:hover {
            background: rgba(138, 178, 255, 0.15);
            border-color: rgba(138, 178, 255, 0.4);
        }
        
        /* Modal para imagen ampliada */
        .modal {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(0,0,0,0.9);
            z-index: 1000;
            align-items: center;
            justify-content: center;
            cursor: pointer;
        }
        
        .modal.active { display: flex; }
        
        .modal img {
            max-width: 90%;
            max-height: 90%;
            border-radius: 12px;
        }
        
        @media (max-width: 600px) {
            body { padding: 10px; }
            .container { padding: 16px; border-radius: 16px; }
            .header h1 { font-size: 1.5em; }
            .messages { height: 350px; }
            .message { max-width: 92%; }
        }
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>🌌 NYXIA</h1>
        <div class="subtitle">Asistente Inteligente • Búsqueda • Imágenes • Memoria</div>
    </div>
    
    <div class="quick-actions">
        <button class="quick-btn" onclick="sendQuick('¿Qué puedes hacer?')">🤖 Ayuda</button>
        <button class="quick-btn" onclick="sendQuick('Investiga sobre inteligencia artificial')">🔍 Investigar</button>
        <button class="quick-btn" onclick="sendQuick('Genera imagen de un gato astronauta')">🎨 Imagen</button>
        <button class="quick-btn" onclick="sendQuick('Qué recuerdas?')">🧠 Recuerdos</button>
    </div>
    
    <div class="messages" id="messages">
        <div class="message bot">
            🌌 Hola, soy <strong>NYXIA</strong>. Puedo ayudarte con muchas cosas:<br><br>
            • 🔍 <strong>Investigar</strong> en internet<br>
            • 🎨 <strong>Generar imágenes</strong><br>
            • 📄 <strong>Crear documentos</strong><br>
            • 🧠 <strong>Recordar</strong> información sobre ti<br><br>
            ¿En qué puedo ayudarte?
        </div>
    </div>
    
    <div class="input-area">
        <div class="input-wrapper">
            <input type="text" id="input" placeholder="Escribe tu mensaje..." autocomplete="off">
        </div>
        <button class="btn btn-send" id="sendBtn" title="Enviar">➤</button>
        <button class="btn btn-mic" id="micBtn" title="Voz">🎤</button>
        <button class="btn btn-help" id="helpBtn" title="Limpiar chat">🗑️</button>
    </div>
</div>

<div class="modal" id="imageModal" onclick="closeModal()">
    <img id="modalImg" src="" alt="Imagen ampliada">
</div>

<script>
    const input = document.getElementById('input');
    const sendBtn = document.getElementById('sendBtn');
    const micBtn = document.getElementById('micBtn');
    const helpBtn = document.getElementById('helpBtn');
    const messagesDiv = document.getElementById('messages');
    const imageModal = document.getElementById('imageModal');
    const modalImg = document.getElementById('modalImg');
    
    let recognition = null;
    let isListening = false;
    let isProcessing = false;
    
    // Formato simple de markdown
    function formatMarkdown(text) {
        return text
            .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.+?)\*/g, '<em>$1</em>')
            .replace(/\[(.+?)\]\((.+?)\)/g, '<a href="$2" target="_blank">$1</a>')
            .replace(/\n/g, '<br>');
    }
    
    function addMessage(content, isUser, imageUrl = null, docUrl = null) {
        const div = document.createElement('div');
        div.className = `message ${isUser ? 'user' : 'bot'}`;
        
        if (isUser) {
            div.textContent = content;
        } else {
            div.innerHTML = formatMarkdown(content);
            
            if (imageUrl) {
                const img = document.createElement('img');
                img.src = imageUrl;
                img.alt = 'Imagen generada';
                img.onclick = () => openModal(imageUrl);
                div.appendChild(img);
            }
            
            if (docUrl) {
                const link = document.createElement('a');
                link.href = docUrl;
                link.className = 'doc-link';
                link.download = '';
                link.textContent = '📥 Descargar documento';
                div.appendChild(link);
            }
        }
        
        messagesDiv.appendChild(div);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
    }
    
    function openModal(src) {
        modalImg.src = src;
        imageModal.classList.add('active');
    }
    
    function closeModal() {
        imageModal.classList.remove('active');
    }
    
    async function sendMessage(text) {
        const msgText = text || input.value.trim();
        if (!msgText || isProcessing) return;
        
        isProcessing = true;
        sendBtn.disabled = true;
        input.value = '';
        
        addMessage(msgText, true);
        
        // Indicador de escritura
        const typingDiv = document.createElement('div');
        typingDiv.className = 'typing';
        typingDiv.id = 'typing';
        typingDiv.textContent = 'NYXIA está pensando';
        messagesDiv.appendChild(typingDiv);
        messagesDiv.scrollTop = messagesDiv.scrollHeight;
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({mensaje: msgText})
            });
            
            const data = await response.json();
            typingDiv.remove();
            
            addMessage(data.respuesta, false, data.imagen, data.documento);
            
        } catch(e) {
            typingDiv.remove();
            addMessage('❌ Error de conexión. Verifica que el servidor esté ejecutándose.', false);
        }
        
        isProcessing = false;
        sendBtn.disabled = false;
        input.focus();
    }
    
    function sendQuick(text) {
        sendMessage(text);
    }
    
    sendBtn.onclick = () => sendMessage();
    input.addEventListener('keypress', (e) => { 
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    
    // Botón limpiar chat
    helpBtn.onclick = () => {
        if (confirm('¿Limpiar el chat visible? (No borra los recuerdos)')) {
            messagesDiv.innerHTML = `
                <div class="message bot">
                    🌌 Chat limpiado. Tus recuerdos siguen guardados. ¿En qué puedo ayudarte?
                </div>`;
        }
    };
    
    // Reconocimiento de voz
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.lang = 'es-ES';
        recognition.continuous = false;
        recognition.interimResults = false;
        
        recognition.onstart = () => {
            isListening = true;
            micBtn.classList.add('listening');
        };
        
        recognition.onend = () => {
            isListening = false;
            micBtn.classList.remove('listening');
        };
        
        recognition.onresult = (event) => {
            const texto = event.results[0][0].transcript;
            input.value = texto;
            sendMessage();
        };
        
        recognition.onerror = (event) => {
            isListening = false;
            micBtn.classList.remove('listening');
            if (event.error !== 'no-speech') {
                console.error('Error de voz:', event.error);
            }
        };
        
        micBtn.onclick = () => {
            if (recognition && !isListening) {
                try { recognition.start(); } catch(e) {}
            } else if (isListening) {
                recognition.stop();
            }
        };
    } else {
        micBtn.disabled = true;
        micBtn.title = "Voz no soportada en este navegador";
    }
    
    // Atajos de teclado
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeModal();
    });
    
    // Focus inicial
    input.focus();
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
    """Endpoint principal de chat."""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'respuesta': '❌ Petición inválida.'}), 400
        
        mensaje = data.get('mensaje', '').strip()
        if not mensaje:
            return jsonify({'respuesta': '❌ No recibí mensaje.'}), 400
        
        if len(mensaje) > 2000:
            return jsonify({'respuesta': '❌ Mensaje demasiado largo (máximo 2000 caracteres).'}), 400
        
        memoria = cargar_memoria()
        resultado = procesar(mensaje, memoria)
        
        return jsonify({
            'respuesta': resultado['respuesta'],
            'imagen': resultado.get('imagen'),
            'documento': resultado.get('documento'),
            'tipo': resultado.get('tipo')
        })
        
    except Exception as e:
        logger.exception("Error en endpoint /chat")
        return jsonify({'respuesta': f'❌ Error interno: {str(e)}'}), 500

@app.route('/static/generated/<path:filename>')
def serve_generated(filename: str):
    """Sirve archivos generados estáticamente."""
    return send_from_directory(STATIC_DIR, filename)

@app.route('/health')
def health():
    """Endpoint de salud para monitoreo."""
    return jsonify({
        'status': 'ok',
        'groq_configured': bool(GROQ_API_KEY),
        'timestamp': datetime.now().isoformat()
    })

# ========== ERROR HANDLERS ==========
@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Recurso no encontrado'}), 404

@app.errorhandler(500)
def server_error(e):
    logger.exception("Error 500")
    return jsonify({'error': 'Error interno del servidor'}), 500

# ========== INICIO ==========
if __name__ == '__main__':
    logger.info(f"🚀 Iniciando NYXIA...")
    logger.info(f"   Groq API: {'Configurada ✓' if GROQ_API_KEY else 'No configurada (modo demo)'}")
    logger.info(f"   Directorio estático: {STATIC_DIR.absolute()}")
    
    app.run(
        host='0.0.0.0', 
        port=int(os.getenv('PORT', 7860)), 
        debug=os.getenv('DEBUG', 'false').lower() == 'true'
    )
