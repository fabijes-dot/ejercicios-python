#!/usr/bin/env python3
"""
contador.py — Conteo de personas en video (entradas / salidas / aforo)
                + PLUS: clasificacion por TAMANO (nino / adulto) y COLOR DE ROPA
=======================================================================

Lee un video (o webcam/RTSP) y, cuadro a cuadro:

  1. DETECTA personas. Tres motores, en orden de preferencia:
       yolo   -> ultralytics YOLO, si lo tienes instalado (el mas preciso)
       dnn    -> MobileNet-SSD via cv2.dnn, si le pasas los archivos del modelo
       hog    -> HOG + SVM, incluido en OpenCV, sin descargar nada (respaldo)
  2. SIGUE a cada persona con un rastreador por centroide (le asigna un ID).
  3. CUENTA los cruces de una LINEA que tu defines: quien la cruza en un sentido
     suma ENTRADA, en el otro SALIDA. El aforo es entradas - salidas.
  4. Registra en CSV cada cruce y una serie temporal de ocupacion.

  [NUEVO] 5. CLASIFICA a cada persona:
       - TAMANO: compara su altura con la de un ADULTO parado en ese mismo
                 punto del piso (corrige la perspectiva: lo lejano se ve chico).
                 Si mide menos de --umbral-nino (0.70) de un adulto -> NINO.
       - ROPA:   recorta la zona del torso, la pasa a HSV y decide el color
                 predominante (negro, blanco, gris, rojo, azul, verde...).
     Cada persona se clasifica con TODAS sus muestras (mediana de altura y
     "votacion" de color cuadro a cuadro), no con un solo cuadro: asi un
     cuadro malo no cambia el resultado.
     Se cuenta de dos maneras:
       - "cruce":  personas que cruzaron la linea (lo mas confiable)
       - "vistos": todas las personas rastreadas al menos --min-muestras cuadros

Todas las partes nuevas estan marcadas con  # [NUEVO]

Instalacion:
    pip install opencv-python numpy
    # opcional (mucho mejor deteccion):  pip install ultralytics

Uso:
    python contador.py entrada.mp4 --linea              # dibujar la linea con el mouse
    python contador.py entrada.mp4 --motor yolo
    python contador.py entrada.mp4 --umbral-nino 0.40 --umbral-negro 0.35
    python contador.py entrada.mp4 --headless --record salida.mp4
    python contador.py 0                                 # webcam

Teclas: q salir | espacio pausa | s captura PNG | r reinicia contadores
"""

# Permite escribir anotaciones de tipo modernas (list[Persona]) en Python 3.8+
from __future__ import annotations

import argparse                      # lee las opciones de la linea de comandos (--motor, --speed...)
import csv                           # escribe los archivos de resultados .csv
import json                          # lee/guarda la linea de conteo en linea.json
import os                            # revisa si existen archivos
import sys                           # sys.exit() para terminar con un mensaje de error
import time                          # reloj real, para reproducir a velocidad real
from collections import Counter, deque   # [NUEVO] Counter = diccionario que cuenta cosas
from datetime import datetime, timedelta   # fechas y formato h:mm:ss

import cv2                           # OpenCV: leer video, dibujar, mostrar ventanas
import numpy as np                   # NumPy: operaciones con vectores y matrices (las imagenes son matrices)

# --------------------------------------------------------------------------- #
# Paleta (BGR)
# --------------------------------------------------------------------------- #
# OJO: OpenCV usa el orden Azul-Verde-Rojo (BGR), no RGB.
# Por eso RED es (60, 60, 255): el 255 esta en la ultima posicion.
CYAN = (230, 220, 60)
GREEN = (120, 220, 0)
AMBER = (0, 190, 255)
RED = (60, 60, 255)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
DARK = (35, 28, 20)
MAGENTA = (200, 80, 255)             # [NUEVO] color del panel de perfil
# Colores que se reparten entre las personas para distinguirlas en pantalla
COLORES = [(230, 220, 60), (120, 220, 0), (0, 190, 255), (200, 120, 255),
           (255, 180, 90), (120, 255, 220), (180, 180, 255), (90, 230, 160)]


# --------------------------------------------------------------------------- #
# 1. Detectores de personas
# --------------------------------------------------------------------------- #
class Detector:
    """Devuelve una lista de cajas (x, y, w, h) de personas."""
    # Una "caja" es un rectangulo: (x, y) es la esquina superior izquierda,
    # w es el ancho y h el alto, todo en pixeles.

    def __init__(self, motor="auto", modelo=None, proto=None, conf_min=0.45):
        # conf_min: el detector da una "confianza" de 0 a 1 por cada caja.
        # Descartamos las que esten por debajo (0.45 = 45 %).
        self.conf_min = conf_min
        self.motor = None               # aun no sabemos que motor vamos a usar

        # --- Intento 1: YOLO (red neuronal moderna, la mas precisa) ---
        if motor in ("auto", "yolo"):
            try:
                # Se importa aqui adentro para que el programa funcione
                # aunque ultralytics no este instalado (se usa otro motor).
                from ultralytics import YOLO                      # noqa: PLC0415
                # yolov8n.pt = version "nano" de YOLOv8: la mas pequena y rapida.
                # Si no existe el archivo, ultralytics lo descarga solo (6 MB).
                self.yolo = YOLO(modelo or "yolov8n.pt")
                self.motor = "yolo"
            except Exception as exc:                              # noqa: BLE001
                # Si el usuario PIDIO yolo explicitamente y fallo, paramos.
                # Si era "auto", seguimos probando el siguiente motor.
                if motor == "yolo":
                    sys.exit(f"YOLO no disponible ({exc}).  pip install ultralytics")

        # --- Intento 2: MobileNet-SSD (red neuronal mas antigua, via OpenCV) ---
        if self.motor is None and motor in ("auto", "dnn"):
            # Necesita dos archivos: el .caffemodel (pesos) y el .prototxt (arquitectura)
            if modelo and proto and os.path.exists(modelo) and os.path.exists(proto):
                self.net = cv2.dnn.readNetFromCaffe(proto, modelo)
                self.motor = "dnn"
            elif motor == "dnn":
                sys.exit("Para --motor dnn pasa --modelo y --proto de MobileNet-SSD.")

        # --- Intento 3 (respaldo): HOG + SVM, viene dentro de OpenCV ---
        # HOG = Histograma de Gradientes Orientados: describe la silueta por
        # la direccion de los bordes. Un SVM ya entrenado decide si es persona.
        if self.motor is None:
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            self.motor = "hog"

        print(f"[detector] motor: {self.motor}")
        if self.motor == "hog":
            print("           (HOG es el respaldo: rapido pero impreciso con gente "
                  "muy junta.\n            Para produccion instala:  pip install ultralytics)")

    def detectar(self, frame):
        # Punto de entrada unico: el resto del programa llama detectar()
        # sin importar que motor hay por dentro.
        if self.motor == "yolo":
            return self._yolo(frame)
        if self.motor == "dnn":
            return self._dnn(frame)
        return self._hog(frame)

    def _yolo(self, frame):
        # classes=[0]: YOLO conoce 80 clases (auto, perro, silla...).
        # La clase 0 es "person", asi que solo pedimos personas.
        res = self.yolo(frame, classes=[0], conf=self.conf_min, verbose=False)
        cajas = []
        for r in res:
            # xyxy = cada caja como (x1, y1, x2, y2): esquina sup. izq. e inf. der.
            # .cpu().numpy() la pasa de tensor de PyTorch a arreglo de NumPy.
            for b in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, b[:4])
                # Convertimos al formato (x, y, ancho, alto) que usa el resto del codigo
                cajas.append((x1, y1, x2 - x1, y2 - y1))
        return cajas

    def _dnn(self, frame):
        H, W = frame.shape[:2]          # alto y ancho de la imagen original
        # La red espera una imagen de 300x300 normalizada:
        # 0.007843 = 1/127.5 y 127.5 es el valor que se resta (pixeles de -1 a 1).
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843,
                                     (300, 300), 127.5)
        self.net.setInput(blob)
        det = self.net.forward()        # corre la red; devuelve todas las detecciones
        cajas = []
        for i in range(det.shape[2]):
            conf = float(det[0, 0, i, 2])            # confianza de la deteccion i
            if conf < self.conf_min or int(det[0, 0, i, 1]) != 15:   # 15 = person
                continue
            # Las coordenadas vienen de 0 a 1; se multiplican por el tamano real
            x1, y1, x2, y2 = (det[0, 0, i, 3:7] * np.array([W, H, W, H])).astype(int)
            cajas.append((x1, y1, x2 - x1, y2 - y1))
        return cajas

    def _hog(self, frame):
        # HOG es lento en imagenes grandes: reducimos a 640 px de ancho como maximo
        escala = 640 / max(frame.shape[1], 1)
        chico = cv2.resize(frame, None, fx=escala, fy=escala) if escala < 1 else frame
        # winStride: cuanto se desplaza la ventana de busqueda (mayor = mas rapido, menos preciso)
        # scale=1.05: busca personas de distintos tamanos agrandando 5 % cada vez
        rects, pesos = self.hog.detectMultiScale(chico, winStride=(8, 8),
                                                 padding=(8, 8), scale=1.05)
        # Devolvemos las cajas al tamano de la imagen original
        inv = 1 / escala if escala < 1 else 1.0
        # "pesos" es la confianza del SVM: descartamos las debiles (< 0.3)
        cajas = [tuple((np.array(r) * inv).astype(int))
                 for r, p in zip(rects, pesos) if p >= 0.3]
        # HOG suele dar varias cajas encimadas sobre la misma persona: las limpiamos
        return nms(cajas, 0.4)


def nms(cajas, thr=0.4):
    """Supresion de no-maximos por area (evita contar 3 veces a la misma persona)."""
    if not cajas:
        return []
    # Ordenamos de la caja mas grande a la mas pequena
    cajas = sorted(cajas, key=lambda b: b[2] * b[3], reverse=True)
    keep = []
    for b in cajas:
        # Nos quedamos con la caja solo si no se solapa mucho con ninguna ya elegida
        if all(iou(b, k) < thr for k in keep):
            keep.append(b)
    return keep


def iou(a, b):
    # IoU = Interseccion sobre Union: cuanto se solapan dos cajas.
    # 0 = no se tocan, 1 = son identicas.
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    # Rectangulo de la interseccion (la zona que comparten)
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)      # 0 si no se solapan
    union = aw * ah + bw * bh - inter             # area total cubierta por ambas
    return inter / union if union else 0.0


# --------------------------------------------------------------------------- #
# 2. Rastreo por centroide
# --------------------------------------------------------------------------- #
# El detector solo dice "hay personas aqui" en CADA cuadro, sin saber quien es
# quien. El rastreador une las detecciones de cuadros seguidos para decir
# "esta caja es la misma persona #7 del cuadro anterior".
class Persona:
    _next_id = 1                        # contador compartido: cada persona nueva recibe el siguiente ID

    def __init__(self, caja, t):
        self.id = Persona._next_id
        Persona._next_id += 1
        self.caja = caja                # ultima caja donde se la vio
        self.t_inicio = self.t_visto = t
        self.perdido = 0                # cuantos cuadros seguidos lleva sin ser detectada
        self.rastro: deque = deque(maxlen=48)   # ultimas 48 posiciones del centro (su "estela")
        self.rastro.append(self.centro)
        self.contada = False          # ya cruzo la linea una vez
        self.color = COLORES[self.id % len(COLORES)]
        # [NUEVO] Memoria para clasificarla:
        self.alturas: deque = deque(maxlen=90)  # (y_pies, altura) en cada cuadro limpio
        self.votos_color: Counter = Counter()   # {"negro": 12, "azul": 3, ...}
        self.muestras = 0                       # en cuantos cuadros la observamos

    @property
    def centro(self):
        # Centro de la caja: se usa como "la posicion" de la persona
        x, y, w, h = self.caja
        return x + w / 2, y + h / 2

    def mover(self, caja, t):
        # La persona fue vista otra vez: actualizamos su caja y reiniciamos "perdido"
        self.caja, self.perdido, self.t_visto = caja, 0, t
        self.rastro.append(self.centro)


class Rastreador:
    """Asocia detecciones con personas ya vistas por cercania del centroide."""

    def __init__(self, dist_max=110, max_perdido=25):
        # dist_max: distancia maxima (px) que una persona puede moverse entre cuadros.
        #           Si la deteccion esta mas lejos, se considera otra persona.
        # max_perdido: cuadros sin verla antes de darla por ida (25 cuadros = ~1 s).
        #              Esto evita perder a alguien que queda tapado un momento.
        self.dist_max, self.max_perdido = dist_max, max_perdido
        self.personas: list[Persona] = []

    def actualizar(self, cajas, t):
        # Al empezar, suponemos que nadie fue visto en este cuadro
        for p in self.personas:
            p.perdido += 1

        libres = list(self.personas)    # personas que aun no tienen caja asignada en este cuadro
        for caja in cajas:
            cx, cy = caja[0] + caja[2] / 2, caja[1] + caja[3] / 2
            # Buscamos la persona libre cuyo centro este mas cerca de esta caja
            mejor, mejor_d = None, self.dist_max
            for p in libres:
                px, py = p.centro
                d = float(np.hypot(cx - px, cy - py))   # distancia en linea recta (Pitagoras)
                if d < mejor_d:
                    mejor, mejor_d = p, d
            if mejor is None:
                # Nadie estaba lo bastante cerca: es una persona nueva
                self.personas.append(Persona(caja, t))
            else:
                # Es la misma persona que ya conociamos: la movemos
                mejor.mover(caja, t)
                libres.remove(mejor)    # ya no puede emparejarse con otra caja

        # Quitamos a quienes llevan demasiado tiempo sin ser vistos
        salieron = [p for p in self.personas if p.perdido > self.max_perdido]
        self.personas = [p for p in self.personas if p.perdido <= self.max_perdido]
        return salieron

    def visibles(self):
        # Solo las personas detectadas en ESTE cuadro
        return [p for p in self.personas if p.perdido == 0]


# --------------------------------------------------------------------------- #
# [NUEVO] 2b. Clasificacion: tamano y color de ropa
# --------------------------------------------------------------------------- #
# Rangos de TONO (Hue) en OpenCV. OJO: en OpenCV el tono va de 0 a 179
# (no de 0 a 360), asi que cada valor es "grados / 2".
RANGOS_TONO = [
    ("rojo", 0, 10), ("naranja", 10, 22), ("amarillo", 22, 35),
    ("verde", 35, 85), ("azul", 85, 130), ("morado", 130, 160),
    ("rojo", 160, 180),                 # el rojo "da la vuelta" al final del circulo
]
V_NEGRO = 60      # brillo (V) por debajo de esto = pixel negro/muy oscuro (0..255)
V_BLANCO = 190    # brillo por encima de esto y poca saturacion = blanco
S_GRIS = 50       # saturacion por debajo de esto = sin color (gris)


def color_ropa(frame, caja, umbral_negro=0.40):
    """Devuelve el nombre del color predominante del TORSO de la persona.

    Como funciona:
      1. Recorta el torso: del 20 % al 55 % del alto de la caja y la franja
         central (25 % a 75 % del ancho), para no agarrar fondo ni cabeza.
      2. Pasa el recorte a HSV: H = tono (que color), S = saturacion (que tan
         vivo), V = brillo. En HSV es mucho mas facil separar colores que en BGR.
      3. Clasifica cada pixel: negro (V bajo), blanco (V alto y S baja),
         gris (S baja) o un color segun su tono H.
      4. Si al menos 'umbral_negro' de los pixeles son negros -> "negro".
         Si no, gana el color con mas pixeles.
    """
    x, y, w, h = caja
    H, W = frame.shape[:2]
    x1, x2 = max(int(x + w * 0.25), 0), min(int(x + w * 0.75), W)
    y1, y2 = max(int(y + h * 0.20), 0), min(int(y + h * 0.55), H)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None                     # recorte muy chico: no se puede opinar
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    tono, sat, brillo = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    # Mascaras booleanas: True en los pixeles que cumplen la condicion
    negro = brillo < V_NEGRO
    blanco = (brillo > V_BLANCO) & (sat < S_GRIS) & ~negro
    gris = (sat < S_GRIS) & ~negro & ~blanco
    con_color = ~(negro | blanco | gris)

    total = tono.size
    if negro.sum() / total >= umbral_negro:
        return "negro"

    conteo = Counter({"blanco": int(blanco.sum()), "gris": int(gris.sum())})
    tonos = tono[con_color]
    for nombre, a, b in RANGOS_TONO:
        conteo[nombre] += int(((tonos >= a) & (tonos < b)).sum())
    conteo["negro"] = int(negro.sum())  # por si igual es el mayor
    return conteo.most_common(1)[0][0]


class Clasificador:
    """Junta las observaciones de cada persona y lleva los contadores del perfil.

    TAMANO CON PERSPECTIVA:
      Una persona lejos de la camara se ve mas chica y aparece mas ARRIBA en
      la imagen (sus pies estan mas arriba). Por eso no sirve comparar alturas
      "en bruto". En su lugar:
        1. Guardamos de TODAS las personas el par (posicion de los pies, altura).
        2. Ajustamos una recta:  altura_adulto = a * y_pies + b
           usando el percentil 75 (la gente alta, o sea adultos) en cada franja.
        3. Para cada persona calculamos  razon = su altura / altura_adulto
           esperada en ESE punto del piso.  ~1.0 = adulto, < umbral = NINO.
    """

    def __init__(self, umbral_nino=0.70, umbral_negro=0.40, min_muestras=15):
        # umbral_nino: razon altura / altura de adulto en ese lugar.
        #              0.70 = mide menos del 70 % de un adulto -> NINO.
        # umbral_negro: fraccion del torso que debe ser oscura para decir "negro".
        # min_muestras: cuadros minimos para contar a alguien en "vistos"
        #               (filtra detecciones fantasma y IDs que se cortan).
        self.umbral_nino = umbral_nino
        self.umbral_negro = umbral_negro
        self.min_muestras = min_muestras
        self.escena: deque = deque(maxlen=6000)   # (y_pies, altura) de todos
        self.modelo = None                          # (a, b) de la recta
        self.rango = (0.0, 1.0)                     # zona del piso con datos
        self._nuevas = 0
        self.reiniciar()

    def reiniciar(self):
        self.cruce_tam = Counter()      # {"NINO": 3, "ADULTO": 5} de quienes cruzaron
        self.cruce_color = Counter()    # {"negro": 2, "azul": 4, ...}
        self.visto_tam = Counter()      # igual, pero de todas las personas rastreadas
        self.visto_color = Counter()
        self.registro: list[tuple] = []  # una fila por persona vista (para el CSV)

    @staticmethod
    def _tapada(caja, otras, max_solape=0.25):
        """True si otra caja tapa mas del 25 % de esta (su altura no es confiable)."""
        x, y, w, h = caja
        area = max(w * h, 1)
        for o in otras:
            if tuple(o) == tuple(caja):
                continue
            ox, oy, ow, oh = o
            iw = max(0, min(x + w, ox + ow) - max(x, ox))
            ih = max(0, min(y + h, oy + oh) - max(y, oy))
            if iw * ih / area > max_solape:
                return True
        return False

    def observar(self, p, frame, cajas):
        """Se llama en cada cuadro para cada persona visible."""
        x, y, w, h = p.caja
        H = frame.shape[0]
        p.muestras += 1
        # La altura solo se usa si la medida es "limpia":
        #  - la caja no toca el borde de arriba/abajo (persona cortada)
        #  - no es muy ancha (w/h > 0.6 = sentada, agachada o 2 personas juntas)
        #  - nadie la tapa
        limpia = (y > 2 and y + h < H - 2 and w / max(h, 1) <= 0.6
                  and not self._tapada(p.caja, cajas))
        if limpia:
            muestra = ((y + h) / H, h / H)      # (y de los pies, altura) en 0..1
            p.alturas.append(muestra)
            self.escena.append(muestra)
            self._nuevas += 1
            if self._nuevas >= 50:              # recalculamos la recta cada 50 muestras
                self._nuevas = 0
                self._ajustar()
        c = color_ropa(frame, p.caja, self.umbral_negro)
        if c:
            p.votos_color[c] += 1       # cada cuadro "vota" por un color

    def _ajustar(self):
        """Ajusta la recta  altura_adulto = a * y_pies + b."""
        if len(self.escena) < 150:
            return                      # aun muy pocos datos
        arr = np.array(self.escena)
        # Partimos la escena en 8 franjas horizontales con la misma cantidad de datos
        cortes = np.quantile(arr[:, 0], np.linspace(0, 1, 9))
        xs, ys = [], []
        for lo, hi in zip(cortes[:-1], cortes[1:]):
            sel = arr[(arr[:, 0] >= lo) & (arr[:, 0] <= hi)]
            if len(sel) >= 10:
                xs.append(float(np.median(sel[:, 0])))
                ys.append(float(np.percentile(sel[:, 1], 75)))  # "adulto tipico" de la franja
        if len(xs) >= 3:
            a, b = np.polyfit(xs, ys, 1)            # recta que mejor pasa por esos puntos
            self.modelo = (float(a), float(b))
            # Zona del piso donde SI hay datos: fuera de ella la recta no es confiable
            self.rango = (min(xs), max(xs))

    def _esperada(self, y_pies):
        if self.modelo is None:
            return None
        a, b = self.modelo
        lo, _ = self.rango
        # Solo limitamos hacia el FONDO (lejos), donde la recta se va a cero y
        # la razon se dispara. Hacia ADELANTE (cerca de la camara) la perspectiva
        # sigue siendo lineal, asi que la recta se puede extender sin problema.
        y_pies = max(y_pies, lo)
        return max(a * y_pies + b, 0.05)

    def altura_rel(self, p):
        """Razon altura / adulto esperado (mediana de todas sus muestras)."""
        if not p.alturas or self.modelo is None:
            return 0.0
        razones = [h / self._esperada(yp) for yp, h in p.alturas]
        return float(np.median(razones))

    def tamano(self, p):
        if len(p.alturas) < 3 or self.modelo is None:
            return "?"                  # aun no tenemos datos suficientes
        return "NINO" if self.altura_rel(p) < self.umbral_nino else "ADULTO"

    def color(self, p):
        return p.votos_color.most_common(1)[0][0] if p.votos_color else "?"

    def registrar_cruce(self, p):
        # Al momento de cruzar puede que aun no haya datos ("?").
        # Esto es solo PROVISIONAL (para el aviso en pantalla y conteo.csv):
        # el conteo definitivo del perfil se hace cuando la persona se va,
        # con todas sus muestras (ver registrar_visto).
        return self.tamano(p), self.color(p)

    def registrar_visto(self, p):
        """Se llama cuando la persona desaparece (o al final del video)."""
        tam, col = self.tamano(p), self.color(p)
        if p.contada:                   # cruzo la linea: va al perfil de "cruce"
            self.cruce_tam[tam] += 1
            self.cruce_color[col] += 1
        if p.muestras < self.min_muestras:
            return None                 # muy poco tiempo en pantalla: probable fantasma
        self.visto_tam[tam] += 1
        self.visto_color[col] += 1
        fila = (p.id, round(self.altura_rel(p), 3), tam, col, p.muestras,
                round(p.t_inicio, 2), round(p.t_visto, 2), p.contada)
        self.registro.append(fila)
        return fila


# --------------------------------------------------------------------------- #
# 3. Linea de conteo
# --------------------------------------------------------------------------- #
class LineaConteo:
    """Cuenta cruces con signo: de lado negativo a positivo = ENTRADA."""

    def __init__(self, p1, p2, invertir=False):
        # p1 y p2: los dos extremos de la linea, en pixeles
        self.p1 = np.array(p1, dtype=np.float64)
        self.p2 = np.array(p2, dtype=np.float64)
        self.invertir = invertir        # True = intercambia que sentido es entrada y cual salida
        self.entradas = 0
        self.salidas = 0
        self.eventos: list[tuple] = []   # (segundo, tipo, id, tamano, color)

    def lado(self, punto):
        """Signo del producto cruz: de que lado de la linea cae el punto."""
        # v = vector a lo largo de la linea; w = vector desde p1 hasta el punto.
        # El producto cruz v x w es positivo de un lado de la linea y negativo del otro.
        v = self.p2 - self.p1
        w = np.array(punto, dtype=np.float64) - self.p1
        c = v[0] * w[1] - v[1] * w[0]
        # Devuelve -1, 0 (justo encima de la linea) o +1
        return 0 if abs(c) < 1e-9 else (1 if c > 0 else -1)

    def revisar(self, persona, t, min_rastro=4):
        """Compara el lado actual contra el de hace unos cuadros."""
        # Si ya la contamos, o aun no tiene suficiente historia, no hacemos nada
        if persona.contada or len(persona.rastro) < min_rastro:
            return None
        # Lado de la linea hace 4 cuadros y lado ahora.
        # Comparar contra hace varios cuadros (y no el anterior) evita
        # contar el "temblor" de una caja que baila justo sobre la linea.
        antes = self.lado(persona.rastro[-min_rastro])
        ahora = self.lado(persona.rastro[-1])
        if antes == 0 or ahora == 0 or antes == ahora:
            return None                # no cambio de lado: no hubo cruce
        if not self._cerca(persona.rastro[-1]):
            return None            # cruzo la recta infinita, pero fuera del segmento
        # Paso de negativo a positivo = entrada (a menos que se haya pedido invertir).
        # "!= self.invertir" funciona como un XOR: si invertir es True, da vuelta el resultado.
        entra = (antes < 0 and ahora > 0) != self.invertir
        persona.contada = True         # cada persona se cuenta UNA sola vez
        if entra:
            self.entradas += 1
        else:
            self.salidas += 1
        tipo = "ENTRADA" if entra else "SALIDA"
        return tipo

    def _cerca(self, punto, margen=1.35):
        """El cruce vale solo si ocurre sobre el segmento dibujado, no fuera."""
        v = self.p2 - self.p1
        largo2 = float(v @ v)              # largo de la linea al cuadrado (@ = producto punto)
        if largo2 < 1e-9:
            return False                   # linea de largo cero: no sirve
        w = np.array(punto, dtype=np.float64) - self.p1
        s = float(v @ w) / largo2          # proyeccion normalizada 0..1
        # s = 0 en p1, s = 1 en p2. Damos un margen extra a cada lado (17.5 %)
        holgura = (margen - 1.0) / 2
        return -holgura <= s <= 1 + holgura

    @property
    def aforo(self):
        # Cuantas personas hay "adentro" ahora mismo
        return self.entradas - self.salidas


def pedir_linea(frame, ruta_cfg):
    """Dos clics definen la linea de conteo. Se guarda para la proxima vez."""
    pts = []                            # aqui se guardan los clics

    def on_mouse(event, x, y, flags, _):
        # OpenCV llama a esta funcion cada vez que se usa el mouse en la ventana
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append((x, y))          # clic izquierdo: agrega un punto
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()                   # clic derecho: borra el ultimo punto

    cv2.namedWindow("linea", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("linea", on_mouse)
    print("\n2 clics para la linea de conteo (puerta, pasillo, torniquete).")
    print("Clic derecho deshace. ENTER acepta. ESC cancela.\n")
    while True:
        # Redibujamos en cada vuelta sobre una copia del primer cuadro
        vis = frame.copy()
        for p in pts:
            cv2.circle(vis, p, 6, CYAN, -1)          # -1 = circulo relleno
        if len(pts) == 2:
            cv2.line(vis, pts[0], pts[1], CYAN, 3)
        cv2.putText(vis, "LINEA DE CONTEO: 2 clics + ENTER", (14, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, CYAN, 2)
        cv2.imshow("linea", vis)
        # waitKey espera 20 ms una tecla; & 0xFF se queda con el codigo de la tecla
        k = cv2.waitKey(20) & 0xFF
        if k == 27:                     # 27 = ESC
            cv2.destroyAllWindows()
            return None
        if k in (13, 10) and len(pts) == 2:          # 13 o 10 = ENTER
            cv2.destroyAllWindows()
            # Guardamos la linea para no tener que dibujarla la proxima vez
            with open(ruta_cfg, "w", encoding="utf-8") as f:
                json.dump({"linea": [list(pts[0]), list(pts[1])]}, f, indent=2)
            print(f"Linea guardada en {ruta_cfg}")
            return pts


# --------------------------------------------------------------------------- #
# 4. HUD
# --------------------------------------------------------------------------- #
# HUD = "Head-Up Display": los paneles de informacion dibujados sobre el video.
def panel(img, x, y, w, h, titulo, filas, accent=CYAN):
    # Dibuja un recuadro semitransparente con un titulo y varias filas de texto
    H, W = img.shape[:2]
    # Recortamos el panel para que no se salga de la imagen
    x, y = max(x, 0), max(y, 0)
    w, h = min(w, W - x), min(h, H - y)
    if w <= 0 or h <= 0:
        return
    sub = img[y:y + h, x:x + w]         # region de la imagen donde va el panel
    # Mezcla 65 % de color oscuro con 35 % del video: efecto "vidrio oscuro"
    cv2.addWeighted(np.full(sub.shape, DARK, np.uint8), 0.65, sub, 0.35, 0, sub)
    cv2.rectangle(img, (x, y), (x + w, y + h), accent, 1)        # borde
    cv2.rectangle(img, (x, y), (x + w, y + 22), accent, -1)      # barra del titulo
    cv2.putText(img, titulo, (x + 8, y + 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, DARK, 1, cv2.LINE_AA)
    for i, fila in enumerate(filas):
        yy = y + 44 + i * 26            # cada fila 26 px mas abajo
        if yy > y + h - 4:
            break                       # no cabe mas texto en el panel
        cv2.putText(img, str(fila), (x + 8, yy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.58, WHITE, 1, cv2.LINE_AA)


def dibujar(frame, personas, linea, t_video, progress, serie, aforo_max, clasif):
    # Devuelve una COPIA del cuadro con todo dibujado encima (no modifica el original)
    out = frame.copy()
    H, W = out.shape[:2]

    if linea is not None:
        # La linea de conteo
        cv2.line(out, tuple(map(int, linea.p1)), tuple(map(int, linea.p2)),
                 CYAN, 3, cv2.LINE_AA)
        # flecha que indica cual sentido cuenta como ENTRADA
        medio = ((linea.p1 + linea.p2) / 2).astype(int)
        v = linea.p2 - linea.p1
        # n = vector perpendicular a la linea (girar v 90 grados)
        n = np.array([-v[1], v[0]], dtype=np.float64)
        # Lo normalizamos a largo 40 px, apuntando al lado de "entrada"
        n = n / (np.linalg.norm(n) + 1e-9) * (-40 if linea.invertir else 40)
        cv2.arrowedLine(out, tuple(medio), tuple((medio + n).astype(int)),
                        GREEN, 3, tipLength=0.35)
        cv2.putText(out, "IN", tuple((medio + n * 1.35).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, GREEN, 2, cv2.LINE_AA)

    # Cada persona: su caja, su centro, su estela y su numero
    for p in personas:
        x, y, w, h = p.caja
        cv2.rectangle(out, (x, y), (x + w, y + h), p.color, 2)
        cx, cy = map(int, p.centro)
        cv2.circle(out, (cx, cy), 4, p.color, -1)
        if len(p.rastro) > 1:
            # polylines necesita los puntos con forma (N, 1, 2) y tipo int32
            pts = np.array(p.rastro, np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], False, p.color, 2, cv2.LINE_AA)
        # [NUEVO] Etiqueta: "#3 NINO negro .31 OK"
        #   0.62 = su altura / la de un adulto en ese lugar (1.0 = adulto tipico)
        #   OK  = ya fue contada al cruzar la linea
        etq = (f"#{p.id} {clasif.tamano(p)} {clasif.color(p)} "
               f"{clasif.altura_rel(p):.2f}"
               + (" OK" if p.contada else ""))
        cv2.putText(out, etq, (x, max(y - 8, 14)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, p.color, 2, cv2.LINE_AA)
        # [NUEVO] Marco del torso que se usa para leer el color de la ropa
        cv2.rectangle(out, (int(x + w * 0.25), int(y + h * 0.20)),
                      (int(x + w * 0.75), int(y + h * 0.55)), WHITE, 1)

    # Panel principal con los contadores
    e = linea.entradas if linea else 0
    s = linea.salidas if linea else 0
    a = linea.aforo if linea else len(personas)
    panel(out, 14, 14, 232, 140, "CONTEO DE PERSONAS", [
        f"Entradas : {e}",
        f"Salidas  : {s}",
        f"Aforo    : {a}",
    ], accent=GREEN if a <= aforo_max else RED)   # rojo si se supera el aforo

    # Panel de la esquina derecha: personas visibles y tiempo del video
    panel(out, W - 254, 14, 240, 100, "SESION", [
        f"Visibles: {len(personas)}",
        f"T: {timedelta(seconds=int(t_video))}",
    ], accent=CYAN)

    # [NUEVO] Panel de perfil: "cruce / vistos"
    ct, vt = clasif.cruce_tam, clasif.visto_tam
    cc, vc = clasif.cruce_color, clasif.visto_color
    top = cc.most_common(1) or vc.most_common(1)
    panel(out, W - 254, 124, 240, 150, "PERFIL  (cruce / vistos)", [
        f"Ninos     : {ct['NINO']} / {vt['NINO']}",
        f"Adultos   : {ct['ADULTO']} / {vt['ADULTO']}",
        f"Ropa negra: {cc['negro']} / {vc['negro']}",
        f"Color top : {top[0][0] if top else '-'}",
    ], accent=MAGENTA)

    # curva de ocupacion
    if len(serie) > 2:
        x0, y0, wg, hg = 14, H - 84, 232, 56          # posicion y tamano del grafico
        cv2.rectangle(out, (x0, y0), (x0 + wg, y0 + hg), (90, 90, 90), 1)
        # Tomamos los ultimos valores (uno por pixel de ancho del grafico)
        vals = np.array([v for _, v in serie][-wg:], dtype=np.float32)
        top_v = max(float(vals.max()), 1.0)            # valor maximo = parte de arriba del grafico
        # Convertimos cada valor a coordenadas de pixel (y crece hacia abajo en imagenes)
        xs = np.linspace(x0, x0 + wg, len(vals)).astype(np.int32)
        ys = (y0 + hg - vals / top_v * (hg - 4)).astype(np.int32)
        cv2.polylines(out, [np.stack([xs, ys], 1)], False, CYAN, 1, cv2.LINE_AA)
        cv2.putText(out, f"ocupacion (max {int(top_v)})", (x0 + 4, y0 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GREY, 1, cv2.LINE_AA)

    # Alerta visual: marco rojo si hay mas gente que el aforo permitido
    if aforo_max < 9999 and a > aforo_max:
        cv2.rectangle(out, (0, 0), (W - 1, H - 1), RED, 6)
        cv2.putText(out, "AFORO EXCEDIDO", (W // 2 - 130, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, RED, 3, cv2.LINE_AA)

    # Barra de progreso en el borde inferior
    if progress is not None:
        cv2.rectangle(out, (0, H - 5), (int(W * progress), H - 1), CYAN, -1)
    return out


# --------------------------------------------------------------------------- #
# 5. Programa principal
# --------------------------------------------------------------------------- #
def main() -> None:
    # ---- Opciones de la linea de comandos (ver: python contador.py --help) ---- #
    ap = argparse.ArgumentParser(description="Conteo de personas en video")
    ap.add_argument("video", help="archivo de video, indice de webcam (0) o URL RTSP")
    ap.add_argument("--motor", default="auto", choices=["auto", "yolo", "dnn", "hog"])
    ap.add_argument("--modelo", default=None, help="pesos YOLO (.pt) o caffemodel")
    ap.add_argument("--proto", default=None, help="prototxt de MobileNet-SSD")
    ap.add_argument("--conf", type=float, default=0.45, help="confianza minima")
    ap.add_argument("--config", default="linea.json")
    ap.add_argument("--linea", action="store_true", help="dibujar la linea con el mouse")
    ap.add_argument("--invertir", action="store_true", help="invertir sentido entrada/salida")
    ap.add_argument("--aforo-max", type=int, default=9999, help="alerta al superarlo")
    ap.add_argument("--csv", default="conteo.csv")
    ap.add_argument("--serie-csv", default=None, help="CSV de ocupacion en el tiempo")
    ap.add_argument("--record", default=None, help="mp4 de salida con el HUD")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--stride", type=int, default=1, help="procesa 1 de cada N cuadros")
    ap.add_argument("--speed", type=float, default=0.0, help="0 = a tope, 1 = tiempo real")
    # [NUEVO] opciones del perfil
    ap.add_argument("--umbral-nino", type=float, default=0.70,
                    help="altura / altura de adulto en ese punto; menos = NINO")
    ap.add_argument("--umbral-negro", type=float, default=0.40,
                    help="fraccion del torso oscura para decir ropa negra")
    ap.add_argument("--min-muestras", type=int, default=15,
                    help="cuadros minimos para contar a alguien en 'vistos'")
    ap.add_argument("--personas-csv", default="personas.csv",
                    help="CSV con una fila por persona vista (tamano, color)")
    args = ap.parse_args()

    # ---- Abrir el video ---- #
    # Si el argumento es un numero ("0"), es una webcam; si no, un archivo o URL
    src = int(args.video) if args.video.isdigit() else args.video
    if isinstance(src, str) and not os.path.exists(src) and "://" not in src:
        sys.exit(f"No existe el archivo: {src}")
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        sys.exit(f"OpenCV no pudo abrir: {args.video}")

    # Datos del video: cuadros por segundo, cantidad de cuadros y duracion
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0      # si no lo sabe (webcam), asume 25
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    dur = n_frames / fps_in if n_frames else 0.0
    if args.start > 0:
        # Saltar al segundo pedido con --start (OpenCV trabaja en milisegundos)
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    # Leemos el primer cuadro para conocer el tamano de la imagen
    ok, frame = cap.read()
    if not ok:
        sys.exit("No pude leer el primer cuadro.")
    H, W = frame.shape[:2]
    print(f"[video] {args.video}  {W}x{H}  {fps_in:.2f} FPS  "
          f"{n_frames} cuadros  {timedelta(seconds=int(dur))}")

    # ---- linea de conteo: mouse, archivo guardado, o media pantalla ---- #
    pts = None
    if args.linea and not args.headless:
        pts = pedir_linea(frame, args.config)          # 1) el usuario la dibuja
    if pts is None and os.path.exists(args.config):
        with open(args.config, encoding="utf-8") as f:  # 2) se lee de linea.json
            pts = [tuple(p) for p in json.load(f)["linea"]]
        print(f"[config] linea desde {args.config}")
    if pts is None:
        pts = [(0, H // 2), (W, H // 2)]                # 3) horizontal a media altura
        print("[aviso] sin linea definida: uso una horizontal a media pantalla.\n"
              "        Corre con --linea para dibujar la tuya.")
    linea = LineaConteo(pts[0], pts[1], invertir=args.invertir)

    # Creamos las piezas del sistema: detector, rastreador, clasificador (y la linea, arriba)
    det = Detector(args.motor, args.modelo, args.proto, args.conf)
    rastreador = Rastreador()
    clasif = Clasificador(args.umbral_nino, args.umbral_negro, args.min_muestras)  # [NUEVO]

    # ---- Grabacion opcional del video con las detecciones (--record) ---- #
    writer = None
    if args.record:
        # "mp4v" = codec MPEG-4. Si procesamos 1 de cada N cuadros, bajamos los FPS
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps_in / max(args.stride, 1), (W, H))
        if not writer.isOpened():
            print("[aviso] no pude abrir el VideoWriter; sigo sin grabar.")
            writer = None

    # ---- CSV de eventos: una fila por cada ENTRADA o SALIDA ---- #
    # OJO: se abre en modo "a" (append = agregar al final). Cada ejecucion
    # AGREGA filas al archivo existente; no lo borra. Solo se escribe el
    # encabezado si el archivo es nuevo.
    # [NUEVO] Ahora tiene 2 columnas mas (tamano, color): si tienes un
    # conteo.csv de la version anterior, BORRALO antes de correr esta.
    nuevo = not os.path.exists(args.csv)
    csv_f = open(args.csv, "a", newline="", encoding="utf-8")
    csv_w = csv.writer(csv_f)
    if nuevo:
        csv_w.writerow(["t_video_s", "timestamp", "evento", "persona_id",
                        "tamano", "color_ropa",
                        "entradas", "salidas", "aforo"])

    # [NUEVO] CSV de personas: se sobrescribe ("w") en cada ejecucion
    pers_f = open(args.personas_csv, "w", newline="", encoding="utf-8")
    pers_w = csv.writer(pers_f)
    pers_w.writerow(["persona_id", "altura_rel", "tamano", "color_ropa",
                     "cuadros", "t_inicio_s", "t_fin_s", "cruzo_linea"])

    def registrar_visto(p):
        # [NUEVO] Cuenta a la persona en "vistos" y la escribe en personas.csv
        fila = clasif.registrar_visto(p)
        if fila:
            pers_w.writerow(fila)
            pers_f.flush()

    # ---- CSV opcional de ocupacion: una fila por segundo (--serie-csv) ---- #
    serie_f = serie_w = None
    if args.serie_csv:
        nuevo_s = not os.path.exists(args.serie_csv)
        serie_f = open(args.serie_csv, "a", newline="", encoding="utf-8")
        serie_w = csv.writer(serie_f)
        if nuevo_s:
            serie_w.writerow(["t_video_s", "timestamp", "visibles", "aforo"])

    # ---- Variables del bucle principal ---- #
    serie: deque = deque(maxlen=2000)    # historial (tiempo, personas visibles) para el grafico
    vis = frame.copy()                   # la imagen que se muestra en pantalla
    idx = processed = 0                  # cuadros leidos y cuadros realmente procesados
    ultimo_log = -1e9                    # ultimo segundo escrito en el CSV de ocupacion
    paused = False
    t0_wall = time.time()                # hora real de inicio (para --speed)
    t_video = args.start                 # segundo actual dentro del video

    print("Procesando...  (q = salir)")
    try:
        # ================= BUCLE PRINCIPAL: una vuelta por cuadro ================= #
        while True:
            if not paused:
                # El primer cuadro ya se leyo arriba; desde el segundo leemos aqui
                if idx > 0:
                    ok, frame = cap.read()
                    if not ok:
                        print("\nFin del video.")
                        break
                idx += 1
                # Tiempo actual del video en segundos
                pos = cap.get(cv2.CAP_PROP_POS_MSEC)
                t_video = pos / 1000.0 if pos and pos > 0 else args.start + idx / fps_in
                if args.end is not None and t_video > args.end:
                    print("\nLlegue al segundo final pedido.")
                    break
                # --stride N: saltamos cuadros para ir mas rapido
                if args.stride > 1 and (idx - 1) % args.stride:
                    continue
                processed += 1

                # PASO 1: detectar personas en este cuadro
                cajas = det.detectar(frame)
                # PASO 2: emparejarlas con las personas que ya conociamos
                salieron = rastreador.actualizar(cajas, t_video)
                # [NUEVO] quienes ya se fueron de la imagen se cuentan en "vistos"
                for p in salieron:
                    registrar_visto(p)

                # [NUEVO] PASO 2b: observar tamano y color de cada persona visible
                for p in rastreador.visibles():
                    clasif.observar(p, frame, cajas)

                # PASO 3: revisar si alguien cruzo la linea
                for p in rastreador.visibles():
                    tipo = linea.revisar(p, t_video)
                    if tipo:
                        tam, col = clasif.registrar_cruce(p)          # [NUEVO]
                        linea.eventos.append((t_video, tipo, p.id, tam, col))
                        # PASO 4: registrar el cruce en el CSV
                        csv_w.writerow([round(t_video, 2),
                                        datetime.now().isoformat(timespec="seconds"),
                                        tipo, p.id, tam, col,
                                        linea.entradas, linea.salidas,
                                        linea.aforo])
                        csv_f.flush()   # escribir al disco ya, por si el programa se corta
                        print(f"\n  [{timedelta(seconds=int(t_video))}] {tipo}"
                              f"  persona #{p.id} ({tam}, ropa {col})"
                              f"  ->  aforo={linea.aforo}")

                # Guardamos cuantas personas se ven (para el grafico y el CSV de ocupacion)
                visibles = rastreador.visibles()
                serie.append((t_video, len(visibles)))
                if serie_w and t_video - ultimo_log >= 1.0:    # como maximo una fila por segundo
                    ultimo_log = t_video
                    serie_w.writerow([round(t_video, 2),
                                      datetime.now().isoformat(timespec="seconds"),
                                      len(visibles), linea.aforo])
                    serie_f.flush()

                # PASO 5: dibujar todo sobre el cuadro
                progress = (t_video / dur) if dur else None    # fraccion del video recorrida (0 a 1)
                vis = dibujar(frame, visibles, linea, t_video, progress,
                              serie, args.aforo_max, clasif)
                if writer:
                    writer.write(vis)                           # agregar el cuadro al video de salida

                # Sin ventana: mostramos el avance en la terminal cada 25 cuadros
                if args.headless and processed % 25 == 0:
                    pct = f"{progress*100:5.1f}%" if progress else f"{processed} cuadros"
                    # \r vuelve al inicio de la linea: el texto se sobrescribe en el mismo renglon
                    sys.stdout.write(f"\r  {pct}  t={timedelta(seconds=int(t_video))}  "
                                     f"visibles={len(visibles)}  "
                                     f"in={linea.entradas} out={linea.salidas} "
                                     f"aforo={linea.aforo}   ")
                    sys.stdout.flush()

                # --speed: si vamos mas rapido que el video real, esperamos un poco
                if args.speed > 0:
                    objetivo = (t_video - args.start) / args.speed   # cuanto tiempo real deberia haber pasado
                    lag = objetivo - (time.time() - t0_wall)         # cuanto vamos adelantados
                    if lag > 0:
                        time.sleep(min(lag, 0.25))

            # ---- Mostrar en ventana y leer el teclado ---- #
            if not args.headless:
                try:
                    cv2.imshow("Conteo de personas", vis)
                except cv2.error:
                    # Algunas instalaciones (servidores, Colab) no pueden abrir ventanas
                    print("[aviso] este OpenCV no tiene ventanas; sigo headless.")
                    args.headless = True
                    continue
                # waitKey(1): espera 1 ms por una tecla. Sin esta llamada la ventana no se actualiza.
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):               # q = salir
                    break
                if k == ord(" "):               # espacio = pausa / continuar
                    paused = not paused
                if k == ord("s"):               # s = guardar captura PNG
                    nombre = f"captura_{int(t_video)}s.png"
                    cv2.imwrite(nombre, vis)
                    print("Guardado", nombre)
                if k == ord("r"):               # r = reiniciar contadores
                    linea.entradas = linea.salidas = 0
                    linea.eventos.clear()
                    clasif.reiniciar()          # [NUEVO]
                    print("Contadores reiniciados.")
    except KeyboardInterrupt:
        # Ctrl+C en la terminal: salimos ordenadamente
        print("\nInterrumpido.")
    finally:
        # [NUEVO] las personas que siguen en pantalla al terminar tambien cuentan
        for p in rastreador.personas:
            registrar_visto(p)
        # "finally" se ejecuta SIEMPRE, aunque haya error: cerramos todo
        # para que el video y los CSV queden bien guardados.
        cap.release()
        if writer:
            writer.release()
        csv_f.close()
        pers_f.close()
        if serie_f:
            serie_f.close()
        if not args.headless:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ----------------------------- resumen ----------------------------- #
    picos = [v for _, v in serie] or [0]
    print("\n" + "=" * 62)
    print(f"Cuadros procesados : {processed}")
    print(f"Tiempo de video    : {timedelta(seconds=int(t_video))}")
    print(f"Entradas           : {linea.entradas}")
    print(f"Salidas            : {linea.salidas}")
    print(f"Aforo final        : {linea.aforo}")
    print(f"Pico simultaneo    : {max(picos)} personas en pantalla")

    # [NUEVO] resumen del perfil
    print("\nPERFIL                cruzaron   vistos")
    for etiqueta, clave in (("Ninos", "NINO"), ("Adultos", "ADULTO"),
                            ("Sin datos (?)", "?")):
        print(f"  {etiqueta:<18} {clasif.cruce_tam[clave]:>8} {clasif.visto_tam[clave]:>8}")
    print(f"  {'Ropa negra':<18} {clasif.cruce_color['negro']:>8} "
          f"{clasif.visto_color['negro']:>8}")
    if clasif.visto_color:
        print("\n  Colores de ropa (vistos):")
        for col, n in clasif.visto_color.most_common():
            print(f"    {col:<10} {n}")
    if clasif.registro:
        # Sirve para calibrar --umbral-nino: mira en que valor se separan
        alts = sorted(f[1] for f in clasif.registro)
        print(f"\n  Razon altura/adulto: min {alts[0]:.2f}  "
              f"max {alts[-1]:.2f}  (umbral nino = {args.umbral_nino})")

    if linea.eventos:
        # Usamos la clasificacion FINAL (con todas las muestras), no la provisional
        final = {f[0]: (f[2], f[3]) for f in clasif.registro}
        print("\nCronologia (clasificacion final):")
        for t, tipo, pid, tam, col in linea.eventos:
            tam, col = final.get(pid, (tam, col))
            print(f"   {str(timedelta(seconds=int(t))):>8}  {tipo:<8} "
                  f"persona #{pid}  {tam:<6} {col}")
    print(f"\nCSV de eventos     : {args.csv}")
    print(f"CSV de personas    : {args.personas_csv}")
    if args.serie_csv:
        print(f"CSV de ocupacion   : {args.serie_csv}")
    if args.record:
        print(f"Video anotado      : {args.record}")
    print("=" * 62)


# Esto hace que main() se ejecute solo cuando corres el archivo directamente
# (python contador.py), y no cuando otro programa lo importa.
if __name__ == "__main__":
    main()
