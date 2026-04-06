"""
Generador de dataset sintético v2 — Keyword Mapping.

Para cada patología del JSON, lee 'caracteristicas' y 'causas' como texto,
busca palabras clave y asigna probabilidades a cada feature.
Reglas de coherencia obligatorias garantizan que no haya combinaciones
físicamente imposibles.

Genera 40 variantes por patología = 3840 filas totales.
"""

import json
import re
import numpy as np
import pandas as pd

np.random.seed(42)

# ══════════════════════════════════════════════════════════════════════
# 1. Cargar catálogo
# ══════════════════════════════════════════════════════════════════════
with open("patologias_estructurales.json", encoding="utf-8") as f:
    catalogo = json.load(f)

# ══════════════════════════════════════════════════════════════════════
# 2. Opciones válidas por feature (esquema exacto, no se modifica)
# ══════════════════════════════════════════════════════════════════════
OPCIONES = {
    "tipo_elemento": ["pilar", "viga", "mensula", "vigueta", "voladizo",
                      "forjado_reticular", "forjado_unidireccional",
                      "tabique_cerramiento", "cimentacion_zapata", "muro"],
    "orientacion_fisura": ["horizontal", "vertical", "diagonal_45",
                           "diagonal_45_75", "red_mapa", "sin_fisura"],
    "abertura": ["cerrada_fina", "media", "abierta", "grieta_separacion"],
    "ubicacion_en_elemento": ["cabeza", "base", "centro", "esquinas",
                              "ambas_caras", "perimetro", "cara_inferior",
                              "union_otro_elemento"],
    "patron": ["unica", "multiples_paralelas", "multiples_distintos_planos",
               "ramificada", "periodica"],
    "comportamiento": ["se_cierra_al_descender", "se_cierra_al_alejarse",
                       "se_abre_con_tiempo", "estable", "aparecio_de_golpe"],
    "desprendimiento": ["no_hay", "recubrimiento_desprendido",
                        "desmoronandose", "perdida_seccion"],
    "color": ["normal", "oscurecido", "manchas_blancas", "manchas_oxido"],
    "textura": ["normal", "poroso", "disgregado", "humedo_filtraciones"],
    "armadura_visible": ["no", "parcialmente", "completamente"],
    "corrosion": ["no_visible", "manchas_oxido", "oxidada_visible",
                  "perdida_seccion"],
    "estado_estribos": ["correctos", "separados", "desplazados", "ausentes"],
    "edad_estructura": ["menos_5", "5_a_20", "20_a_50", "mas_50"],
    "ambiente": ["interior", "exterior_continental", "costero_marino",
                 "sumergido", "industrial"],
    "velocidad_aparicion": ["durante_construccion", "primeras_semanas",
                            "primeros_meses", "anos_despues",
                            "subita_reciente"],
    "carga": ["normal", "sobrecarga", "impacto_sismo", "descarga"],
    "intervencion_previa": ["original", "reparada", "reforzada",
                            "recien_desencofrada"],
    "deformacion_visible": ["no", "flecha_abajo", "desplazamiento_lateral",
                            "giro_inclinacion", "aplastamiento"],
    "sonido_golpe": ["solido", "hueco", "diferente_zonas"],
    "presencia_agua": ["seco", "humedo", "filtracion_activa", "salpicadura"],
}

VARIANTES_POR_PATOLOGIA = 40


# ══════════════════════════════════════════════════════════════════════
# 3. Funciones auxiliares
# ══════════════════════════════════════════════════════════════════════

def tiene(texto, *palabras):
    """True si alguna palabra clave aparece en el texto."""
    t = texto.lower()
    return any(p.lower() in t for p in palabras)


def elegir(opciones, pesos=None):
    """Elige un valor de la lista con pesos normalizados."""
    pesos = np.array(pesos if pesos else [1.0] * len(opciones), dtype=float)
    pesos /= pesos.sum()
    return np.random.choice(opciones, p=pesos)


def uniforme(feature):
    """Distribución uniforme para una feature."""
    return (OPCIONES[feature], None)


def dist(feature, mapa_pesos):
    """Crea distribución: mapa_pesos = {valor: peso_relativo}.
    Valores no listados reciben peso 1."""
    opciones = OPCIONES[feature]
    pesos = []
    for op in opciones:
        pesos.append(mapa_pesos.get(op, 1.0))
    return (opciones, pesos)


# ══════════════════════════════════════════════════════════════════════
# 4. Mapeo categoría → tipo_elemento
# ══════════════════════════════════════════════════════════════════════

CATEGORIA_ELEMENTO = {
    "Pilares": dist("tipo_elemento", {"pilar": 50}),
    "Vigas": dist("tipo_elemento", {"viga": 50}),
    "Pilares y Vigas": dist("tipo_elemento", {"pilar": 25, "viga": 25}),
    "Ménsulas": dist("tipo_elemento", {"mensula": 50}),
    "Viguetas": dist("tipo_elemento", {"vigueta": 50}),
    "Voladizos": dist("tipo_elemento", {"voladizo": 50}),
    "Forjados": dist("tipo_elemento", {
        "forjado_reticular": 25, "forjado_unidireccional": 25}),
    "Deformaciones": dist("tipo_elemento", {
        "viga": 12, "forjado_unidireccional": 12,
        "tabique_cerramiento": 12, "voladizo": 8}),
    "Térmicos": dist("tipo_elemento", {
        "tabique_cerramiento": 15, "forjado_unidireccional": 10, "muro": 8}),
    "Cerramientos": dist("tipo_elemento", {
        "tabique_cerramiento": 30, "muro": 15}),
    "Cimentaciones": dist("tipo_elemento", {
        "cimentacion_zapata": 30, "muro": 12}),
}


# ══════════════════════════════════════════════════════════════════════
# 5. Keyword mapping: texto → distribuciones de probabilidad
# ══════════════════════════════════════════════════════════════════════

def construir_distribuciones(texto, gravedad_pts, categoria):
    """Analiza el texto (caracteristicas + causas) y devuelve un dict
    {feature: (opciones, pesos)} con las distribuciones de probabilidad."""

    d = {}

    # ── tipo_elemento (por categoría) ──────────────────────────────
    d["tipo_elemento"] = CATEGORIA_ELEMENTO.get(
        categoria, uniforme("tipo_elemento"))

    # ── orientacion_fisura ─────────────────────────────────────────
    if tiene(texto, "45°", "cortante", "punzonamiento"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "diagonal_45": 45, "diagonal_45_75": 30})
    elif tiene(texto, "horizontal", "flexión", "flexion"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "horizontal": 35, "vertical": 25})
    elif tiene(texto, "vertical"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "vertical": 40, "horizontal": 10})
    elif tiene(texto, "red", "contorno", "torsión", "torsion"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "red_mapa": 35, "diagonal_45_75": 20})
    elif tiene(texto, "diagonal", "inclinada"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "diagonal_45": 25, "diagonal_45_75": 25, "vertical": 8})
    elif tiene(texto, "aplastamiento", "desagregación", "desagreg"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "red_mapa": 20, "vertical": 12, "sin_fisura": 15})
    elif tiene(texto, "corrosión", "corrosion", "óxido", "oxido"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "horizontal": 20, "sin_fisura": 15, "vertical": 10})
    elif tiene(texto, "retracción", "retraccion"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "vertical": 20, "horizontal": 15, "red_mapa": 12})
    elif tiene(texto, "flecha", "deformación", "deformacion"):
        d["orientacion_fisura"] = dist("orientacion_fisura", {
            "horizontal": 18, "diagonal_45": 12, "sin_fisura": 10})
    else:
        d["orientacion_fisura"] = uniforme("orientacion_fisura")

    # ── abertura (por gravedad) ────────────────────────────────────
    if gravedad_pts >= 4:
        d["abertura"] = dist("abertura", {
            "abierta": 30, "grieta_separacion": 25, "media": 10})
    elif gravedad_pts == 3:
        d["abertura"] = dist("abertura", {
            "media": 30, "abierta": 18, "cerrada_fina": 6})
    elif gravedad_pts == 2:
        d["abertura"] = dist("abertura", {
            "cerrada_fina": 25, "media": 20, "abierta": 5})
    else:
        d["abertura"] = dist("abertura", {
            "cerrada_fina": 40, "media": 10})

    # ── ubicacion_en_elemento ──────────────────────────────────────
    if tiene(texto, "cabeza", "coronación", "coronacion"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "cabeza": 35, "esquinas": 10, "union_otro_elemento": 8})
    elif tiene(texto, "base", "pie", "zapata", "cimiento"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "base": 35, "union_otro_elemento": 10})
    elif tiene(texto, "esquina"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "esquinas": 40, "cabeza": 8})
    elif tiene(texto, "cara inferior", "inferior"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "cara_inferior": 35, "centro": 10})
    elif tiene(texto, "centro", "luz", "mitad"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "centro": 35, "cara_inferior": 10})
    elif tiene(texto, "perímetro", "perimetro", "contorno", "alrededor"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "perimetro": 35, "ambas_caras": 10})
    elif tiene(texto, "dos caras", "ambas caras", "opuestas"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "ambas_caras": 35, "esquinas": 10})
    elif tiene(texto, "apoyo", "unión", "union", "empotramiento"):
        d["ubicacion_en_elemento"] = dist("ubicacion_en_elemento", {
            "union_otro_elemento": 30, "base": 12, "cabeza": 10})
    else:
        d["ubicacion_en_elemento"] = uniforme("ubicacion_en_elemento")

    # ── patron ─────────────────────────────────────────────────────
    if tiene(texto, "periódica", "distancias periódicas", "periodica"):
        d["patron"] = dist("patron", {"periodica": 40, "multiples_paralelas": 15})
    elif tiene(texto, "paralela"):
        d["patron"] = dist("patron", {"multiples_paralelas": 40, "periodica": 10})
    elif tiene(texto, "red", "contorno", "ramificada"):
        d["patron"] = dist("patron", {
            "ramificada": 30, "multiples_distintos_planos": 20})
    elif tiene(texto, "distintos planos"):
        d["patron"] = dist("patron", {
            "multiples_distintos_planos": 35, "ramificada": 12})
    elif tiene(texto, "secciona", "corta la"):
        d["patron"] = dist("patron", {"unica": 30, "multiples_paralelas": 12})
    else:
        d["patron"] = uniforme("patron")

    # ── comportamiento ─────────────────────────────────────────────
    if tiene(texto, "se cierra") and tiene(texto, "descend", "desciende"):
        d["comportamiento"] = dist("comportamiento", {
            "se_cierra_al_descender": 40, "estable": 8})
    elif tiene(texto, "se cierra") and tiene(texto, "aleja"):
        d["comportamiento"] = dist("comportamiento", {
            "se_cierra_al_alejarse": 40, "estable": 8})
    elif tiene(texto, "se abre", "tiende a abrirse"):
        d["comportamiento"] = dist("comportamiento", {
            "se_abre_con_tiempo": 40, "estable": 8})
    elif tiene(texto, "rápida", "instantánea", "instantanea", "golpe", "súbita"):
        d["comportamiento"] = dist("comportamiento", {
            "aparecio_de_golpe": 40, "se_abre_con_tiempo": 10})
    elif tiene(texto, "retracción", "retraccion"):
        d["comportamiento"] = dist("comportamiento", {
            "estable": 30, "se_cierra_al_alejarse": 12})
    else:
        d["comportamiento"] = uniforme("comportamiento")

    # ── desprendimiento ────────────────────────────────────────────
    if tiene(texto, "desmoron", "desagreg", "disgregado"):
        d["desprendimiento"] = dist("desprendimiento", {
            "desmoronandose": 35, "perdida_seccion": 15,
            "recubrimiento_desprendido": 8})
    elif tiene(texto, "despren", "caen", "se desprenden"):
        d["desprendimiento"] = dist("desprendimiento", {
            "recubrimiento_desprendido": 30, "perdida_seccion": 15})
    elif tiene(texto, "aplastamiento"):
        d["desprendimiento"] = dist("desprendimiento", {
            "recubrimiento_desprendido": 22, "desmoronandose": 15})
    elif gravedad_pts >= 4:
        d["desprendimiento"] = dist("desprendimiento", {
            "recubrimiento_desprendido": 20, "perdida_seccion": 10})
    else:
        d["desprendimiento"] = dist("desprendimiento", {
            "no_hay": 40, "recubrimiento_desprendido": 6})

    # ── color ──────────────────────────────────────────────────────
    if tiene(texto, "corrosión", "corrosion", "óxido", "oxido", "oxidada"):
        d["color"] = dist("color", {"manchas_oxido": 45, "normal": 5})
    elif tiene(texto, "aluminosis", "oscuro", "oscurecido"):
        d["color"] = dist("color", {"oscurecido": 45, "normal": 5})
    elif tiene(texto, "eflorescencia", "manchas blancas", "carbonatación"):
        d["color"] = dist("color", {"manchas_blancas": 40, "normal": 8})
    elif tiene(texto, "humedad", "agua"):
        d["color"] = dist("color", {
            "manchas_blancas": 18, "manchas_oxido": 12, "normal": 10})
    else:
        d["color"] = dist("color", {"normal": 30, "manchas_oxido": 4,
                                     "oscurecido": 3, "manchas_blancas": 3})

    # ── textura ────────────────────────────────────────────────────
    if tiene(texto, "desagreg", "desmoron", "disgregado"):
        d["textura"] = dist("textura", {"disgregado": 40, "poroso": 15})
    elif tiene(texto, "poroso", "poco compacto", "escaso vibrado"):
        d["textura"] = dist("textura", {"poroso": 30, "disgregado": 12})
    elif tiene(texto, "humedad", "filtra", "agua", "depósito"):
        d["textura"] = dist("textura", {
            "humedo_filtraciones": 30, "poroso": 12})
    else:
        d["textura"] = dist("textura", {"normal": 30, "poroso": 6})

    # ── armadura_visible ───────────────────────────────────────────
    if tiene(texto, "barras vistas", "barras", "se pueden perder"):
        d["armadura_visible"] = dist("armadura_visible", {
            "completamente": 30, "parcialmente": 20})
    elif tiene(texto, "corrosión", "corrosion"):
        d["armadura_visible"] = dist("armadura_visible", {
            "parcialmente": 30, "completamente": 12, "no": 5})
    elif tiene(texto, "despren") or gravedad_pts >= 4:
        d["armadura_visible"] = dist("armadura_visible", {
            "parcialmente": 20, "no": 15, "completamente": 8})
    else:
        d["armadura_visible"] = dist("armadura_visible", {
            "no": 35, "parcialmente": 8})

    # ── corrosion ──────────────────────────────────────────────────
    if tiene(texto, "corrosión", "corrosion"):
        d["corrosion"] = dist("corrosion", {
            "oxidada_visible": 30, "perdida_seccion": 18, "manchas_oxido": 15})
    elif tiene(texto, "marítim", "costero", "marino", "mar "):
        d["corrosion"] = dist("corrosion", {
            "manchas_oxido": 25, "oxidada_visible": 15})
    elif gravedad_pts >= 3 and tiene(texto, "armadura"):
        d["corrosion"] = dist("corrosion", {
            "manchas_oxido": 20, "oxidada_visible": 10})
    else:
        d["corrosion"] = dist("corrosion", {
            "no_visible": 40, "manchas_oxido": 6})

    # ── estado_estribos ────────────────────────────────────────────
    if tiene(texto, "estribo", "cercos caídos", "caída de estribos"):
        d["estado_estribos"] = dist("estado_estribos", {
            "desplazados": 25, "ausentes": 20, "separados": 15})
    elif tiene(texto, "cercos", "estribos") and tiene(texto, "separad", "insuficiente"):
        d["estado_estribos"] = dist("estado_estribos", {
            "separados": 28, "desplazados": 15, "ausentes": 10})
    elif tiene(texto, "armadura transversal"):
        d["estado_estribos"] = dist("estado_estribos", {
            "separados": 22, "ausentes": 15})
    else:
        d["estado_estribos"] = dist("estado_estribos", {
            "correctos": 35, "separados": 6})

    # ── edad_estructura ────────────────────────────────────────────
    if tiene(texto, "retracción plástica", "retraccion plastica"):
        d["edad_estructura"] = dist("edad_estructura", {
            "menos_5": 45})
    elif tiene(texto, "retracción hidráulica", "retraccion hidraulica"):
        d["edad_estructura"] = dist("edad_estructura", {
            "menos_5": 35, "5_a_20": 15})
    elif tiene(texto, "retracción", "retraccion"):
        d["edad_estructura"] = dist("edad_estructura", {
            "menos_5": 30, "5_a_20": 15})
    elif tiene(texto, "aluminosis"):
        d["edad_estructura"] = dist("edad_estructura", {
            "20_a_50": 25, "mas_50": 25})
    elif tiene(texto, "corrosión", "corrosion"):
        d["edad_estructura"] = dist("edad_estructura", {
            "20_a_50": 22, "mas_50": 22, "5_a_20": 6})
    elif tiene(texto, "desencofrad", "hormigonado", "fresco"):
        d["edad_estructura"] = dist("edad_estructura", {
            "menos_5": 40, "5_a_20": 8})
    elif tiene(texto, "en servicio"):
        d["edad_estructura"] = dist("edad_estructura", {
            "anos_despues": 20, "5_a_20": 15, "20_a_50": 12})
    else:
        d["edad_estructura"] = uniforme("edad_estructura")

    # ── ambiente ───────────────────────────────────────────────────
    if tiene(texto, "marítim", "costero", "marino", "mar ", "salpicadura"):
        d["ambiente"] = dist("ambiente", {
            "costero_marino": 40, "exterior_continental": 8})
    elif tiene(texto, "químic", "agresiv", "industrial"):
        d["ambiente"] = dist("ambiente", {
            "industrial": 30, "costero_marino": 10})
    elif tiene(texto, "humedad", "agua", "depósito", "sumergido"):
        d["ambiente"] = dist("ambiente", {
            "exterior_continental": 18, "costero_marino": 12, "sumergido": 12})
    elif tiene(texto, "soleamiento", "sol ", "calurosa", "viento"):
        d["ambiente"] = dist("ambiente", {
            "exterior_continental": 35, "interior": 8})
    else:
        d["ambiente"] = uniforme("ambiente")

    # ── velocidad_aparicion ────────────────────────────────────────
    if tiene(texto, "retracción plástica", "retraccion plastica"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "durante_construccion": 40, "primeras_semanas": 15})
    elif tiene(texto, "retracción hidráulica", "retraccion hidraulica"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "primeras_semanas": 30, "primeros_meses": 20})
    elif tiene(texto, "retracción térmica", "retraccion termica"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "primeros_meses": 25, "anos_despues": 15})
    elif tiene(texto, "desencofrad", "hormigonado", "fresco"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "durante_construccion": 35, "primeras_semanas": 15})
    elif tiene(texto, "primeros meses", "primeras semanas"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "primeras_semanas": 22, "primeros_meses": 22})
    elif tiene(texto, "corrosión", "corrosion", "aluminosis"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "anos_despues": 40, "primeros_meses": 8})
    elif tiene(texto, "en servicio"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "anos_despues": 30, "primeros_meses": 12})
    elif tiene(texto, "rápida", "instantánea", "instantanea", "súbita"):
        d["velocidad_aparicion"] = dist("velocidad_aparicion", {
            "subita_reciente": 35, "durante_construccion": 10})
    else:
        d["velocidad_aparicion"] = uniforme("velocidad_aparicion")

    # ── carga ──────────────────────────────────────────────────────
    if tiene(texto, "sobrecarga", "exceso de carga"):
        d["carga"] = dist("carga", {"sobrecarga": 40, "normal": 8})
    elif tiene(texto, "sismo", "impacto", "empuje"):
        d["carga"] = dist("carga", {
            "impacto_sismo": 30, "sobrecarga": 12})
    elif tiene(texto, "descarga", "desencofrad"):
        d["carga"] = dist("carga", {"descarga": 25, "normal": 18})
    else:
        d["carga"] = dist("carga", {"normal": 30, "sobrecarga": 10})

    # ── intervencion_previa ────────────────────────────────────────
    if tiene(texto, "desencofrad", "fresco", "hormigonado"):
        d["intervencion_previa"] = dist("intervencion_previa", {
            "recien_desencofrada": 35, "original": 12})
    elif tiene(texto, "reparad", "reforzad"):
        d["intervencion_previa"] = dist("intervencion_previa", {
            "reparada": 20, "reforzada": 15, "original": 10})
    else:
        d["intervencion_previa"] = dist("intervencion_previa", {
            "original": 30, "reparada": 5, "reforzada": 4,
            "recien_desencofrada": 3})

    # ── deformacion_visible ────────────────────────────────────────
    if tiene(texto, "flecha"):
        d["deformacion_visible"] = dist("deformacion_visible", {
            "flecha_abajo": 45, "no": 5})
    elif tiene(texto, "giro", "inclinación", "inclinacion"):
        d["deformacion_visible"] = dist("deformacion_visible", {
            "giro_inclinacion": 35, "desplazamiento_lateral": 12})
    elif tiene(texto, "aplastamiento"):
        d["deformacion_visible"] = dist("deformacion_visible", {
            "aplastamiento": 30, "no": 10, "flecha_abajo": 6})
    elif tiene(texto, "pandeo"):
        d["deformacion_visible"] = dist("deformacion_visible", {
            "desplazamiento_lateral": 35, "giro_inclinacion": 12})
    elif tiene(texto, "desplazamiento", "lateral"):
        d["deformacion_visible"] = dist("deformacion_visible", {
            "desplazamiento_lateral": 35, "no": 8})
    elif gravedad_pts >= 3:
        d["deformacion_visible"] = dist("deformacion_visible", {
            "flecha_abajo": 12, "no": 18, "desplazamiento_lateral": 10})
    else:
        d["deformacion_visible"] = dist("deformacion_visible", {
            "no": 40, "flecha_abajo": 5})

    # ── sonido_golpe ───────────────────────────────────────────────
    if tiene(texto, "desagreg", "poroso", "disgregado"):
        d["sonido_golpe"] = dist("sonido_golpe", {
            "hueco": 30, "diferente_zonas": 20})
    elif tiene(texto, "despren"):
        d["sonido_golpe"] = dist("sonido_golpe", {
            "hueco": 22, "diferente_zonas": 18})
    elif gravedad_pts >= 3:
        d["sonido_golpe"] = dist("sonido_golpe", {
            "diferente_zonas": 22, "solido": 10, "hueco": 12})
    else:
        d["sonido_golpe"] = dist("sonido_golpe", {
            "solido": 28, "diferente_zonas": 8})

    # ── presencia_agua ─────────────────────────────────────────────
    if tiene(texto, "filtra", "humedad relativa alta"):
        d["presencia_agua"] = dist("presencia_agua", {
            "filtracion_activa": 30, "humedo": 18})
    elif tiene(texto, "marítim", "costero", "salpicadura"):
        d["presencia_agua"] = dist("presencia_agua", {
            "salpicadura": 28, "humedo": 15})
    elif tiene(texto, "agua", "humedad"):
        d["presencia_agua"] = dist("presencia_agua", {
            "humedo": 25, "filtracion_activa": 12})
    else:
        d["presencia_agua"] = dist("presencia_agua", {
            "seco": 28, "humedo": 8})

    return d


# ══════════════════════════════════════════════════════════════════════
# 6. Reglas de coherencia (nunca se violan, ni con ruido)
# ══════════════════════════════════════════════════════════════════════

def aplicar_coherencia(obs):
    """Corrige valores para garantizar coherencia física."""

    # Regla 1: sin_fisura → abertura cerrada, patron unica, estable
    if obs["orientacion_fisura"] == "sin_fisura":
        obs["abertura"] = "cerrada_fina"
        obs["patron"] = "unica"
        obs["comportamiento"] = "estable"

    # Regla 2: corrosión visible → armadura debe ser visible
    if obs["corrosion"] in ("oxidada_visible", "perdida_seccion"):
        if obs["armadura_visible"] == "no":
            obs["armadura_visible"] = np.random.choice(
                ["parcialmente", "completamente"], p=[0.7, 0.3])

    # Regla 3: desmoronándose → textura disgregado o poroso
    if obs["desprendimiento"] == "desmoronandose":
        if obs["textura"] not in ("disgregado", "poroso"):
            obs["textura"] = np.random.choice(
                ["disgregado", "poroso"], p=[0.7, 0.3])

    return obs


# ══════════════════════════════════════════════════════════════════════
# 7. Generación del dataset
# ══════════════════════════════════════════════════════════════════════

print("Generando dataset sintético v2 (keyword mapping)...")
filas = []

for pat in catalogo:
    texto = " ".join(pat["caracteristicas"] + pat["causas"]).lower()
    gravedad_pts = pat["gravedad_puntos"]
    categoria = pat["categoria"]
    distribuciones = construir_distribuciones(texto, gravedad_pts, categoria)

    for i in range(VARIANTES_POR_PATOLOGIA):
        obs = {}

        # Generar cada feature desde su distribución
        for feature in OPCIONES:
            if feature in distribuciones:
                opciones, pesos = distribuciones[feature]
                obs[feature] = elegir(opciones, pesos)
            else:
                obs[feature] = elegir(OPCIONES[feature])

        # 5% de ruido: perturbar 1-2 features aleatorias (sin violar coherencia)
        if np.random.random() < 0.05:
            features_perturbables = list(OPCIONES.keys())
            n_perturbar = np.random.randint(1, 3)
            features_ruido = np.random.choice(
                features_perturbables, size=n_perturbar, replace=False)
            for f_ruido in features_ruido:
                obs[f_ruido] = np.random.choice(OPCIONES[f_ruido])

        # Aplicar reglas de coherencia SIEMPRE (post-ruido)
        obs = aplicar_coherencia(obs)
        obs["defecto_numero"] = pat["numero"]
        filas.append(obs)

df = pd.DataFrame(filas)
features = [c for c in df.columns if c != "defecto_numero"]
df = df[features + ["defecto_numero"]]
df = df.sample(frac=1, random_state=42).reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════
# 8. Validación de coherencia
# ══════════════════════════════════════════════════════════════════════

print(f"\nDataset generado: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Patologías únicas: {df['defecto_numero'].nunique()}")

# Validar reglas
v1 = df[df["orientacion_fisura"] == "sin_fisura"]
v1_fallo = v1[
    (v1["abertura"] != "cerrada_fina") |
    (v1["patron"] != "unica") |
    (v1["comportamiento"] != "estable")
]

v2 = df[df["corrosion"].isin(["oxidada_visible", "perdida_seccion"])]
v2_fallo = v2[v2["armadura_visible"] == "no"]

v3 = df[df["desprendimiento"] == "desmoronandose"]
v3_fallo = v3[~v3["textura"].isin(["disgregado", "poroso"])]

print(f"\n── VALIDACIÓN DE COHERENCIA ──")
print(f"  Regla 1 (sin_fisura → cerrada/unica/estable): "
      f"{'OK' if len(v1_fallo) == 0 else f'FALLO: {len(v1_fallo)} filas'} "
      f"({len(v1)} filas afectadas)")
print(f"  Regla 2 (corrosión visible → armadura visible): "
      f"{'OK' if len(v2_fallo) == 0 else f'FALLO: {len(v2_fallo)} filas'} "
      f"({len(v2)} filas afectadas)")
print(f"  Regla 3 (desmoronándose → textura disgregado/poroso): "
      f"{'OK' if len(v3_fallo) == 0 else f'FALLO: {len(v3_fallo)} filas'} "
      f"({len(v3)} filas afectadas)")

print(f"\n── DISTRIBUCIÓN POR CLASE ──")
conteo = df["defecto_numero"].value_counts()
print(f"  Filas por clase: {conteo.min()} (todas {VARIANTES_POR_PATOLOGIA})")
print(f"  Total clases: {conteo.shape[0]}")

# Exportar
df.to_csv("dataset_patologias_sintetico.csv", index=False)
print(f"\n✓ Archivo guardado: dataset_patologias_sintetico.csv")
