"""
Script para generar el dataset sintético de entrenamiento.
Cada fila simula una observación de campo de un inspector que evalúa
patologías estructurales en hormigón armado.
Se generan entre 15 y 30 variantes por cada una de las 96 patologías.
"""

import json
import numpy as np
import pandas as pd

np.random.seed(42)

# ── Cargar catálogo de patologías ──────────────────────────────────────────
with open("patologias_estructurales.json", encoding="utf-8") as f:
    catalogo = json.load(f)

# ── Opciones válidas por feature ───────────────────────────────────────────
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

# ── Función auxiliar: elegir con pesos ─────────────────────────────────────
def elegir(opciones, pesos=None):
    """Elige un valor de la lista con pesos opcionales."""
    if pesos is None:
        return np.random.choice(opciones)
    pesos = np.array(pesos, dtype=float)
    pesos /= pesos.sum()
    return np.random.choice(opciones, p=pesos)


# ── Mapeo categoría → tipo_elemento ────────────────────────────────────────
def elemento_por_categoria(cat):
    """Asigna tipo_elemento según la categoría del catálogo."""
    cat_lower = cat.lower()
    if "pilar" in cat_lower and "viga" in cat_lower:
        return elegir(["pilar", "viga"], [0.5, 0.5])
    if "pilar" in cat_lower:
        return elegir(["pilar"], [1.0])
    if "ménsula" in cat_lower or "mensul" in cat_lower:
        return elegir(["mensula"], [1.0])
    if "vigueta" in cat_lower:
        return elegir(["vigueta"], [1.0])
    if "voladizo" in cat_lower:
        return elegir(["voladizo"], [1.0])
    if "forjado" in cat_lower:
        return elegir(["forjado_reticular", "forjado_unidireccional"], [0.5, 0.5])
    if "viga" in cat_lower:
        return elegir(["viga"], [1.0])
    if "deformacion" in cat_lower or "térmico" in cat_lower or "termico" in cat_lower:
        return elegir(["viga", "forjado_unidireccional", "tabique_cerramiento",
                        "voladizo"], [0.3, 0.3, 0.3, 0.1])
    if "cerramiento" in cat_lower:
        return elegir(["tabique_cerramiento", "muro"], [0.7, 0.3])
    if "cimentacion" in cat_lower:
        return elegir(["cimentacion_zapata", "muro"], [0.7, 0.3])
    # fallback
    return elegir(OPCIONES["tipo_elemento"])


# ── Perfiles de observación por patología ──────────────────────────────────
# Cada perfil define distribuciones de probabilidad para las features.
# Se usan las características y causas del JSON como guía.

def perfil_por_defecto():
    """Retorna un diccionario con valores por defecto (distribución uniforme)."""
    return {k: (v, None) for k, v in OPCIONES.items() if k != "tipo_elemento"}


def generar_observacion(patologia):
    """Genera una observación de campo para una patología dada."""
    num = patologia["numero"]
    cat = patologia["categoria"]
    defecto = patologia["defecto"].lower()
    caract = " ".join(patologia["caracteristicas"]).lower()
    causas = " ".join(patologia["causas"]).lower()
    gravedad_pts = patologia["gravedad_puntos"]

    obs = {}
    obs["tipo_elemento"] = elemento_por_categoria(cat)

    # ── Orientación de fisura ──────────────────────────────────────────
    if "45°" in caract or "45°" in defecto or "cortante" in defecto:
        obs["orientacion_fisura"] = elegir(
            ["diagonal_45", "diagonal_45_75"], [0.6, 0.4])
    elif "horizontal" in caract:
        obs["orientacion_fisura"] = elegir(
            ["horizontal", "vertical"], [0.8, 0.2])
    elif "vertical" in caract or "vertical" in defecto:
        obs["orientacion_fisura"] = elegir(
            ["vertical", "horizontal"], [0.8, 0.2])
    elif "red" in caract or "contorno" in caract or "mapa" in caract:
        obs["orientacion_fisura"] = elegir(
            ["red_mapa", "diagonal_45_75"], [0.7, 0.3])
    elif "diagonal" in caract or "inclinada" in caract:
        obs["orientacion_fisura"] = elegir(
            ["diagonal_45", "diagonal_45_75", "vertical"], [0.4, 0.4, 0.2])
    elif "aplastamiento" in defecto or "desagregación" in defecto or "desagreg" in defecto:
        obs["orientacion_fisura"] = elegir(
            ["red_mapa", "vertical", "sin_fisura"], [0.4, 0.3, 0.3])
    elif "corrosión" in defecto or "corrosion" in defecto:
        obs["orientacion_fisura"] = elegir(
            ["horizontal", "sin_fisura", "vertical"], [0.4, 0.3, 0.3])
    elif "retracción" in defecto or "retraccion" in defecto:
        obs["orientacion_fisura"] = elegir(
            ["vertical", "horizontal", "red_mapa"], [0.4, 0.3, 0.3])
    elif "deformación" in defecto or "deformacion" in defecto or "flecha" in defecto:
        obs["orientacion_fisura"] = elegir(
            ["horizontal", "diagonal_45", "sin_fisura"], [0.4, 0.3, 0.3])
    else:
        obs["orientacion_fisura"] = elegir(
            OPCIONES["orientacion_fisura"], [0.15, 0.2, 0.2, 0.15, 0.1, 0.2])

    # ── Abertura ───────────────────────────────────────────────────────
    if gravedad_pts >= 4:
        obs["abertura"] = elegir(
            ["abierta", "grieta_separacion", "media"], [0.4, 0.4, 0.2])
    elif gravedad_pts == 3:
        obs["abertura"] = elegir(
            ["media", "abierta", "cerrada_fina"], [0.5, 0.3, 0.2])
    elif gravedad_pts == 2:
        obs["abertura"] = elegir(
            ["cerrada_fina", "media", "abierta"], [0.5, 0.35, 0.15])
    else:
        obs["abertura"] = elegir(
            ["cerrada_fina", "media"], [0.7, 0.3])

    # ── Ubicación en elemento ──────────────────────────────────────────
    if "cabeza" in caract or "cabeza" in defecto:
        obs["ubicacion_en_elemento"] = elegir(
            ["cabeza", "esquinas", "union_otro_elemento"], [0.6, 0.2, 0.2])
    elif "base" in caract or "pie" in caract or "zapata" in caract:
        obs["ubicacion_en_elemento"] = elegir(
            ["base", "union_otro_elemento", "centro"], [0.6, 0.2, 0.2])
    elif "esquina" in caract:
        obs["ubicacion_en_elemento"] = elegir(
            ["esquinas", "cabeza", "perimetro"], [0.6, 0.2, 0.2])
    elif "cara inferior" in caract or "inferior" in caract:
        obs["ubicacion_en_elemento"] = elegir(
            ["cara_inferior", "centro", "base"], [0.6, 0.2, 0.2])
    elif "centro" in caract or "luz" in caract:
        obs["ubicacion_en_elemento"] = elegir(
            ["centro", "cara_inferior", "ambas_caras"], [0.6, 0.2, 0.2])
    elif "perimetro" in caract or "contorno" in caract or "alrededor" in caract:
        obs["ubicacion_en_elemento"] = elegir(
            ["perimetro", "ambas_caras", "centro"], [0.6, 0.2, 0.2])
    elif "dos caras" in caract or "ambas" in caract:
        obs["ubicacion_en_elemento"] = elegir(
            ["ambas_caras", "esquinas", "perimetro"], [0.6, 0.2, 0.2])
    elif "apoyo" in caract:
        obs["ubicacion_en_elemento"] = elegir(
            ["union_otro_elemento", "base", "cabeza"], [0.5, 0.3, 0.2])
    else:
        obs["ubicacion_en_elemento"] = elegir(OPCIONES["ubicacion_en_elemento"])

    # ── Patrón ─────────────────────────────────────────────────────────
    if "periódica" in caract or "distancias periódicas" in caract:
        obs["patron"] = elegir(
            ["periodica", "multiples_paralelas"], [0.7, 0.3])
    elif "paralela" in caract:
        obs["patron"] = elegir(
            ["multiples_paralelas", "periodica"], [0.7, 0.3])
    elif "red" in caract or "contorno" in caract:
        obs["patron"] = elegir(
            ["ramificada", "multiples_distintos_planos"], [0.6, 0.4])
    elif "distintos planos" in caract:
        obs["patron"] = elegir(
            ["multiples_distintos_planos", "ramificada"], [0.7, 0.3])
    elif "secciona" in caract or "corta" in caract:
        obs["patron"] = elegir(
            ["unica", "multiples_paralelas"], [0.6, 0.4])
    else:
        obs["patron"] = elegir(OPCIONES["patron"])

    # ── Comportamiento ─────────────────────────────────────────────────
    if "se cierra" in caract and "descend" in caract:
        obs["comportamiento"] = elegir(
            ["se_cierra_al_descender", "estable"], [0.7, 0.3])
    elif "se cierra" in caract and "aleja" in caract:
        obs["comportamiento"] = elegir(
            ["se_cierra_al_alejarse", "estable"], [0.7, 0.3])
    elif "se abre" in caract or "tiende a abrirse" in caract:
        obs["comportamiento"] = elegir(
            ["se_abre_con_tiempo", "estable"], [0.7, 0.3])
    elif "rápida" in caract or "instantánea" in caract or "golpe" in caract:
        obs["comportamiento"] = elegir(
            ["aparecio_de_golpe", "se_abre_con_tiempo"], [0.7, 0.3])
    elif "retracción" in defecto or "retraccion" in defecto:
        obs["comportamiento"] = elegir(
            ["estable", "se_cierra_al_alejarse"], [0.6, 0.4])
    else:
        obs["comportamiento"] = elegir(OPCIONES["comportamiento"])

    # ── Desprendimiento ────────────────────────────────────────────────
    if "desmoron" in caract or "desagreg" in defecto or "desmoron" in defecto:
        obs["desprendimiento"] = elegir(
            ["desmoronandose", "perdida_seccion", "recubrimiento_desprendido"],
            [0.5, 0.3, 0.2])
    elif "despren" in caract or "despren" in defecto:
        obs["desprendimiento"] = elegir(
            ["recubrimiento_desprendido", "perdida_seccion", "no_hay"],
            [0.5, 0.3, 0.2])
    elif "aplastamiento" in defecto:
        obs["desprendimiento"] = elegir(
            ["recubrimiento_desprendido", "desmoronandose", "no_hay"],
            [0.4, 0.3, 0.3])
    elif gravedad_pts >= 4:
        obs["desprendimiento"] = elegir(
            ["recubrimiento_desprendido", "no_hay", "perdida_seccion"],
            [0.4, 0.3, 0.3])
    else:
        obs["desprendimiento"] = elegir(
            ["no_hay", "recubrimiento_desprendido"], [0.7, 0.3])

    # ── Color ──────────────────────────────────────────────────────────
    if "corrosión" in defecto or "corrosion" in defecto or "oxid" in caract:
        obs["color"] = elegir(
            ["manchas_oxido", "normal", "oscurecido"], [0.6, 0.2, 0.2])
    elif "aluminosis" in defecto:
        obs["color"] = elegir(
            ["oscurecido", "normal"], [0.8, 0.2])
    elif "eflorescencia" in caract or "manchas blancas" in caract:
        obs["color"] = elegir(
            ["manchas_blancas", "normal"], [0.7, 0.3])
    elif "humedad" in causas or "agua" in causas:
        obs["color"] = elegir(
            ["manchas_blancas", "normal", "manchas_oxido"], [0.4, 0.3, 0.3])
    else:
        obs["color"] = elegir(
            ["normal", "manchas_oxido", "oscurecido", "manchas_blancas"],
            [0.5, 0.2, 0.15, 0.15])

    # ── Textura ────────────────────────────────────────────────────────
    if "desagreg" in defecto or "desmoron" in caract or "disgregado" in caract:
        obs["textura"] = elegir(
            ["disgregado", "poroso"], [0.7, 0.3])
    elif "poroso" in caract or "poco compacto" in causas:
        obs["textura"] = elegir(
            ["poroso", "disgregado", "normal"], [0.5, 0.3, 0.2])
    elif "humedad" in causas or "agua" in caract or "filtra" in caract:
        obs["textura"] = elegir(
            ["humedo_filtraciones", "poroso", "normal"], [0.5, 0.3, 0.2])
    else:
        obs["textura"] = elegir(
            ["normal", "poroso", "humedo_filtraciones"], [0.6, 0.25, 0.15])

    # ── Armadura visible ───────────────────────────────────────────────
    if "barras vistas" in caract or "barras" in caract and "perder" in caract:
        obs["armadura_visible"] = elegir(
            ["completamente", "parcialmente"], [0.6, 0.4])
    elif "corrosión" in defecto or "corrosion" in defecto:
        obs["armadura_visible"] = elegir(
            ["parcialmente", "completamente", "no"], [0.5, 0.3, 0.2])
    elif "despren" in caract or gravedad_pts >= 4:
        obs["armadura_visible"] = elegir(
            ["parcialmente", "no", "completamente"], [0.4, 0.4, 0.2])
    else:
        obs["armadura_visible"] = elegir(
            ["no", "parcialmente", "completamente"], [0.6, 0.3, 0.1])

    # ── Corrosión ──────────────────────────────────────────────────────
    if "corrosión" in defecto or "corrosion" in defecto:
        obs["corrosion"] = elegir(
            ["oxidada_visible", "perdida_seccion", "manchas_oxido"],
            [0.4, 0.3, 0.3])
    elif "marítim" in causas or "costero" in causas or "marino" in causas:
        obs["corrosion"] = elegir(
            ["manchas_oxido", "oxidada_visible", "no_visible"], [0.5, 0.3, 0.2])
    elif gravedad_pts >= 3 and ("armadura" in causas):
        obs["corrosion"] = elegir(
            ["manchas_oxido", "no_visible", "oxidada_visible"], [0.4, 0.3, 0.3])
    else:
        obs["corrosion"] = elegir(
            ["no_visible", "manchas_oxido"], [0.7, 0.3])

    # ── Estado estribos ────────────────────────────────────────────────
    if "estribo" in defecto or "cercos" in defecto:
        obs["estado_estribos"] = elegir(
            ["desplazados", "ausentes", "separados"], [0.4, 0.3, 0.3])
    elif "cercos" in causas or "estribos" in causas:
        obs["estado_estribos"] = elegir(
            ["separados", "desplazados", "ausentes", "correctos"],
            [0.4, 0.3, 0.2, 0.1])
    elif "armadura transversal" in causas:
        obs["estado_estribos"] = elegir(
            ["separados", "ausentes", "correctos"], [0.4, 0.3, 0.3])
    else:
        obs["estado_estribos"] = elegir(
            ["correctos", "separados", "desplazados"], [0.6, 0.25, 0.15])

    # ── Edad estructura ────────────────────────────────────────────────
    if "retracción" in defecto or "retraccion" in defecto:
        obs["edad_estructura"] = elegir(
            ["menos_5", "5_a_20"], [0.7, 0.3])
    elif "aluminosis" in defecto:
        obs["edad_estructura"] = elegir(
            ["20_a_50", "mas_50"], [0.5, 0.5])
    elif "corrosión" in defecto or "corrosion" in defecto:
        obs["edad_estructura"] = elegir(
            ["20_a_50", "mas_50", "5_a_20"], [0.4, 0.4, 0.2])
    elif "desencofrad" in causas or "durante el hormigonado" in causas:
        obs["edad_estructura"] = elegir(
            ["menos_5", "5_a_20"], [0.8, 0.2])
    else:
        obs["edad_estructura"] = elegir(OPCIONES["edad_estructura"])

    # ── Ambiente ───────────────────────────────────────────────────────
    if "marítim" in causas or "costero" in causas or "marino" in causas or "mar" in causas:
        obs["ambiente"] = elegir(
            ["costero_marino", "exterior_continental"], [0.7, 0.3])
    elif "químic" in causas or "agresiv" in causas or "industrial" in causas:
        obs["ambiente"] = elegir(
            ["industrial", "costero_marino", "exterior_continental"],
            [0.5, 0.3, 0.2])
    elif "humedad" in causas or "agua" in causas:
        obs["ambiente"] = elegir(
            ["exterior_continental", "costero_marino", "sumergido"],
            [0.4, 0.3, 0.3])
    elif "soleamiento" in causas or "sol" in causas:
        obs["ambiente"] = elegir(
            ["exterior_continental", "interior"], [0.7, 0.3])
    else:
        obs["ambiente"] = elegir(OPCIONES["ambiente"])

    # ── Velocidad aparición ────────────────────────────────────────────
    if "retracción plástica" in defecto:
        obs["velocidad_aparicion"] = elegir(
            ["durante_construccion", "primeras_semanas"], [0.7, 0.3])
    elif "retracción hidráulica" in defecto or "retraccion hidraulica" in defecto:
        obs["velocidad_aparicion"] = elegir(
            ["primeras_semanas", "primeros_meses"], [0.6, 0.4])
    elif "retracción térmica" in defecto:
        obs["velocidad_aparicion"] = elegir(
            ["primeros_meses", "anos_despues"], [0.6, 0.4])
    elif "desencofrad" in causas:
        obs["velocidad_aparicion"] = elegir(
            ["durante_construccion", "primeras_semanas"], [0.6, 0.4])
    elif "primeros meses" in causas or "primeras semanas" in causas:
        obs["velocidad_aparicion"] = elegir(
            ["primeras_semanas", "primeros_meses"], [0.5, 0.5])
    elif "corrosión" in defecto or "corrosion" in defecto:
        obs["velocidad_aparicion"] = elegir(
            ["anos_despues", "primeros_meses"], [0.7, 0.3])
    elif "servicio" in causas or "en servicio" in caract:
        obs["velocidad_aparicion"] = elegir(
            ["anos_despues", "primeros_meses"], [0.6, 0.4])
    elif "rápida" in caract or "instantánea" in caract:
        obs["velocidad_aparicion"] = elegir(
            ["subita_reciente", "durante_construccion"], [0.6, 0.4])
    else:
        obs["velocidad_aparicion"] = elegir(OPCIONES["velocidad_aparicion"])

    # ── Carga ──────────────────────────────────────────────────────────
    if "sobrecarga" in causas or "exceso de carga" in causas:
        obs["carga"] = elegir(
            ["sobrecarga", "normal"], [0.7, 0.3])
    elif "sismo" in causas or "impacto" in causas or "empuje" in causas:
        obs["carga"] = elegir(
            ["impacto_sismo", "sobrecarga", "normal"], [0.5, 0.3, 0.2])
    elif "descarga" in causas or "desencofrad" in causas:
        obs["carga"] = elegir(
            ["descarga", "normal"], [0.5, 0.5])
    else:
        obs["carga"] = elegir(
            ["normal", "sobrecarga"], [0.6, 0.4])

    # ── Intervención previa ────────────────────────────────────────────
    if "desencofrad" in causas or "fresco" in causas:
        obs["intervencion_previa"] = elegir(
            ["recien_desencofrada", "original"], [0.6, 0.4])
    elif "reparad" in causas or "reforzad" in causas:
        obs["intervencion_previa"] = elegir(
            ["reparada", "reforzada", "original"], [0.4, 0.3, 0.3])
    else:
        obs["intervencion_previa"] = elegir(
            ["original", "reparada", "reforzada", "recien_desencofrada"],
            [0.5, 0.2, 0.15, 0.15])

    # ── Deformación visible ────────────────────────────────────────────
    if "flecha" in defecto or "flecha" in caract:
        obs["deformacion_visible"] = elegir(
            ["flecha_abajo", "no"], [0.8, 0.2])
    elif "giro" in defecto or "inclinación" in defecto or "giro" in caract:
        obs["deformacion_visible"] = elegir(
            ["giro_inclinacion", "desplazamiento_lateral"], [0.7, 0.3])
    elif "aplastamiento" in defecto:
        obs["deformacion_visible"] = elegir(
            ["aplastamiento", "no", "flecha_abajo"], [0.5, 0.3, 0.2])
    elif "pandeo" in defecto:
        obs["deformacion_visible"] = elegir(
            ["desplazamiento_lateral", "giro_inclinacion"], [0.6, 0.4])
    elif "desplazamiento" in caract or "lateral" in caract:
        obs["deformacion_visible"] = elegir(
            ["desplazamiento_lateral", "no"], [0.7, 0.3])
    elif gravedad_pts >= 3:
        obs["deformacion_visible"] = elegir(
            ["flecha_abajo", "no", "desplazamiento_lateral"],
            [0.3, 0.4, 0.3])
    else:
        obs["deformacion_visible"] = elegir(
            ["no", "flecha_abajo"], [0.7, 0.3])

    # ── Sonido golpe ───────────────────────────────────────────────────
    if "desagreg" in defecto or "poroso" in caract or "disgregado" in caract:
        obs["sonido_golpe"] = elegir(
            ["hueco", "diferente_zonas"], [0.6, 0.4])
    elif "despren" in caract:
        obs["sonido_golpe"] = elegir(
            ["hueco", "diferente_zonas", "solido"], [0.4, 0.4, 0.2])
    elif gravedad_pts >= 3:
        obs["sonido_golpe"] = elegir(
            ["diferente_zonas", "solido", "hueco"], [0.4, 0.3, 0.3])
    else:
        obs["sonido_golpe"] = elegir(
            ["solido", "diferente_zonas", "hueco"], [0.5, 0.3, 0.2])

    # ── Presencia agua ─────────────────────────────────────────────────
    if "filtra" in caract or "humedad" in caract:
        obs["presencia_agua"] = elegir(
            ["filtracion_activa", "humedo"], [0.6, 0.4])
    elif "marítim" in causas or "costero" in causas or "salpicadura" in causas:
        obs["presencia_agua"] = elegir(
            ["salpicadura", "humedo", "seco"], [0.5, 0.3, 0.2])
    elif "agua" in causas:
        obs["presencia_agua"] = elegir(
            ["humedo", "filtracion_activa", "seco"], [0.5, 0.3, 0.2])
    else:
        obs["presencia_agua"] = elegir(
            ["seco", "humedo", "filtracion_activa", "salpicadura"],
            [0.5, 0.25, 0.15, 0.1])

    return obs


# ── Generar el dataset completo ────────────────────────────────────────────
print("Generando dataset sintético...")
filas = []

for pat in catalogo:
    n_variantes = np.random.randint(15, 31)  # entre 15 y 30
    for _ in range(n_variantes):
        obs = generar_observacion(pat)
        obs["defecto_numero"] = pat["numero"]
        filas.append(obs)

df = pd.DataFrame(filas)

# Reordenar columnas: features primero, target al final
features = [c for c in df.columns if c != "defecto_numero"]
df = df[features + ["defecto_numero"]]

# Aleatorizar orden de filas
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Dataset generado: {df.shape[0]} filas × {df.shape[1]} columnas")
print(f"Patologías únicas: {df['defecto_numero'].nunique()}")
print(f"\nDistribución de muestras por patología:")
print(df["defecto_numero"].value_counts().describe())

# Exportar
df.to_csv("dataset_patologias_sintetico.csv", index=False)
print("\n✓ Archivo guardado: dataset_patologias_sintetico.csv")
