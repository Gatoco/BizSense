"""
Generador de datasets realistas para NicheTech CL.
Tienda especializada en tecnología de nicho: ThinkPads, GPD, Sony Vaio,
GPUs RTX, Raspberry Pi, etc.

Este script genera:
- pyme_sales.csv: dataset principal de ventas semanales (~11.700 filas)
- ventas_semanales.csv: agregación semanal del negocio
- clientes_churn.csv: dataset para regresión logística
- segmentacion_clientes.csv: dataset para K-means
- precio_optimo.csv: curva precio vs demanda vs ganancia
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------------------
# Configuración general
# ---------------------------------------------------------------------------
N_SEMANAS = 156  # 3 años
N_CLIENTES = 5000
REGIONES = ["Metropolitana", "Valparaíso", "Biobío", "Araucanía", "Antofagasta"]

# Pesos de población por región (usado para distribuir clientes y ventas)
PESOS_REGION = {
    "Metropolitana": 0.45,
    "Valparaíso": 0.15,
    "Biobío": 0.18,
    "Araucanía": 0.12,
    "Antofagasta": 0.10,
}

# ---------------------------------------------------------------------------
# Catálogo de productos de nicho
# ---------------------------------------------------------------------------
PRODUCTOS = [
    # Laptops Compactas / UMPC
    {"id": "PROD-001", "nombre": "GPD Win 4", "categoria": "Laptops Compactas",
     "precio_base": 1_400_000, "costo_base": 1_100_000, "demanda_base": 8},
    {"id": "PROD-002", "nombre": "GPD Pocket 3", "categoria": "Laptops Compactas",
     "precio_base": 950_000, "costo_base": 750_000, "demanda_base": 12},

    # ThinkPad Business
    {"id": "PROD-003", "nombre": "ThinkPad T480", "categoria": "ThinkPad Business",
     "precio_base": 320_000, "costo_base": 220_000, "demanda_base": 35},
    {"id": "PROD-004", "nombre": "ThinkPad X1 Carbon Gen 6", "categoria": "ThinkPad Business",
     "precio_base": 680_000, "costo_base": 480_000, "demanda_base": 15},
    {"id": "PROD-005", "nombre": "ThinkPad X230", "categoria": "ThinkPad Business",
     "precio_base": 180_000, "costo_base": 120_000, "demanda_base": 28},

    # Laptops Vintage
    {"id": "PROD-006", "nombre": "Sony Vaio P VGN-P", "categoria": "Laptops Vintage",
     "precio_base": 520_000, "costo_base": 350_000, "demanda_base": 6},
    {"id": "PROD-007", "nombre": "Sony Vaio Z", "categoria": "Laptops Vintage",
     "precio_base": 450_000, "costo_base": 300_000, "demanda_base": 8},
    {"id": "PROD-008", "nombre": "Toshiba Libretto W100", "categoria": "Laptops Vintage",
     "precio_base": 380_000, "costo_base": 250_000, "demanda_base": 5},

    # Tarjetas Gráficas
    {"id": "PROD-009", "nombre": "NVIDIA RTX 4070", "categoria": "Tarjetas Gráficas",
     "precio_base": 720_000, "costo_base": 580_000, "demanda_base": 22},
    {"id": "PROD-010", "nombre": "NVIDIA RTX 4080", "categoria": "Tarjetas Gráficas",
     "precio_base": 1_100_000, "costo_base": 900_000, "demanda_base": 12},
    {"id": "PROD-011", "nombre": "NVIDIA RTX 4090", "categoria": "Tarjetas Gráficas",
     "precio_base": 2_000_000, "costo_base": 1_650_000, "demanda_base": 5},

    # Mini PCs / SBC
    {"id": "PROD-012", "nombre": "Raspberry Pi 5 8GB", "categoria": "Mini PCs / SBC",
     "precio_base": 95_000, "costo_base": 70_000, "demanda_base": 80},
    {"id": "PROD-013", "nombre": "Orange Pi 5", "categoria": "Mini PCs / SBC",
     "precio_base": 85_000, "costo_base": 62_000, "demanda_base": 45},
]

N_PRODUCTOS = len(PRODUCTOS)

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def generar_fechas_semanales(n_semanas):
    """Genera una lista de lunes consecutivos a partir del 3 de enero de 2022."""
    fecha_inicio = datetime(2022, 1, 3)
    return [fecha_inicio + timedelta(weeks=i) for i in range(n_semanas)]


def temperatura_semanal(semana, anio):
    """Temperatura aproximada para Chile (hemisferio sur) según estación."""
    # Pico de calor en verano (semanas 1-8 y 48-52), frío en invierno (semanas 22-35)
    base = 14.0
    verano = max(0, np.cos((semana - 1) * 2 * np.pi / 52))
    invierno = max(0, -np.cos((semana - 1) * 2 * np.pi / 52))
    temp = base + 8 * verano - 6 * invierno
    # Efecto año a año leve
    temp += (anio - 2022) * 0.3
    return round(temp + np.random.normal(0, 1.5), 1)


def es_feriado_relevante(semana, anio):
    """
    Retorna 1 si la semana tiene un feriado/evento relevante para ventas.
    - Semana 1: Año Nuevo
    - Semana 10: Vuelta a clases
    - Semana 22-23: CyberDays Mayo
    - Semana 37: Fiestas Patrias (18 de septiembre)
    - Semana 44: CyberDays Noviembre
    - Semana 51-52: Navidad / Año Nuevo
    """
    semanas_importantes = {1, 10, 22, 23, 37, 44, 51, 52}
    return 1 if semana in semanas_importantes else 0


def factor_estacionalidad(semana, categoria):
    """
    Factor multiplicativo según estacionalidad del producto.
    """
    base = 1.0

    # Picos generales: Navidad, CyberDays, Fiestas Patrias
    if semana in {1, 52}:
        base += 0.25  # Navidad / Año Nuevo
    if semana in {22, 23, 44}:
        base += 0.35  # CyberDays
    if semana == 37:
        base += 0.20  # Fiestas Patrias
    if semana == 10:
        base += 0.15  # Vuelta a clases

    # Efectos por categoría
    if categoria == "Tarjetas Gráficas":
        # Las GPUs tienen picos muy fuertes en CyberDays
        if semana in {22, 23, 44}:
            base += 0.30
    elif categoria == "Mini PCs / SBC":
        # Raspberry Pi muy demandada en vuelta a clases y proyectos de verano
        if semana in {1, 10, 52}:
            base += 0.25
    elif categoria == "Laptops Vintage":
        # Coleccionistas compran más en Navidad
        if semana in {51, 52}:
            base += 0.30
    elif categoria == "ThinkPad Business":
        # Empresas compran más en marzo y septiembre
        if semana in {10, 37}:
            base += 0.20

    return base


def variacion_precio(semana, anio, precio_base):
    """Simula cambios de precio por inflación, restock y ofertas."""
    # Inflación acumulada ~8% anual
    inflacion = 1 + (anio - 2022) * 0.08
    precio = precio_base * inflacion

    # Descuentos en CyberDays
    if semana in {22, 23, 44}:
        precio *= np.random.uniform(0.88, 0.96)

    # Pequeñas fluctuaciones semanales
    precio *= np.random.uniform(0.98, 1.02)

    return int(round(precio, -3))


def calcular_unidades_vendidas(demanda_base, precio_actual, precio_base, costo,
                                marketing, es_feriado, factor_estacion,
                                temperatura, categoria):
    """
    Modelo de demanda realista.
    - Elasticidad precio: subir precio reduce demanda
    - Marketing: más inversión aumenta demanda (rendimientos decrecientes)
    - Feriados y estacionalidad
    - Ruido aleatorio
    """
    # Elasticidad al precio (productos caros son más elásticos)
    if categoria in {"Tarjetas Gráficas", "Laptops Compactas", "Laptops Vintage"}:
        elasticidad = 0.000004
    else:
        elasticidad = 0.000002

    cambio_precio = (precio_actual - precio_base) / precio_base
    factor_precio = max(0.3, 1 - elasticidad * precio_base * abs(cambio_precio) * 10)

    # Marketing: retorno decreciente
    marketing_pct = marketing / max(precio_actual, 1)
    factor_marketing = 1 + 0.5 * np.log1p(marketing_pct * 100)

    # Temperatura: leve efecto (menos ventas en pleno invierno para algunas categorías)
    factor_temperatura = 1 + (temperatura - 14) * 0.005

    # Feriado
    factor_feriado = 1 + 0.25 * es_feriado

    demanda = (
        demanda_base
        * factor_estacion
        * factor_precio
        * factor_marketing
        * factor_temperatura
        * factor_feriado
    )

    # Ruido realista (Poisson-like)
    demanda = np.random.poisson(max(0, demanda))

    # Outliers ocasionales: stock agotado o rebaja especial
    if np.random.random() < 0.01:
        demanda = max(0, int(demanda * np.random.uniform(0.2, 0.5)))
    elif np.random.random() < 0.01:
        demanda = int(demanda * np.random.uniform(1.5, 2.0))

    return int(demanda)


# ---------------------------------------------------------------------------
# Generación del dataset principal de ventas
# ---------------------------------------------------------------------------

def generar_pyme_sales():
    fechas = generar_fechas_semanales(N_SEMANAS)
    registros = []

    for i, fecha in enumerate(fechas):
        semana = i + 1
        anio = fecha.year
        mes = fecha.month
        trimestre = (mes - 1) // 3 + 1
        temp = temperatura_semanal(semana, anio)
        feriado = es_feriado_relevante(semana, anio)

        for producto in PRODUCTOS:
            categoria = producto["categoria"]
            factor_estacion = factor_estacionalidad(semana, categoria)
            precio = variacion_precio(semana, anio, producto["precio_base"])
            costo = int(producto["costo_base"] * (1 + (anio - 2022) * 0.06))

            for region in REGIONES:
                # Marketing semanal varía por región y categoría
                marketing_base = precio * np.random.uniform(0.02, 0.08)
                if feriado:
                    marketing_base *= np.random.uniform(1.3, 2.0)
                marketing = int(round(marketing_base, -3))

                # Ajuste regional de demanda
                demanda_base_regional = producto["demanda_base"] * PESOS_REGION[region]

                unidades = calcular_unidades_vendidas(
                    demanda_base_regional,
                    precio,
                    producto["precio_base"],
                    costo,
                    marketing,
                    feriado,
                    factor_estacion,
                    temp,
                    categoria,
                )

                # Clientes nuevos y recurrentes derivados de las unidades vendidas
                if unidades == 0:
                    clientes_nuevos = 0
                    clientes_recurrentes = 0
                else:
                    clientes_totales = max(1, int(unidades * np.random.uniform(0.8, 1.5)))
                    clientes_nuevos = int(clientes_totales * np.random.uniform(0.25, 0.45))
                    clientes_recurrentes = clientes_totales - clientes_nuevos

                ingreso = precio * unidades
                ganancia = (precio - costo) * unidades
                stock = max(0, int(unidades * np.random.uniform(1.5, 4.0)))

                registros.append({
                    "fecha": fecha.strftime("%Y-%m-%d"),
                    "semana": semana,
                    "mes": mes,
                    "trimestre": trimestre,
                    "anio": anio,
                    "producto_id": producto["id"],
                    "producto_nombre": producto["nombre"],
                    "categoria": categoria,
                    "region": region,
                    "precio_unitario": precio,
                    "costo_unitario": costo,
                    "stock_disponible": stock,
                    "unidades_vendidas": unidades,
                    "gasto_marketing": marketing,
                    "es_feriado": feriado,
                    "temperatura": temp,
                    "clientes_nuevos": clientes_nuevos,
                    "clientes_recurrentes": clientes_recurrentes,
                    "ingreso_total": ingreso,
                    "ganancia_total": ganancia,
                })

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Generación del dataset de agregación semanal
# ---------------------------------------------------------------------------

def generar_ventas_semanales(df_sales):
    return df_sales.groupby(["semana", "anio"], as_index=False).agg({
        "ingreso_total": "sum",
        "ganancia_total": "sum",
        "unidades_vendidas": "sum",
        "gasto_marketing": "sum",
        "clientes_nuevos": "sum",
        "clientes_recurrentes": "sum",
    }).rename(columns={
        "gasto_marketing": "gasto_marketing_total",
        "clientes_nuevos": "clientes_nuevos_total",
        "clientes_recurrentes": "clientes_recurrentes_total",
    })


# ---------------------------------------------------------------------------
# Generación del dataset de clientes para churn
# ---------------------------------------------------------------------------

def generar_clientes_churn():
    clientes = []

    for i in range(N_CLIENTES):
        region = np.random.choice(REGIONES, p=list(PESOS_REGION.values()))
        edad = int(np.clip(np.random.normal(34, 10), 18, 70))
        meses_cliente = int(np.clip(np.random.exponential(24) + 3, 1, 120))

        # Ingreso anual con distribución más realista (evita saturación en el tope)
        ingreso_anual_estimado = int(np.random.lognormal(16.5, 0.45))

        # Gasto mensual correlacionado con ingreso (proporción razonable del presupuesto)
        proporcion_gasto = np.random.beta(2, 8) * 0.04 + 0.002
        gasto_mensual = int(ingreso_anual_estimado * proporcion_gasto + np.random.normal(0, 8000))
        gasto_mensual = max(15000, min(600_000, gasto_mensual))

        frecuencia = max(0.1, np.random.exponential(0.8))
        tickets_soporte = int(np.random.poisson(0.7))
        ultima_compra_dias = int(np.clip(np.random.exponential(45), 1, 365))

        # Probabilidad de quedarse: más meses, más gasto, menos tickets, compra reciente
        logit = (
            0.02 * meses_cliente
            + 0.00001 * gasto_mensual
            + 0.5 * frecuencia
            - 0.4 * tickets_soporte
            - 0.015 * ultima_compra_dias
            + np.random.normal(0, 0.8)
        )
        prob = 1 / (1 + np.exp(-logit))
        se_queda = int(np.random.random() < prob)

        categoria_preferida = np.random.choice([
            "Laptops Compactas", "ThinkPad Business", "Laptops Vintage",
            "Tarjetas Gráficas", "Mini PCs / SBC"
        ])

        clientes.append({
            "cliente_id": f"CLI-{i+1:05d}",
            "edad": edad,
            "meses_cliente": meses_cliente,
            "gasto_mensual_promedio": gasto_mensual,
            "frecuencia_compra_mensual": round(frecuencia, 2),
            "tickets_soporte": tickets_soporte,
            "categoria_preferida": categoria_preferida,
            "region": region,
            "ultima_compra_dias": ultima_compra_dias,
            "se_queda": se_queda,
        })

    return pd.DataFrame(clientes)


# ---------------------------------------------------------------------------
# Generación del dataset de segmentación de clientes
# ---------------------------------------------------------------------------

def generar_segmentacion_clientes(df_churn):
    segmentacion = []

    for _, row in df_churn.iterrows():
        # Convertir métricas mensuales a anuales
        frecuencia_anual = max(1, int(row["frecuencia_compra_mensual"] * 12 + np.random.normal(0, 1)))
        gasto_anual = int(row["gasto_mensual_promedio"] * 12 * np.random.uniform(0.85, 1.15))
        ticket_promedio = int(gasto_anual / max(1, frecuencia_anual))

        # Ingreso anual coherente con el gasto mensual (proporción realista)
        factor_ingreso = np.random.uniform(10, 30)
        ingreso_anual_estimado = int(row["gasto_mensual_promedio"] * 12 * factor_ingreso)

        segmentacion.append({
            "cliente_id": row["cliente_id"],
            "edad": row["edad"],
            "ingreso_anual_estimado": ingreso_anual_estimado,
            "gasto_anual_total": gasto_anual,
            "frecuencia_compra_anual": frecuencia_anual,
            "meses_cliente": row["meses_cliente"],
            "ticket_promedio": ticket_promedio,
            "region": row["region"],
        })

    return pd.DataFrame(segmentacion)


# ---------------------------------------------------------------------------
# Generación del dataset de precio óptimo
# ---------------------------------------------------------------------------

def generar_precio_optimo():
    """
    Genera curva de demanda y ganancia para un producto de nicho.
    Usamos la ThinkPad T480 como ejemplo.
    """
    costo = 220_000
    # Función de demanda: D(p) = D_max - k * p
    # Si p = 320.000, demanda ~ 35 unidades semanales
    # Si p sube, demanda baja
    k = 35 / 320_000 * 1.2  # elasticidad
    d_max = 70  # demanda máxima teórica

    precios = np.arange(220_000, 460_000, 5_000)
    registros = []

    for precio in precios:
        demanda_base = max(0, d_max - k * precio)
        demanda = max(0, int(np.random.poisson(demanda_base)))
        ganancia = (precio - costo) * demanda

        registros.append({
            "precio": int(precio),
            "demanda_estimada": demanda,
            "ganancia": int(ganancia),
        })

    return pd.DataFrame(registros)


# ---------------------------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------------------------

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print("Generando pyme_sales.csv...")
    df_sales = generar_pyme_sales()
    df_sales.to_csv(os.path.join(base_dir, "pyme_sales.csv"), index=False)
    print(f"  -> {len(df_sales):,} filas generadas")

    print("Generando ventas_semanales.csv...")
    df_semanal = generar_ventas_semanales(df_sales)
    df_semanal.to_csv(os.path.join(base_dir, "ventas_semanales.csv"), index=False)
    print(f"  -> {len(df_semanal):,} filas generadas")

    print("Generando clientes_churn.csv...")
    df_churn = generar_clientes_churn()
    df_churn.to_csv(os.path.join(base_dir, "clientes_churn.csv"), index=False)
    print(f"  -> {len(df_churn):,} filas generadas")

    print("Generando segmentacion_clientes.csv...")
    df_segmentacion = generar_segmentacion_clientes(df_churn)
    df_segmentacion.to_csv(os.path.join(base_dir, "segmentacion_clientes.csv"), index=False)
    print(f"  -> {len(df_segmentacion):,} filas generadas")

    print("Generando precio_optimo.csv...")
    df_precio = generar_precio_optimo()
    df_precio.to_csv(os.path.join(base_dir, "precio_optimo.csv"), index=False)
    print(f"  -> {len(df_precio):,} filas generadas")

    print("\nTodos los datasets fueron generados exitosamente.")


if __name__ == "__main__":
    main()
