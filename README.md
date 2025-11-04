
# Kinetic Theory Sim - Simulación de Gas Ideal 2D

## 🏗️ Arquitectura del Sistema

### Estructura Modular
```
KineticTheorySim/
├── gas_ideal/
│   ├── particula.py      # RF1: Modelado de partículas individuales
│   └── simulacion.py     # RF2: Gestión del sistema completo
├── analisis/
│   ├── test_gas.py       # RF3: Validación física automatizada
│   └── graficar.py       # RF4: Visualización científica
└── README.md
```

### Módulos Principales

#### 1. particula.py
**Responsabilidad**: Modelar el comportamiento individual de cada partícula

```python
class Particula:
    def mover(self, dt): ...           # Movimiento lineal: x = x + v·dt
    def colisionar_pared(self): ...    # Colisiones elásticas con paredes
    def energia_cinetica(self): ...    # E_k = ½·m·v²
```

#### 2. simulacion.py  
**Responsabilidad**: Coordinar la simulación completa y cálculos termodinámicos

```python
class Simulacion:
    def crear_gas(self): ...           # Inicializa N partículas aleatorias
    def paso(self): ...               # Avanza un paso temporal Δt
    def energia_total(self): ...       # Σ E_k de todas las partículas
    def temperatura(self): ...         # T = (Σ m·v²)/(2·N·k_B)
```

#### 3. test_gas.py
**Responsabilidad**: Validación automática de la física del sistema

```python
# Pruebas unitarias que verifican:
- Conservación de energía (<1% variación)
- Confinamiento en la caja
- Relación temperatura-velocidad
- Comportamiento de colisiones
```

#### 4. graficar.py
**Responsabilidad**: Visualización científica en tiempo real

```python
class Visualizador:
    def animar(self): ...              # Animación con matplotlib
    def update(self): ...              # Actualización en tiempo real
```

## 🚀 Utilización

### Instalación Rápida
```bash
git clone https://github.com/angielorenapm/KineticTheorySim.git
cd KineticTheorySim

python -m venv .venv
source .venv/bin/activate      # Linux / macOS
.venv\Scripts\activate         # Windows (PowerShell)

pip install -r requirements.txt
```

### Ejecución Básica

#### 1. Simulación con Visualización
```bash
python graficar.py
```
**Resultado**: Ventana con:
- Partículas moviéndose en tiempo real
- Trayectorias de las primeras 5 partículas
- Gráficas de energía y temperatura
- Estadísticas actualizadas

#### 2. Validación Física
```bash
python -m unittest test_gas.py -v
```

#### 3. Uso Programático
```python
from gas_ideal.simulacion import crear_gas, paso, energia_total

# Crear 20 partículas en caja 1e-8×1e-8 m
particulas = crear_gas(20, 1e-8, 1e-8, 800)

# Simular 100 pasos de 1e-12 segundos
for _ in range(100):
    paso(particulas, 1e-12, 1e-8, 1e-8)

# Calcular energía total
energia = energia_total(particulas)
print(f"Energía del sistema: {energia:.2e} J")
```

### Parámetros Configurables

#### Tamaño de Caja
```python
# Valores típicos: 1e-9 a 1e-8 metros
ancho = 5e-9
alto = 5e-9
```

#### Número de Partículas
```python
# Rango recomendado: 10-200 partículas
N = 20      # Para demostración
N = 50      # Para análisis
N = 200     # Máximo para tiempo real
```

#### Velocidad Media
```python
# Velocidades moleculares típicas
v_media = 500    # m/s (baja temperatura)
v_media = 1000   # m/s (temperatura ambiente)
v_media = 2000   # m/s (alta temperatura)
```

#### Paso Temporal
```python
# Para estabilidad numérica
dt = 1e-13   # Alto precisión
dt = 1e-12   # Balance precisión/rendimiento  
dt = 1e-11   # Máximo rendimiento
```

## 📊 Flujo de Datos

```
Inicialización → Simulación → Análisis → Visualización
     ↓              ↓           ↓           ↓
  crear_gas()     paso()    energía()    animar()
  10-200 part.   Δt=1e-12s  T, P, E_k   matplotlib
```

## 🔍 Monitoreo en Tiempo Real

La visualización muestra:
- **Posiciones actuales**: Partículas azules en la caja
- **Trayectorias**: Líneas de colores para las primeras 5 partículas  
- **Energía total**: Gráfica de conservación en tiempo real
- **Temperatura**: Evolución temporal de T efectiva
- **Estadísticas**: Velocidad media, dispersión, tiempo simulado

Autores: Angie Lorena Pineda [angielorenapm], Pablo Patiño Bonilla [ElitSpartan]
