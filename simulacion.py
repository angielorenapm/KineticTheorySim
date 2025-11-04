import numpy as np
from particula import Particula

class Simulacion:
    """
    Simula un gas ideal confinado en una caja cuadrada 2D.
    
    Implementa colisiones elásticas entre partículas y con las paredes,
    conservando energía y momento lineal total.
    
    Parámetros:
        tamaño_caja (float): Tamaño de la caja cuadrada en metros (m)
        n_particulas (int): Número de partículas en la simulación  
        dt (float): Paso temporal de integración en segundos (s)
    """
    
    def __init__(self, tamaño_caja, n_particulas, dt=1e-12):
        self.tamaño_caja = tamaño_caja  # m
        self.n_particulas = n_particulas
        self.dt = dt  # s
        self.particulas = []
        self.tiempo_actual = 0.0
        
        # Parámetros físicos típicos para gas ideal
        self.masa_particula = 4.65e-26  # kg (aproximadamente masa de molécula de N2)
        self.radio_particula = 1e-10    # m (radio molecular típico)
        
        # Historial para análisis
        self.historial_energia = []
        self.historial_temperatura = []
        self.historial_presion = []
    
    def crear_gas(self, N, ancho, alto, v_media):
        """
        Crea N partículas con posiciones y velocidades aleatorias.
        
        Parámetros:
            N (int): Número de partículas
            ancho (float): Ancho de la caja en metros
            alto (float): Alto de la caja en metros
            v_media (float): Velocidad media de las partículas en m/s
        """
        self.n_particulas = N
        self.tamaño_caja = ancho  # Asumimos caja cuadrada
        
        for i in range(N):
            # Posición aleatoria evitando superposiciones
            pos_valida = False
            while not pos_valida:
                posicion = np.random.uniform(self.radio_particula, 
                                           ancho - self.radio_particula, 2)
                pos_valida = self._posicion_valida(posicion, i)
            
            # Velocidad con dirección aleatoria y magnitud alrededor de v_media
            angulo = np.random.uniform(0, 2*np.pi)
            # Distribución de Maxwell-Boltzmann simplificada
            sigma = v_media / np.sqrt(2)  # Para distribución normal
            magnitud = np.random.normal(v_media, sigma/3)
            magnitud = max(0.1 * v_media, min(3 * v_media, magnitud))  # Limitar extremos
            velocidad = magnitud * np.array([np.cos(angulo), np.sin(angulo)])
            
            particula = Particula(i, self.masa_particula, self.radio_particula,
                                posicion, velocidad)
            self.particulas.append(particula)
    
    def _posicion_valida(self, posicion, id_actual):
        """Verifica que la posición no cause superposición."""
        for i, p in enumerate(self.particulas):
            if i >= id_actual:
                break
            dist = np.linalg.norm(posicion - p.posicion)
            if dist < 2 * self.radio_particula:
                return False
        return True
    
    def paso(self, particulas, dt, ancho, alto):
        """
        Actualiza las posiciones y maneja colisiones para un paso temporal.
        
        Parámetros:
            particulas (list): Lista de partículas
            dt (float): Paso temporal en segundos
            ancho (float): Ancho de la caja en metros
            alto (float): Alto de la caja en metros
        """
        # Primero mover todas las partículas
        for particula in particulas:
            particula.mover(dt)
            particula.colisionar_pared(ancho, alto)
        
        # Luego detectar y resolver colisiones entre partículas
        self.detectar_colisiones()
    
    def aplicar_condiciones_frontera(self):
        """Aplica colisiones elásticas con las paredes de la caja."""
        for particula in self.particulas:
            particula.colisionar_pared(self.tamaño_caja, self.tamaño_caja)
    
    def detectar_colisiones(self):
        """Detecta y resuelve colisiones entre partículas."""
        for i in range(len(self.particulas)):
            for j in range(i + 1, len(self.particulas)):
                p1 = self.particulas[i]
                p2 = self.particulas[j]
                
                if p1.colisiona_con(p2):
                    self._resolver_colision(p1, p2)
    
    def _resolver_colision(self, p1, p2):
        """
        Resuelve colisión elástica entre dos partículas.
        
        Ecuaciones de conservación:
            Conservación de momento: m1·v1 + m2·v2 = m1·v1' + m2·v2'
            Conservación de energía: ½m1·v1² + ½m2·v2² = ½m1·v1'² + ½m2·v2'²
        """
        # Vector de diferencia de posición
        r12 = p2.posicion - p1.posicion
        distancia = np.linalg.norm(r12)
        
        if distancia == 0:  # Evitar división por cero
            return
        
        # Vector unitario normal
        n = r12 / distancia
        
        # Velocidades relativas
        v1 = p1.velocidad
        v2 = p2.velocidad
        
        # Proyecciones de velocidad en dirección normal
        v1n = np.dot(v1, n)
        v2n = np.dot(v2, n)
        
        # Coeficientes para colisión elástica
        m1, m2 = p1.masa, p2.masa
        masa_total = m1 + m2
        
        # Nuevas velocidades normales
        v1n_nueva = (v1n * (m1 - m2) + 2 * m2 * v2n) / masa_total
        v2n_nueva = (v2n * (m2 - m1) + 2 * m1 * v1n) / masa_total
        
        # Actualizar velocidades
        p1.velocidad += (v1n_nueva - v1n) * n
        p2.velocidad += (v2n_nueva - v2n) * n
        
        # Separar partículas para evitar superposición
        superposicion = (p1.radio + p2.radio) - distancia
        if superposicion > 0:
            correccion = superposicion * 0.5
            p1.posicion -= correccion * n
            p2.posicion += correccion * n
    
    def energia_total(self, particulas=None):
        """
        Calcula la energía cinética total del sistema.
        
        Ecuación: E_total = Σ ½·m_i·v_i²
        Unidades: Energía en julios (J)
        """
        if particulas is None:
            particulas = self.particulas
            
        return sum(p.energia_cinetica() for p in particulas)
    
    def temperatura(self, particulas=None, k_B=1.38e-23):
        """
        Calcula la temperatura del gas usando teoría cinética.
        
        Para 2D: T = (Σ m_i·v_i²) / (2 · N · k_B)
        Unidades: Temperatura en kelvin (K)
        """
        if particulas is None:
            particulas = self.particulas
            
        if not particulas:
            return 0.0
            
        suma_v_cuad = 0.0
        for particula in particulas:
            v_cuad = np.dot(particula.velocidad, particula.velocidad)
            suma_v_cuad += particula.masa * v_cuad
        
        return suma_v_cuad / (2 * len(particulas) * k_B)
    
    def calcular_temperatura(self):
        """Alias para temperatura() para mantener compatibilidad"""
        return self.temperatura(self.particulas)
    
    def calcular_presion(self):
        """
        Calcula la presión ejercida sobre las paredes.
        
        Para 2D: P = (N · k_B · T) / A
        Unidades: Presión en pascales (Pa)
        """
        k_B = 1.38e-23
        area = self.tamaño_caja ** 2
        temperatura = self.temperatura()
        
        return (self.n_particulas * k_B * temperatura) / area
    
    def avanzar(self, pasos=1):
        """
        Avanza la simulación un número determinado de pasos.
        """
        for _ in range(pasos):
            # Aplicar condiciones de frontera
            self.aplicar_condiciones_frontera()
            
            # Detectar y resolver colisiones
            self.detectar_colisiones()
            
            # Mover partículas (usando el método simple)
            for particula in self.particulas:
                particula.mover(self.dt)
            
            # Actualizar tiempo
            self.tiempo_actual += self.dt
            
            # Guardar datos para análisis
            self.historial_energia.append(self.energia_total())
            self.historial_temperatura.append(self.temperatura())
            self.historial_presion.append(self.calcular_presion())
    
    def obtener_estadisticas(self):
        """Retorna estadísticas importantes del sistema."""
        return {
            'energia_total': self.energia_total(),
            'temperatura': self.temperatura(),
            'presion': self.calcular_presion(),
            'velocidad_promedio': np.mean([np.linalg.norm(p.velocidad) 
                                         for p in self.particulas]),
            'tiempo': self.tiempo_actual
        }


# Funciones adicionales para cumplir con requerimientos específicos
def crear_gas(N, ancho, alto, v_media):
    """
    Crea un gas con N partículas en una caja de dimensiones ancho x alto.
    
    Parámetros:
        N (int): Número de partículas
        ancho (float): Ancho de la caja en metros
        alto (float): Alto de la caja en metros  
        v_media (float): Velocidad media de las partículas en m/s
    """
    sim = Simulacion(ancho, N)
    sim.crear_gas(N, ancho, alto, v_media)
    return sim.particulas

def paso(particulas, dt, ancho, alto):
    """
    Avanza la simulación un paso temporal.
    
    Parámetros:
        particulas (list): Lista de partículas
        dt (float): Paso temporal en segundos
        ancho (float): Ancho de la caja en metros
        alto (float): Alto de la caja en metros
    """
    for particula in particulas:
        particula.mover(dt)
        particula.colisionar_pared(ancho, alto)

def energia_total(particulas):
    """Calcula la energía cinética total del sistema."""
    return sum(p.energia_cinetica() for p in particulas)

def temperatura(particulas, k_B=1.38e-23):
    """Calcula la temperatura efectiva del gas."""
    if not particulas:
        return 0.0
        
    suma_v_cuad = 0.0
    for particula in particulas:
        v_cuad = np.dot(particula.velocidad, particula.velocidad)
        suma_v_cuad += particula.masa * v_cuad
    
    return suma_v_cuad / (2 * len(particulas) * k_B)