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
    
    def inicializar_particulas_aleatorias(self):
        """
        Inicializa partículas con posiciones y velocidades aleatorias.
        
        Las velocidades siguen distribución de Maxwell-Boltzmann
        para una temperatura inicial dada.
        """
        temperatura_inicial = 300  # K (temperatura ambiente)
        
        for i in range(self.n_particulas):
            # Posición aleatoria evitando superposiciones
            pos_valida = False
            while not pos_valida:
                posicion = np.random.uniform(self.radio_particula, 
                                           self.tamaño_caja - self.radio_particula, 2)
                pos_valida = self._posicion_valida(posicion, i)
            
            # Velocidad con distribución Maxwell-Boltzmann
            k_B = 1.38e-23
            sigma = np.sqrt(k_B * temperatura_inicial / self.masa_particula)
            velocidad = np.random.normal(0, sigma, 2)
            
            particula = Particula(i, self.masa_particula, self.radio_particula,
                                posicion, velocidad)
            self.particulas.append(particula)
    
    def _posicion_valida(self, posicion, id_actual):
        """
        Verifica que la posición no cause superposición con partículas existentes.
        """
        for i, p in enumerate(self.particulas):
            if i >= id_actual:
                break
            dist = np.linalg.norm(posicion - p.posicion)
            if dist < 2 * self.radio_particula:
                return False
        return True
    
    def aplicar_condiciones_frontera(self):
        """
        Aplica colisiones elásticas con las paredes de la caja.
        """
        for particula in self.particulas:
            # Pared izquierda (x = 0)
            if particula.posicion[0] - particula.radio <= 0:
                particula.velocidad[0] = abs(particula.velocidad[0])
                particula.posicion[0] = particula.radio
            
            # Pared derecha (x = L)
            if particula.posicion[0] + particula.radio >= self.tamaño_caja:
                particula.velocidad[0] = -abs(particula.velocidad[0])
                particula.posicion[0] = self.tamaño_caja - particula.radio
            
            # Pared inferior (y = 0)
            if particula.posicion[1] - particula.radio <= 0:
                particula.velocidad[1] = abs(particula.velocidad[1])
                particula.posicion[1] = particula.radio
            
            # Pared superior (y = L)
            if particula.posicion[1] + particula.radio >= self.tamaño_caja:
                particula.velocidad[1] = -abs(particula.velocidad[1])
                particula.posicion[1] = self.tamaño_caja - particula.radio
    
    def detectar_colisiones(self):
        """
        Detecta y resuelve colisiones entre partículas.
        
        Usa colisiones elásticas conservando energía y momento.
        """
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
    
    def calcular_energia_total(self):
        """
        Calcula la energía cinética total del sistema.
        
        Ecuación: E_total = Σ ½·m_i·v_i²
        Unidades: Energía en julios (J)
        """
        energia_total = 0.0
        for particula in self.particulas:
            energia_total += particula.energia_cinetica()
        return energia_total
    
    def calcular_temperatura(self):
        """
        Calcula la temperatura del gas usando teoría cinética.
        
        Para 2D: T = (Σ m_i·v_i²) / (2 · N · k_B)
        Unidades: Temperatura en kelvin (K)
        """
        k_B = 1.38e-23  # J/K
        suma_v_cuad = 0.0
        
        for particula in self.particulas:
            v_cuad = np.dot(particula.velocidad, particula.velocidad)
            suma_v_cuad += particula.masa * v_cuad
        
        if self.n_particulas > 0:
            return suma_v_cuad / (2 * self.n_particulas * k_B)
        return 0.0
    
    def calcular_presion(self):
        """
        Calcula la presión ejercida sobre las paredes.
        
        Basado en el teorema de equipartición y ecuación de estado.
        Para 2D: P = (N · k_B · T) / A
        Unidades: Presión en pascales (Pa)
        """
        k_B = 1.38e-23
        area = self.tamaño_caja ** 2
        temperatura = self.calcular_temperatura()
        
        return (self.n_particulas * k_B * temperatura) / area
    
    def avanzar(self, pasos=1):
        """
        Avanza la simulación un número determinado de pasos.
        """
        for _ in range(pasos):
            # Resetear fuerzas
            for particula in self.particulas:
                particula.fuerza = np.zeros(2)
            
            # Aplicar condiciones de frontera
            self.aplicar_condiciones_frontera()
            
            # Detectar y resolver colisiones
            self.detectar_colisiones()
            
            # Avanzar partículas
            for particula in self.particulas:
                particula.avanzar(self.dt)
            
            # Actualizar tiempo
            self.tiempo_actual += self.dt
            
            # Guardar datos para análisis
            self.historial_energia.append(self.calcular_energia_total())
            self.historial_temperatura.append(self.calcular_temperatura())
            self.historial_presion.append(self.calcular_presion())
    
    def obtener_estadisticas(self):
        """
        Retorna estadísticas importantes del sistema.
        """
        return {
            'energia_total': self.calcular_energia_total(),
            'temperatura': self.calcular_temperatura(),
            'presion': self.calcular_presion(),
            'velocidad_promedio': np.mean([np.linalg.norm(p.velocidad) 
                                         for p in self.particulas]),
            'tiempo': self.tiempo_actual
        }
