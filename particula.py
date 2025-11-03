import numpy as np

class Particula:
    """
    Representa una partícula de gas ideal en una caja 2D.
    
    Modela el movimiento de partículas con colisiones elásticas contra las paredes
    y entre partículas, conservando energía y momento.
    
    Parámetros:
        id (int): Identificador único de la partícula
        masa (float): Masa de la partícula en kg
        radio (float): Radio de la partícula en m
        posicion (array): Posición inicial [x, y] en m
        velocidad (array): Velocidad inicial [vx, vy] en m/s
    """
    
    def __init__(self, id, masa, radio, posicion, velocidad):
        self.id = id
        self.masa = masa  # kg
        self.radio = radio  # m
        self.posicion = np.array(posicion, dtype=float)  # [x, y] (m)
        self.velocidad = np.array(velocidad, dtype=float)  # [vx, vy] (m/s)
        self.fuerza = np.zeros(2)  # Fuerza neta [Fx, Fy] (N)
        
        # Historial para análisis
        self.historial_posiciones = [self.posicion.copy()]
        self.historial_velocidades = [self.velocidad.copy()]
    
    def avanzar(self, dt):
        """
        Avanza la partícula en el tiempo usando integración de Verlet.
        
        Ecuaciones:
            v_{n+1/2} = v_n + (F_n / m) * (Δt/2)
            x_{n+1} = x_n + v_{n+1/2} * Δt
            v_{n+1} = v_{n+1/2} + (F_{n+1} / m) * (Δt/2)
            
        Unidades:
            dt: segundos (s)
            x: metros (m)
            v: metros/segundo (m/s)
            F: newtons (N)
        """
        # Medio paso de velocidad
        v_medio = self.velocidad + (self.fuerza / self.masa) * (dt / 2)
        
        # Paso completo de posición
        self.posicion += v_medio * dt
        
        # Actualizar fuerza (se calculará externamente)
        # Para el siguiente medio paso de velocidad
        
        # Guardar historial
        self.historial_posiciones.append(self.posicion.copy())
        self.historial_velocidades.append(self.velocidad.copy())
    
    def energia_cinetica(self):
        """
        Calcula la energía cinética de la partícula.
        
        Ecuación: E_k = ½ · m · v²
        Unidades: Energía en julios (J)
        """
        v_cuad = np.dot(self.velocidad, self.velocidad)
        return 0.5 * self.masa * v_cuad
    
    def momento_lineal(self):
        """
        Calcula el momento lineal de la partícula.
        
        Ecuación: p = m · v
        Unidades: Momento en kg·m/s
        """
        return self.masa * self.velocidad
    
    def temperatura_equivalente(self, k_B=1.38e-23):
        """
        Calcula la temperatura equivalente según teoría cinética de gases.
        
        Ecuación: T = (m · v²) / (2 · k_B)
        Para 2D: T = (m · v²) / (2 · k_B)
        
        Unidades:
            T: kelvin (K)
            k_B: constante de Boltzmann (J/K)
        """
        v_cuad = np.dot(self.velocidad, self.velocidad)
        return (self.masa * v_cuad) / (2 * k_B)
    
    def distancia_a(self, otra_particula):
        """
        Calcula la distancia entre centros de dos partículas.
        """
        return np.linalg.norm(self.posicion - otra_particula.posicion)
    
    def colisiona_con(self, otra_particula):
        """
        Verifica si hay colisión con otra partícula.
        
        Condición: distancia ≤ suma de radios
        """
        distancia = self.distancia_a(otra_particula)
        return distancia <= (self.radio + otra_particula.radio)
