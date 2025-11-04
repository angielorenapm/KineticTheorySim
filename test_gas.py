import unittest
import numpy as np
from particula import Particula
from simulacion import Simulacion

class TestGasIdeal(unittest.TestCase):
    """Pruebas unitarias para verificar la física del gas ideal."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.simulacion = Simulacion(tamaño_caja=1e-8, n_particulas=10)
        self.simulacion.inicializar_particulas_aleatorias()
    
    def test_conservacion_energia(self):
        """Verifica que la energía total se conserve."""
        energia_inicial = self.simulacion.calcular_energia_total()
        
        # Avanzar simulación
        self.simulacion.avanzar(pasos=100)
        energia_final = self.simulacion.calcular_energia_total()
        
        # La energía debe conservarse (con pequeña tolerancia numérica)
        tolerancia = 1e-15
        self.assertAlmostEqual(energia_inicial, energia_final, delta=tolerancia)
    
    def test_conservacion_momento(self):
        """Verifica que el momento lineal total se conserve."""
        momento_inicial = np.sum([p.momento_lineal() for p in self.simulacion.particulas], axis=0)
        
        self.simulacion.avanzar(pasos=50)
        momento_final = np.sum([p.momento_lineal() for p in self.simulacion.particulas], axis=0)
        
        # El momento total debe conservarse
        tolerancia = 1e-20
        for i in range(2):
            self.assertAlmostEqual(momento_inicial[i], momento_final[i], delta=tolerancia)
    
    def test_colision_elastica(self):
        """Verifica que las colisiones sean elásticas."""
        # Crear dos partículas en colisión frontal
        p1 = Particula(1, masa=1e-26, radio=1e-10, posicion=[0.2e-8, 0.5e-8], 
                       velocidad=[100, 0])
        p2 = Particula(2, masa=1e-26, radio=1e-10, posicion=[0.3e-8, 0.5e-8], 
                       velocidad=[-100, 0])
        
        # Energía cinética antes de la colisión
        energia_inicial = p1.energia_cinetica() + p2.energia_cinetica()
        
        # Simular colisión
        sim = Simulacion(tamaño_caja=1e-8, n_particulas=0)
        sim.particulas = [p1, p2]
        sim.detectar_colisiones()
        
        # Energía cinética después de la colisión
        energia_final = p1.energia_cinetica() + p2.energia_cinetica()
        
        self.assertAlmostEqual(energia_inicial, energia_final, delta=1e-20)
    
    def test_relacion_temperatura_velocidad(self):
        """Verifica la relación entre temperatura y velocidad promedio."""
        k_B = 1.38e-23
        
        # Para un gas ideal en 2D: T = (m * v_prom²) / (2 * k_B)
        temperatura_medida = self.simulacion.calcular_temperatura()
        
        # Calcular temperatura a partir de velocidad promedio
        velocidades = [np.linalg.norm(p.velocidad) for p in self.simulacion.particulas]
        v_prom_cuad = np.mean(np.square(velocidades))
        temperatura_calculada = (self.simulacion.masa_particula * v_prom_cuad) / (2 * k_B)
        
        # Deben ser aproximadamente iguales
        self.assertAlmostEqual(temperatura_medida, temperatura_calculada, delta=1.0)
    
    def test_ecuacion_estado(self):
        """Verifica la ecuación de estado para gas ideal en 2D."""
        stats = self.simulacion.obtener_estadisticas()
        
        # Para gas ideal 2D: P * A = N * k_B * T
        k_B = 1.38e-23
        area = self.simulacion.tamaño_caja ** 2
        presion_calculada = (self.simulacion.n_particulas * k_B * stats['temperatura']) / area
        
        self.assertAlmostEqual(stats['presion'], presion_calculada, delta=1e-5)

if __name__ == '__main__':
    unittest.main()
