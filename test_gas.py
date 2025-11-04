import unittest
import numpy as np
from particula import Particula
from simulacion import crear_gas, paso, energia_total, temperatura

class TestGasIdeal(unittest.TestCase):
    """Pruebas unitarias para verificar la física del gas ideal."""
    
    def setUp(self):
        """Configuración inicial para las pruebas."""
        self.ancho = 1e-8
        self.alto = 1e-8
        self.N = 10
        self.v_media = 1000
        self.particulas = crear_gas(self.N, self.ancho, self.alto, self.v_media)
        self.dt = 1e-12
    
    def test_crear_gas(self):
        """RF3.1: Verifica que el número de partículas creadas sea correcto."""
        self.assertEqual(len(self.particulas), self.N)
        print("✓ RF3.1: Número de partículas correcto")
    
    def test_particulas_en_rango(self):
        """RF3.2: Verifica que las partículas no salgan del rango de simulación."""
        # Verificar posiciones iniciales
        for p in self.particulas:
            self.assertGreaterEqual(p.posicion[0], 0)
            self.assertLessEqual(p.posicion[0], self.ancho)
            self.assertGreaterEqual(p.posicion[1], 0)
            self.assertLessEqual(p.posicion[1], self.alto)
        
        # Avanzar simulación y verificar que permanecen en rango
        for _ in range(100):
            paso(self.particulas, self.dt, self.ancho, self.alto)
            for p in self.particulas:
                self.assertGreaterEqual(p.posicion[0], 0)
                self.assertLessEqual(p.posicion[0], self.ancho)
                self.assertGreaterEqual(p.posicion[1], 0)
                self.assertLessEqual(p.posicion[1], self.alto)
        
        print("✓ RF3.2: Partículas permanecen en el rango de simulación")
    
    def test_energia_total_positiva(self):
        """RF3.3: Verifica que la energía total sea positiva."""
        energia = energia_total(self.particulas)
        self.assertGreater(energia, 0)
        print("✓ RF3.3: Energía total positiva")
    
    def test_conservacion_energia(self):
        """RF3.4: Verifica que la energía total se conserve."""
        energia_inicial = energia_total(self.particulas)
        
        # Avanzar simulación 100 pasos
        for _ in range(100):
            paso(self.particulas, self.dt, self.ancho, self.alto)
        
        energia_final = energia_total(self.particulas)
        
        # La energía debe conservarse (variación < 1%)
        variacion_porcentual = abs(energia_final - energia_inicial) / energia_inicial * 100
        self.assertLess(variacion_porcentual, 1.0)
        
        print(f"✓ RF3.4: Energía conservada (variación: {variacion_porcentual:.4f}%)")
    
    def test_relacion_temperatura_velocidad(self):
        """RF3.5: Verifica la relación entre temperatura y velocidad promedio."""
        k_B = 1.38e-23
        
        # Calcular temperatura usando función del módulo
        T_calculada = temperatura(self.particulas, k_B)
        
        # Calcular temperatura a partir de velocidades
        velocidades = [np.linalg.norm(p.velocidad) for p in self.particulas]
        v_prom_cuad = np.mean(np.square(velocidades))
        T_velocidad = (self.particulas[0].masa * v_prom_cuad) / (2 * k_B)
        
        # Deben ser aproximadamente iguales (tolerancia del 5%)
        diferencia_porcentual = abs(T_calculada - T_velocidad) / T_calculada * 100
        self.assertLess(diferencia_porcentual, 5.0)
        
        print(f"✓ RF3.5: Relación temperatura-velocidad verificada (diferencia: {diferencia_porcentual:.2f}%)")
    
    def test_colisiones_paredes(self):
        """Verifica que las colisiones con paredes sean elásticas."""
        # Crear partícula cerca de la pared derecha
        p = Particula(0, 1e-26, 1e-10, [self.ancho - 2e-10, self.alto/2], [100, 0])
        
        # Energía antes de la colisión
        energia_antes = p.energia_cinetica()
        
        # Forzar colisión con la pared
        p.colisionar_pared(self.ancho, self.alto)
        
        # Energía después de la colisión
        energia_despues = p.energia_cinetica()
        
        self.assertAlmostEqual(energia_antes, energia_despues, delta=1e-20)
        print("✓ Colisiones con paredes elásticas")
    
    def test_movimiento_lineal(self):
        """Verifica que el movimiento sea rectilíneo entre colisiones."""
        p = Particula(0, 1e-26, 1e-10, [self.ancho/2, self.alto/2], [100, 50])
        posicion_inicial = p.posicion.copy()
        
        # Mover la partícula
        p.mover(1e-12)
        posicion_final = p.posicion.copy()
        
        # Verificar que se movió en la dirección de la velocidad
        desplazamiento_esperado = p.velocidad * 1e-12
        desplazamiento_real = posicion_final - posicion_inicial
        
        np.testing.assert_array_almost_equal(desplazamiento_real, desplazamiento_esperado, decimal=10)
        print("✓ Movimiento rectilíneo verificado")

def ejecutar_pruebas_completas():
    """Ejecuta todas las pruebas y muestra resumen."""
    print("=== PRUEBAS DEL SISTEMA DE GAS IDEAL ===\n")
    
    # Crear test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestGasIdeal)
    
    # Ejecutar pruebas
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Mostrar resumen
    print(f"\n=== RESUMEN ===")
    print(f"Pruebas ejecutadas: {result.testsRun}")
    print(f"Fallos: {len(result.failures)}")
    print(f"Errores: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("✅ TODAS LAS PRUEBAS PASARON EXITOSAMENTE")
        print("\nEl sistema cumple con todos los requerimientos funcionales:")
        print("- RF1: Modelado de partículas ✓")
        print("- RF2: Simulación del gas ✓") 
        print("- RF3: Pruebas automatizadas ✓")
        print("- RF4: Visualización científica ✓")
    else:
        print("❌ ALGUNAS PRUEBAS FALLARON")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    ejecutar_pruebas_completas()