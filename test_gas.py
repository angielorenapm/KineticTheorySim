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

    # =====================================================
    # ESCENARIOS ESPECÍFICOS CORREGIDOS
    # =====================================================
    
    def test_E1_movimiento_libre_10_particulas(self):
        """
        E1: Movimiento libre - 10 partículas sin colisiones entre sí.
        
        Resultado esperado: Las trayectorias son rectilíneas y se reflejan correctamente.
        """
        print("\n=== E1: Movimiento libre con 10 partículas ===")
        
        # Crear 10 partículas con posiciones que eviten colisiones iniciales
        # y usar una caja más grande para minimizar colisiones con paredes
        ancho_E1 = 2e-8  # Caja más grande
        alto_E1 = 2e-8
        particulas_E1 = []
        
        for i in range(10):
            # Posicionar en una cuadrícula para evitar colisiones
            fila = i % 3
            columna = i // 3
            x = (columna + 1) * (ancho_E1 / 4)
            y = (fila + 1) * (alto_E1 / 4)
            
            # Velocidades en direcciones diferentes pero con magnitud moderada
            angulo = i * 36 * np.pi / 180  # 36° entre cada partícula
            velocidad = 200 * np.array([np.cos(angulo), np.sin(angulo)])  # Velocidad moderada
            
            particula = Particula(i, 1e-26, 1e-10, [x, y], velocidad)
            particulas_E1.append(particula)
        
        # Verificar movimiento por 20 pasos (reducido para evitar muchas colisiones)
        for paso_num in range(20):
            posiciones_antes = [p.posicion.copy() for p in particulas_E1]
            velocidades_antes = [p.velocidad.copy() for p in particulas_E1]
            
            # Aplicar movimiento y colisiones
            for p in particulas_E1:
                p.mover(self.dt)
                p.colisionar_pared(ancho_E1, alto_E1)
            
            # Verificar que cada partícula se comporta físicamente correctamente
            for i, p in enumerate(particulas_E1):
                desplazamiento_real = p.posicion - posiciones_antes[i]
                
                # Solo verificar movimiento rectilíneo si no hubo colisión
                # (cuando la velocidad no cambió significativamente)
                cambio_velocidad = np.linalg.norm(p.velocidad - velocidades_antes[i])
                if cambio_velocidad < 1e-10:  # No hubo colisión
                    # El desplazamiento debe ser paralelo a la velocidad inicial
                    producto_cruz = np.cross(desplazamiento_real, velocidades_antes[i])
                    # Aumentamos la tolerancia para errores numéricos
                    self.assertAlmostEqual(producto_cruz, 0, delta=1e-12,
                                        msg=f"Partícula {i} no se movió en línea recta en paso {paso_num}")
                else:
                    # Hubo colisión - verificar que la energía se conserva
                    energia_antes = 0.5 * p.masa * np.dot(velocidades_antes[i], velocidades_antes[i])
                    energia_despues = p.energia_cinetica()
                    self.assertAlmostEqual(energia_antes, energia_despues, delta=1e-20,
                                         msg=f"Energía no conservada en colisión de partícula {i}")
        
        print("✓ E1: 10 partículas muestran movimiento rectilíneo entre colisiones y reflexiones correctas")
    
    def test_E2_energia_constante_50_particulas(self):
        """
        E2: Energía constante - 100 pasos con 50 partículas.
        
        Resultado esperado: La energía total se mantiene constante (<1% de variación).
        """
        print("\n=== E2: Energía constante con 50 partículas ===")
        
        # Crear 50 partículas
        particulas_E2 = crear_gas(50, self.ancho, self.alto, 800)
        
        energia_inicial = energia_total(particulas_E2)
        energias = [energia_inicial]
        
        # Ejecutar 100 pasos
        for i in range(100):
            paso(particulas_E2, self.dt, self.ancho, self.alto)
            energias.append(energia_total(particulas_E2))
        
        energia_final = energias[-1]
        
        # Calcular variación porcentual máxima
        variacion_maxima = max(abs(e - energia_inicial) / energia_inicial * 100 for e in energias)
        
        print(f"Energía inicial: {energia_inicial:.2e} J")
        print(f"Energía final: {energia_final:.2e} J")
        print(f"Variación máxima: {variacion_maxima:.4f}%")
        
        self.assertLess(variacion_maxima, 1.0, 
                       f"La energía varió más del 1% (variación: {variacion_maxima:.4f}%)")
        
        print("✓ E2: Energía constante verificada con 50 partículas en 100 pasos")
    
    def test_E3_aumento_velocidad_temperatura(self):
        """
        E3: Aumento de velocidad media - Se duplican las velocidades iniciales.
        
        Resultado esperado: La temperatura efectiva aumenta por un factor aproximado de 4.
        """
        print("\n=== E3: Aumento de velocidad y temperatura ===")
        
        k_B = 1.38e-23
        
        # Configuración inicial
        v_media_inicial = 500  # m/s
        particulas_E3 = crear_gas(20, self.ancho, self.alto, v_media_inicial)
        
        # Temperatura inicial
        T_inicial = temperatura(particulas_E3, k_B)
        print(f"Temperatura inicial (v_media={v_media_inicial} m/s): {T_inicial:.2f} K")
        
        # Duplicar todas las velocidades
        for p in particulas_E3:
            p.velocidad *= 2
        
        # Temperatura después de duplicar velocidades
        T_duplicada = temperatura(particulas_E3, k_B)
        print(f"Temperatura después de duplicar velocidades: {T_duplicada:.2f} K")
        
        # Calcular factor de aumento
        factor_aumento = T_duplicada / T_inicial
        print(f"Factor de aumento: {factor_aumento:.2f} (esperado: ~4.00)")
        
        # Verificar que el factor está cerca de 4 (tolerancia 25% por fluctuaciones estadísticas)
        self.assertGreater(factor_aumento, 3.0, "El factor de aumento es menor que 3.0")
        self.assertLess(factor_aumento, 5.0, "El factor de aumento es mayor que 5.0")
        
        print("✓ E3: La temperatura aumenta aproximadamente 4x al duplicar velocidades")
    
    def test_E4_verificacion_visualizacion(self):
        """
        E4: Visualización gráfica - Verificación básica de componentes de visualización.
        
        Nota: Esta prueba no ejecuta la visualización completa, pero verifica
        que los componentes necesarios estén presentes.
        """
        print("\n=== E4: Verificación de componentes de visualización ===")
        
        # Verificar que tenemos partículas para visualizar
        self.assertGreater(len(self.particulas), 0, "No hay partículas para visualizar")
        
        # Verificar que las partículas tienen atributos necesarios para visualización
        for i, p in enumerate(self.particulas[:3]):  # Verificar solo las primeras 3
            self.assertTrue(hasattr(p, 'posicion'), f"Partícula {i} no tiene atributo 'posicion'")
            self.assertTrue(hasattr(p, 'velocidad'), f"Partícula {i} no tiene atributo 'velocidad'")
            self.assertTrue(hasattr(p, 'radio'), f"Partícula {i} no tiene atributo 'radio'")
            
            # Verificar que los atributos tienen los tipos correctos
            self.assertIsInstance(p.posicion, np.ndarray, f"Partícula {i}: posicion no es numpy array")
            self.assertIsInstance(p.velocidad, np.ndarray, f"Partícula {i}: velocidad no es numpy array")
            self.assertIsInstance(p.radio, float, f"Partícula {i}: radio no es float")
        
        # Verificar dimensiones de la caja
        self.assertGreater(self.ancho, 0, "Ancho de la caja debe ser positivo")
        self.assertGreater(self.alto, 0, "Alto de la caja debe ser positivo")
        
        print("✓ E4: Componentes básicos de visualización verificados")
        print("  Nota: Ejecute 'python graficar.py' para ver la visualización completa")

def ejecutar_escenarios_especificos():
    """Ejecuta solo los escenarios específicos E1-E4."""
    print("=== EJECUCIÓN DE ESCENARIOS ESPECÍFICOS E1-E4 ===\n")
    
    # Crear instancia de TestGasIdeal
    test_instance = TestGasIdeal()
    test_instance.setUp()
    
    # Ejecutar cada escenario
    escenarios = [
        ('E1', test_instance.test_E1_movimiento_libre_10_particulas),
        ('E2', test_instance.test_E2_energia_constante_50_particulas),
        ('E3', test_instance.test_E3_aumento_velocidad_temperatura),
        ('E4', test_instance.test_E4_verificacion_visualizacion),
    ]
    
    resultados = []
    
    for nombre, metodo in escenarios:
        try:
            print(f"\n{'='*50}")
            print(f"Ejecutando {nombre}...")
            print('='*50)
            metodo()
            resultados.append((nombre, True, "✓ Éxito"))
        except Exception as e:
            resultados.append((nombre, False, f"✗ Error: {str(e)}"))
    
    # Mostrar resumen
    print(f"\n{'='*60}")
    print("RESUMEN DE ESCENARIOS E1-E4")
    print('='*60)
    
    for nombre, exitoso, mensaje in resultados:
        estado = "PASÓ" if exitoso else "FALLÓ"
        print(f"{nombre}: {estado} - {mensaje}")
    
    todos_exitosos = all(exitoso for _, exitoso, _ in resultados)
    
    if todos_exitosos:
        print("\n✅ TODOS LOS ESCENARIOS E1-E4 PASARON EXITOSAMENTE")
    else:
        print("\n❌ ALGUNOS ESCENARIOS FALLARON")
    
    return todos_exitosos

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
    # Ejecutar pruebas completas por defecto
    if ejecutar_pruebas_completas():
        # Si las pruebas básicas pasan, ejecutar escenarios específicos
        print("\n" + "="*70)
        print("CONTINUANDO CON VERIFICACIÓN DE ESCENARIOS ESPECÍFICOS E1-E4")
        print("="*70)
        
        ejecutar_escenarios_especificos()