import matplotlib.pyplot as plt
import numpy as np
from simulacion import Simulacion

class Visualizador:
    """Clase para visualizar los resultados de la simulación."""
    
    def __init__(self, simulacion):
        self.simulacion = simulacion
    
    def graficar_estado_instantaneo(self, paso_actual):
        """
        Grafica el estado actual del sistema.
        
        Muestra:
        - Posiciones de las partículas
        - Velocidades como flechas
        - Contorno de la caja
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Dibujar caja
        caja = plt.Rectangle((0, 0), self.simulacion.tamaño_caja, 
                           self.simulacion.tamaño_caja, fill=False, 
                           edgecolor='black', linewidth=2)
        ax.add_patch(caja)
        
        # Dibujar partículas
        for particula in self.simulacion.particulas:
            circulo = plt.Circle(particula.posicion, particula.radio, 
                               color='blue', alpha=0.6)
            ax.add_patch(circulo)
            
            # Flecha de velocidad (escalada para visualización)
            escala_velocidad = 1e-12
            ax.arrow(particula.posicion[0], particula.posicion[1],
                    particula.velocidad[0] * escala_velocidad,
                    particula.velocidad[1] * escala_velocidad,
                    head_width=0.1e-9, head_length=0.2e-9, fc='red', ec='red')
        
        ax.set_xlim(-0.1e-8, self.simulacion.tamaño_caja + 0.1e-8)
        ax.set_ylim(-0.1e-8, self.simulacion.tamaño_caja + 0.1e-8)
        ax.set_aspect('equal')
        ax.set_title(f'Estado del Gas - Paso {paso_actual}')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        
        # Mostrar estadísticas
        stats = self.simulacion.obtener_estadisticas()
        texto = (f'Temperatura: {stats["temperatura"]:.2f} K\n'
                f'Energía: {stats["energia_total"]:.2e} J\n'
                f'Presión: {stats["presion"]:.2e} Pa\n'
                f'Vel. prom: {stats["velocidad_promedio"]:.2f} m/s')
        
        ax.text(0.02, 0.98, texto, transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig, ax
    
    def graficar_evolucion_temporal(self):
        """
        Grafica la evolución temporal de las magnitudes físicas.
        """
        tiempo = np.arange(len(self.simulacion.historial_energia)) * self.simulacion.dt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # Energía total
        ax1.plot(tiempo, self.simulacion.historial_energia)
        ax1.set_xlabel('Tiempo (s)')
        ax1.set_ylabel('Energía Total (J)')
        ax1.set_title('Conservación de Energía')
        ax1.grid(True)
        
        # Temperatura
        ax2.plot(tiempo, self.simulacion.historial_temperatura)
        ax2.set_xlabel('Tiempo (s)')
        ax2.set_ylabel('Temperatura (K)')
        ax2.set_title('Temperatura del Gas')
        ax2.grid(True)
        
        # Presión
        ax3.plot(tiempo, self.simulacion.historial_presion)
        ax3.set_xlabel('Tiempo (s)')
        ax3.set_ylabel('Presión (Pa)')
        ax3.set_title('Presión en la Caja')
        ax3.grid(True)
        
        # Distribución de velocidades (último paso)
        velocidades = [np.linalg.norm(p.velocidad) for p in self.simulacion.particulas]
        ax4.hist(velocidades, bins=20, alpha=0.7, density=True)
        ax4.set_xlabel('Velocidad (m/s)')
        ax4.set_ylabel('Densidad de probabilidad')
        ax4.set_title('Distribución de Velocidades')
        ax4.grid(True)
        
        # Añadir distribución teórica de Maxwell-Boltzmann
        if len(velocidades) > 0:
            k_B = 1.38e-23
            T = self.simulacion.calcular_temperatura()
            m = self.simulacion.masa_particula
            
            # Para 2D, distribución de Rayleigh
            v_range = np.linspace(0, max(velocidades) * 1.2, 100)
            sigma = np.sqrt(k_B * T / m)
            rayleigh = (v_range / sigma**2) * np.exp(-v_range**2 / (2 * sigma**2))
            
            ax4.plot(v_range, rayleigh, 'r-', linewidth=2, label='Maxwell-Boltzmann (2D)')
            ax4.legend()
        
        plt.tight_layout()
        return fig

def ejecutar_simulacion_completa():
    """Ejecuta una simulación completa y genera visualizaciones."""
    # Crear y configurar simulación
    sim = Simulacion(tamaño_caja=1e-8, n_particulas=20, dt=1e-12)
    sim.inicializar_particulas_aleatorias()
    
    vis = Visualizador(sim)
    
    # Ejecutar simulación y guardar frames
    pasos_totales = 1000
    pasos_por_frame = 50
    
    for paso in range(0, pasos_totales, pasos_por_frame):
        sim.avanzar(pasos=pasos_por_frame)
        
        # Graficar estado actual
        fig, ax = vis.graficar_estado_instantaneo(paso)
        plt.savefig(f'frame_{paso//pasos_por_frame:04d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Graficar evolución temporal
    fig_evo = vis.graficar_evolucion_temporal()
    plt.savefig('evolucion_temporal.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Mostrar estadísticas finales
    stats_final = sim.obtener_estadisticas()
    print("\n--- ESTADÍSTICAS FINALES ---")
    for key, value in stats_final.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    ejecutar_simulacion_completa()
