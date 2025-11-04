import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from simulacion import Simulacion

class Visualizador:
    """Clase para visualizar la simulación en tiempo real."""
    
    def __init__(self, simulacion):
        self.simulacion = simulacion
        self.fig = None
        self.ax = None
        self.scat = None
        self.quiver = None
        self.text_stats = None
        self.energia_data = []
        self.temperatura_data = []
        self.tiempo_data = []
        
    def setup_animation(self):
        """Configura la figura y ejes para la animación."""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Subplot para la simulación
        self.ax_sim = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        self.ax_energia = plt.subplot2grid((2, 2), (0, 1))
        self.ax_temperatura = plt.subplot2grid((2, 2), (1, 1))
        
        # Configurar el subplot de simulación
        self.ax_sim.set_xlim(0, self.simulacion.tamaño_caja)
        self.ax_sim.set_ylim(0, self.simulacion.tamaño_caja)
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_xlabel('x (m)')
        self.ax_sim.set_ylabel('y (m)')
        self.ax_sim.set_title('Simulación de Gas Ideal - Tiempo Real')
        
        # Dibujar la caja
        caja = plt.Rectangle((0, 0), self.simulacion.tamaño_caja, 
                           self.simulacion.tamaño_caja, fill=False, 
                           edgecolor='black', linewidth=2)
        self.ax_sim.add_patch(caja)
        
        # Inicializar scatter plot para partículas
        posiciones = np.array([p.posicion for p in self.simulacion.particulas])
        self.scat = self.ax_sim.scatter(posiciones[:, 0], posiciones[:, 1], 
                                      s=100, c='blue', alpha=0.7)
        
        # Configurar gráficas de energía y temperatura
        self.ax_energia.set_xlabel('Tiempo (s)')
        self.ax_energia.set_ylabel('Energía Total (J)')
        self.ax_energia.set_title('Conservación de Energía')
        self.ax_energia.grid(True)
        
        self.ax_temperatura.set_xlabel('Tiempo (s)')
        self.ax_temperatura.set_ylabel('Temperatura (K)')
        self.ax_temperatura.set_title('Temperatura del Gas')
        self.ax_temperatura.grid(True)
        
        # Texto para estadísticas
        self.text_stats = self.ax_sim.text(0.02, 0.98, '', transform=self.ax_sim.transAxes,
                                         verticalalignment='top',
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
    def update(self, frame):
        """Función de actualización para la animación."""
        # Avanzar la simulación varios pasos por frame para mejor rendimiento
        pasos_por_frame = 10
        self.simulacion.avanzar(pasos=pasos_por_frame)
        
        # Actualizar posiciones de las partículas
        posiciones = np.array([p.posicion for p in self.simulacion.particulas])
        self.scat.set_offsets(posiciones)
        
        # Actualizar datos de energía y temperatura
        if len(self.simulacion.historial_energia) > 0:
            self.energia_data.append(self.simulacion.historial_energia[-1])
            self.temperatura_data.append(self.simulacion.historial_temperatura[-1])
            self.tiempo_data.append(self.simulacion.tiempo_actual)
        
        # Actualizar gráficas de energía y temperatura
        self.ax_energia.clear()
        self.ax_temperatura.clear()
        
        if len(self.tiempo_data) > 0:
            self.ax_energia.plot(self.tiempo_data, self.energia_data, 'b-')
            self.ax_temperatura.plot(self.tiempo_data, self.temperatura_data, 'r-')
        
        # Reconfigurar las gráficas después de limpiarlas
        self.ax_energia.set_xlabel('Tiempo (s)')
        self.ax_energia.set_ylabel('Energía Total (J)')
        self.ax_energia.set_title('Conservación de Energía')
        self.ax_energia.grid(True)
        
        self.ax_temperatura.set_xlabel('Tiempo (s)')
        self.ax_temperatura.set_ylabel('Temperatura (K)')
        self.ax_temperatura.set_title('Temperatura del Gas')
        self.ax_temperatura.grid(True)
        
        # Actualizar estadísticas
        stats = self.simulacion.obtener_estadisticas()
        texto_stats = (f'Tiempo: {stats["tiempo"]:.2e} s\n'
                      f'Partículas: {len(self.simulacion.particulas)}\n'
                      f'Temperatura: {stats["temperatura"]:.2f} K\n'
                      f'Energía: {stats["energia_total"]:.2e} J\n'
                      f'Presión: {stats["presion"]:.2e} Pa\n'
                      f'Vel. prom: {stats["velocidad_promedio"]:.2f} m/s')
        
        self.text_stats.set_text(texto_stats)
        
        return self.scat, self.text_stats
    
    def animar(self, duracion=10, fps=30):
        """Ejecuta la animación en tiempo real."""
        self.setup_animation()
        
        # Calcular número total de frames
        total_frames = duracion * fps
        
        # Crear animación
        anim = animation.FuncAnimation(
            self.fig, 
            self.update, 
            frames=total_frames,
            interval=1000/fps,  # ms entre frames
            blit=False,
            repeat=True
        )
        
        plt.show()
        
        return anim

    def graficar_distribucion_velocidades(self):
        """Grafica la distribución de velocidades final."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Distribución de magnitudes de velocidad
        velocidades = [np.linalg.norm(p.velocidad) for p in self.simulacion.particulas]
        
        ax1.hist(velocidades, bins=20, alpha=0.7, density=True, color='blue')
        ax1.set_xlabel('Velocidad (m/s)')
        ax1.set_ylabel('Densidad de probabilidad')
        ax1.set_title('Distribución de Velocidades')
        ax1.grid(True)
        
        # Añadir distribución teórica de Maxwell-Boltzmann para 2D
        if len(velocidades) > 0:
            k_B = 1.38e-23
            T = self.simulacion.calcular_temperatura()
            m = self.simulacion.masa_particula
            
            v_range = np.linspace(0, max(velocidades) * 1.2, 100)
            sigma = np.sqrt(k_B * T / m)
            # Distribución de Rayleigh para velocidad en 2D
            rayleigh = (v_range / sigma**2) * np.exp(-v_range**2 / (2 * sigma**2))
            
            ax1.plot(v_range, rayleigh, 'r-', linewidth=2, label='Maxwell-Boltzmann (2D)')
            ax1.legend()
        
        # Distribución de componentes de velocidad
        vx = [p.velocidad[0] for p in self.simulacion.particulas]
        vy = [p.velocidad[1] for p in self.simulacion.particulas]
        
        ax2.hist(vx, bins=15, alpha=0.7, density=True, color='red', label='vx')
        ax2.hist(vy, bins=15, alpha=0.7, density=True, color='green', label='vy')
        ax2.set_xlabel('Componente de Velocidad (m/s)')
        ax2.set_ylabel('Densidad de probabilidad')
        ax2.set_title('Distribución de Componentes de Velocidad')
        ax2.legend()
        ax2.grid(True)
        
        # Añadir distribución gaussiana teórica
        if len(vx) > 0:
            v_componente = np.linspace(min(vx + vy), max(vx + vy), 100)
            gaussiana = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-v_componente**2 / (2 * sigma**2))
            ax2.plot(v_componente, gaussiana, 'k-', linewidth=2, label='Distribución Gaussiana')
            ax2.legend()
        
        plt.tight_layout()
        plt.show()

    def graficar_trayectorias(self):
        """Grafica las trayectorias de algunas partículas seleccionadas."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Dibujar caja
        caja = plt.Rectangle((0, 0), self.simulacion.tamaño_caja, 
                           self.simulacion.tamaño_caja, fill=False, 
                           edgecolor='black', linewidth=2)
        ax.add_patch(caja)
        
        # Graficar trayectorias de las primeras 5 partículas
        colores = ['red', 'blue', 'green', 'orange', 'purple']
        for i in range(min(5, len(self.simulacion.particulas))):
            p = self.simulacion.particulas[i]
            if hasattr(p, 'historial_posiciones') and len(p.historial_posiciones) > 0:
                trayectoria = np.array(p.historial_posiciones)
                ax.plot(trayectoria[:, 0], trayectoria[:, 1], 
                       color=colores[i], alpha=0.6, linewidth=1, 
                       label=f'Partícula {p.id}')
                
                # Marcar posición inicial y final
                ax.plot(trayectoria[0, 0], trayectoria[0, 1], 'o', 
                       color=colores[i], markersize=6, label=f'Inicio {p.id}')
                ax.plot(trayectoria[-1, 0], trayectoria[-1, 1], 's', 
                       color=colores[i], markersize=6, label=f'Fin {p.id}')
        
        ax.set_xlim(0, self.simulacion.tamaño_caja)
        ax.set_ylim(0, self.simulacion.tamaño_caja)
        ax.set_aspect('equal')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_title('Trayectorias de Partículas')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def demostracion_rapida():
    """Función de demostración rápida."""
    print("Iniciando simulación de gas ideal...")
    
    # Configuración más pequeña para demostración rápida
    sim = Simulacion(tamaño_caja=5e-9, n_particulas=15, dt=1e-13)
    sim.inicializar_particulas_aleatorias()
    
    vis = Visualizador(sim)
    
    print("Iniciando animación en tiempo real...")
    print("La animación se ejecutará durante 10 segundos reales.")
    print("Presiona Ctrl+C en la terminal para detenerla antes.")
    
    try:
        # Animación por 10 segundos reales
        vis.animar(duracion=10, fps=20)
        
        # Gráficas adicionales después de la animación
        print("\nGenerando gráficas de análisis...")
        vis.graficar_distribucion_velocidades()
        vis.graficar_trayectorias()
        
    except KeyboardInterrupt:
        print("\nSimulación interrumpida por el usuario.")
    
    finally:
        # Mostrar estadísticas finales
        stats = sim.obtener_estadisticas()
        print("\n--- ESTADÍSTICAS FINALES ---")
        for key, value in stats.items():
            print(f"{key}: {value}")

def simulacion_completa():
    """Simulación más larga para análisis detallado."""
    print("Configurando simulación completa...")
    
    sim = Simulacion(tamaño_caja=1e-8, n_particulas=30, dt=1e-12)
    sim.inicializar_particulas_aleatorias()
    
    vis = Visualizador(sim)
    
    print("Ejecutando simulación con visualización...")
    vis.animar(duracion=30, fps=15)
    
    # Análisis posterior
    print("Generando análisis detallado...")
    vis.graficar_distribucion_velocidades()
    vis.graficar_trayectorias()
    
    return sim

if __name__ == "__main__":
    # Ejecutar demostración rápida por defecto
    demostracion_rapida()