import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from simulacion import crear_gas, paso, energia_total, temperatura

class Visualizador:
    """Clase para visualizar la simulación en tiempo real."""
    
    def __init__(self, ancho, alto, N, v_media, dt):
        self.ancho = ancho
        self.alto = alto
        self.N = N
        self.v_media = v_media
        self.dt = dt
        self.particulas = crear_gas(N, ancho, alto, v_media)
        self.tiempo_actual = 0.0
        
        self.fig = None
        self.ax_sim = None
        self.ax_energia = None
        self.ax_temperatura = None
        self.scat = None
        self.lineas_trayectoria = None
        self.text_stats = None
        self.energia_data = []
        self.temperatura_data = []
        self.tiempo_data = []
        
        # Para mostrar trayectorias
        self.trayectorias = [[] for _ in range(min(5, N))]  # Solo primeras 5 partículas
        
    def setup_animation(self):
        """Configura la figura y ejes para la animación."""
        self.fig = plt.figure(figsize=(15, 10))
        
        # Subplot para la simulación
        self.ax_sim = plt.subplot2grid((2, 2), (0, 0), rowspan=2)
        self.ax_energia = plt.subplot2grid((2, 2), (0, 1))
        self.ax_temperatura = plt.subplot2grid((2, 2), (1, 1))
        
        # Configurar el subplot de simulación
        self.ax_sim.set_xlim(0, self.ancho)
        self.ax_sim.set_ylim(0, self.alto)
        self.ax_sim.set_aspect('equal')
        self.ax_sim.set_xlabel('x (m)')
        self.ax_sim.set_ylabel('y (m)')
        self.ax_sim.set_title('Simulación de Gas Ideal - Movimiento en Tiempo Real')
        
        # Dibujar la caja
        caja = plt.Rectangle((0, 0), self.ancho, self.alto, fill=False, 
                           edgecolor='black', linewidth=2)
        self.ax_sim.add_patch(caja)
        
        # Inicializar scatter plot para partículas
        posiciones = np.array([p.posicion for p in self.particulas])
        self.scat = self.ax_sim.scatter(posiciones[:, 0], posiciones[:, 1], 
                                      s=100, c='blue', alpha=0.7, label='Partículas')
        
        # Inicializar líneas de trayectoria para las primeras 5 partículas
        colores = ['red', 'green', 'orange', 'purple', 'brown']
        self.lineas_trayectoria = []
        for i in range(min(5, self.N)):
            linea, = self.ax_sim.plot([], [], color=colores[i], linewidth=1, 
                                    alpha=0.5, label=f'Trayectoria {i+1}')
            self.lineas_trayectoria.append(linea)
        
        self.ax_sim.legend()
        
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
                                         verticalalignment='top', fontsize=10,
                                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
    def update(self, frame):
        """Función de actualización para la animación."""
        # Avanzar la simulación un paso
        paso(self.particulas, self.dt, self.ancho, self.alto)
        self.tiempo_actual += self.dt
        
        # Actualizar posiciones de las partículas
        posiciones = np.array([p.posicion for p in self.particulas])
        self.scat.set_offsets(posiciones)
        
        # Actualizar trayectorias de las primeras 5 partículas
        for i in range(min(5, self.N)):
            if len(self.trayectorias[i]) < 50:  # Mantener últimas 50 posiciones
                self.trayectorias[i].append(self.particulas[i].posicion.copy())
            else:
                self.trayectorias[i].pop(0)
                self.trayectorias[i].append(self.particulas[i].posicion.copy())
            
            if len(self.trayectorias[i]) > 1:
                trayectoria = np.array(self.trayectorias[i])
                self.lineas_trayectoria[i].set_data(trayectoria[:, 0], trayectoria[:, 1])
        
        # Calcular energía y temperatura
        energia = energia_total(self.particulas)
        temp = temperatura(self.particulas)
        
        # Actualizar datos de energía y temperatura
        self.energia_data.append(energia)
        self.temperatura_data.append(temp)
        self.tiempo_data.append(self.tiempo_actual)
        
        # Actualizar gráficas de energía y temperatura
        self.ax_energia.clear()
        self.ax_temperatura.clear()
        
        if len(self.tiempo_data) > 0:
            self.ax_energia.plot(self.tiempo_data, self.energia_data, 'b-', linewidth=2)
            self.ax_temperatura.plot(self.tiempo_data, self.temperatura_data, 'r-', linewidth=2)
        
        # Reconfigurar las gráficas después de limpiarlas
        self.ax_energia.set_xlabel('Tiempo (s)')
        self.ax_energia.set_ylabel('Energía Total (J)')
        self.ax_energia.set_title('Conservación de Energía')
        self.ax_energia.grid(True, alpha=0.3)
        
        self.ax_temperatura.set_xlabel('Tiempo (s)')
        self.ax_temperatura.set_ylabel('Temperatura (K)')
        self.ax_temperatura.set_title('Temperatura del Gas')
        self.ax_temperatura.grid(True, alpha=0.3)
        
        # Actualizar estadísticas
        velocidades = [np.linalg.norm(p.velocidad) for p in self.particulas]
        texto_stats = (f'Tiempo: {self.tiempo_actual:.2e} s\n'
                      f'Partículas: {len(self.particulas)}\n'
                      f'Temperatura: {temp:.2f} K\n'
                      f'Energía: {energia:.2e} J\n'
                      f'Vel. media: {np.mean(velocidades):.2f} m/s\n'
                      f'Dispersión: {np.std(velocidades):.2f} m/s')
        
        self.text_stats.set_text(texto_stats)
        
        return [self.scat] + self.lineas_trayectoria + [self.text_stats]
    
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
            blit=True,
            repeat=True
        )
        
        plt.show()
        
        return anim

def demostracion_rapida():
    """Función de demostración rápida."""
    print("Iniciando simulación de gas ideal...")
    print("Modelo físico: Partículas puntuales con colisiones elásticas")
    print("Propósito: Visualizar dinámica molecular y verificar conservación de energía")
    
    # Configuración para demostración
    ancho = 5e-9
    alto = 5e-9
    N = 20
    v_media = 800  # m/s (velocidad típica molecular a temperatura ambiente)
    dt = 1e-12
    
    vis = Visualizador(ancho, alto, N, v_media, dt)
    
    print(f"\nConfiguración:")
    print(f"- Caja: {ancho:.1e} m × {alto:.1e} m")
    print(f"- Partículas: {N}")
    print(f"- Velocidad media: {v_media} m/s")
    print(f"- Paso temporal: {dt:.1e} s")
    
    print("\nIniciando animación en tiempo real...")
    print("La animación se ejecutará durante 10 segundos reales.")
    print("Características:")
    print("- Partículas azules: posiciones actuales")
    print("- Líneas de colores: trayectorias recientes")
    print("- Gráficas: energía total y temperatura")
    print("- Estadísticas en tiempo real")
    
    try:
        # Animación por 10 segundos reales
        vis.animar(duracion=10, fps=20)
        
    except KeyboardInterrupt:
        print("\nSimulación interrumpida por el usuario.")
    
    finally:
        # Mostrar estadísticas finales
        energia = energia_total(vis.particulas)
        temp = temperatura(vis.particulas)
        velocidades = [np.linalg.norm(p.velocidad) for p in vis.particulas]
        
        print("\n--- ESTADÍSTICAS FINALES ---")
        print(f"Energía total: {energia:.2e} J")
        print(f"Temperatura: {temp:.2f} K")
        print(f"Velocidad promedio: {np.mean(velocidades):.2f} m/s")
        print(f"Tiempo simulado: {vis.tiempo_actual:.2e} s")
        print(f"Pasos completados: {int(vis.tiempo_actual / dt)}")

if __name__ == "__main__":
    demostracion_rapida()