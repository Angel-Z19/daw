# src/main.py
from engine.processor import AudioEngine
import sys

def main():
    """
    Punto de entrada principal del Sistema DAW.
    Este archivo orquesta la inicialización de los recursos de audio 
    y la ejecución del motor DSP.
    """
    print("====================================================")
    print("   DAW - Estación de Trabajo de Audio Digital v1.0  ")
    print("        PROTOTIPO DE AVANCE - ETAPA 3             ")
    print("====================================================")
    
    try:
        # 1. Inicialización del Sistema (Instancia los recursos clave)
        engine = AudioEngine()
        
        # 2. Ejecución del Proceso Central
        # Esto arranca el flujo ADC -> Motor DSP -> DAC
        engine.start()
        
    except Exception as e:
        print(f"Error crítico del sistema: {e}")
        sys.exit(1)
    finally:
        print("\nSesión finalizada correctamente.")

if __name__ == "__main__":
    main()