import sys
import os

# Ajuste de ruta para encontrar el módulo 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.engine.processor import AudioEngine

def main():
    print("==============================================")
    print("   DAW - Estación de Trabajo de Audio Digital")
    print("       PROTOTIPO DE AVANCE - ETAPA 3")
    print("==============================================")
    
    engine = AudioEngine()
    
    try:
        engine.start()
    except Exception as e:
        print(f"Error en la ejecución: {e}")
    finally:
        print("\nSesión finalizada.")

if __name__ == "__main__":
    main()