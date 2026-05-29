import os

# ¡ESTA LÍNEA ES LA CLAVE! 
# Le dice a sounddevice que cargue el binario compatible con ASIO
os.environ["SD_ENABLE_ASIO"] = "1"

import sounddevice as sd

# Ahora imprimimos para ver si ocurrió el milagro
print(sd.query_devices())