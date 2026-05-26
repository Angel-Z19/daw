import pyaudio
p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    # Si el hostapi es el de ASIO (suele ser el índice 3 o 4 en Windows, o búscalo por nombre)
    print(f"ID: {i} | Nombre: {dev['name']} | Canales Entrada: {dev['maxInputChannels']}")