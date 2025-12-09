import socket
import json
import math
import time

# --- CONFIGURATION ---
UDP_IP = "127.0.0.1"
UDP_PORT = 5052

# Setup Socket (Non-blocking so it doesn't freeze Blender)
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)


print(f"Listening on {UDP_IP}:{UDP_PORT}...")
# Try to receive data
while True:
    try:
        data, addr = sock.recvfrom(1024) # buffer size is 1024 bytes
        data_str = data.decode("utf-8")
        hand_data = json.loads(data_str)
        
        print("Received hand data:", hand_data)

    except BlockingIOError:
        # No data received this frame, which is fine
        pass
    except Exception as e:
        print(f"Error: {e}")

    time.sleep(0.01)
