from picamera2 import Picamera2

picam2 = Picamera2()
config=picam2.create_preview_configuration()
picam2.configure(config)
picam2.start()