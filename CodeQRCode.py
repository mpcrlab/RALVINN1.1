import pyqrcode

qr = pyqrcode.create("TURN LEFT")
qr.png('left.png', scale=6)