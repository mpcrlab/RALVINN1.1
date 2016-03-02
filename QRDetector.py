from sys import argv
import zbar
import Image
import cv2


class detectQRCode(object):
    def detect_qr(self, scan_image):
        scanner = zbar.ImageScanner()

        scanner.parse_config('enabled')

        # image data.
        gray = cv2.cvtColor(scan_image, cv2.COLOR_BGR2GRAY, dstCn=0)
        pil = Image.fromarray(gray)
        width, height = pil.size
        raw = pil.tostring()

        # wrap image data

        scan_image = zbar.Image(width, height, 'Y800', raw)

        # scan image for barcode

        scanner.scan(scan_image)

        # extract result

        for symbol in scan_image:
            if symbol.data == 'None':
                return "No QR Code Found in Image"
            else:
                return symbol.data
