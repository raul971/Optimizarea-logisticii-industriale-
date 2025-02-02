import barcode
from barcode.writer import ImageWriter

def generate_barcode(data, output_filename, barcode_format='code128'):
    """
    Generează un cod de bare în format PNG.

    Parametri:
      - data: șirul de caractere care trebuie codificat.
              Pentru EAN-13 trebuie să aibă 12 cifre (ultima cifră este calculată automat).
      - output_filename: numele fișierului de ieșire (fără extensie);
                           fișierul generat va avea extensia .png.
      - barcode_format: tipul codului de bare, de exemplu 'ean13', 'code128', 'upca', etc.

    Returnează numele complet al fișierului generat sau None dacă apare o eroare.
    """
    try:
        BarcodeClass = barcode.get_barcode_class(barcode_format)
    except barcode.errors.BarcodeNotFoundError:
        print("Format de cod de bare invalid:", barcode_format)
        return None

    # Se folosește ImageWriter pentru a genera un fișier PNG.
    barcode_obj = BarcodeClass(data, writer=ImageWriter())
    full_filename = barcode_obj.save(output_filename)
    return full_filename

if __name__ == '__main__':
    # Exemplu: generare cod de bare EAN-13 (data trebuie să aibă 12 cifre)
    data_ean = "123456789012"
    output_ean = generate_barcode(data_ean, "ean13_barcode", barcode_format="ean13")
    print("EAN-13 Barcode generat și salvat ca:", output_ean)

    # Exemplu: generare cod de bare Code 128 (poți folosi orice șir)
    data_code128 = "ABC123456789"
    output_code128 = generate_barcode(data_code128, "code128_barcode", barcode_format="code128")
    print("Code 128 Barcode generat și salvat ca:", output_code128)
