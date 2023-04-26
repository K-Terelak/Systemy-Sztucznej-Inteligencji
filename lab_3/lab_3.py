# zbiór parametrów opisujących warzywa/owoce
E = {'świeże', 'mrożone', 'ostre', 'słodkie', 'zielone', 'czerwone', 'żółte', 'lokalne', 'zagraniczne', 'tropikalne', 'liściaste', 'bulwowe', 'zimowe', 'letnie'}

# zbiór preferencji klienta
preferencje = {'świeże': 3, 'słodkie': 2, 'lokalne':1, 'zielone': 1}


warzywa = [
    {'nazwa': 'marchew', 'parametry': {'świeże', 'liściaste', 'czerwone', 'bulwowe'}},
    {'nazwa': 'papryka', 'parametry': {'świeże', 'czerwone', 'ostre'}},
    {'nazwa': 'truskawka', 'parametry': {'świeże', 'czerwone', 'słodkie', 'lokalne', 'letnie'}},
    {'nazwa': 'szpinak', 'parametry': {'świeże', 'liściaste', 'zielone'}},
    {'nazwa': 'dynia', 'parametry': {'świeże', 'liściaste', 'słodkie', 'bulwowe'}},
    {'nazwa': 'ananas', 'parametry': {'świeże', 'tropikalne', 'słodkie'}}
]


def stopien_spelnienia(preferencje, warzywo):
    stopien = 0
    for parametr, waga in preferencje.items():
        if parametr in warzywo['parametry']:
            stopien += waga * 1
    return stopien / sum(preferencje.values())

wynik = None
max_stopien_spelnienia = 0

for w in warzywa:
    stopien = stopien_spelnienia(preferencje, w)

    if stopien > max_stopien_spelnienia:
        wynik = w
        max_stopien_spelnienia = stopien

# Wyświetlmy wynik
if wynik:
    print(f"Najlepsze warzywo dla klienta to {wynik['nazwa']}.")
else:
    print("Nie udało się znaleźć warzywa spełniającego wymagania klienta.")