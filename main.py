import sys
import subprocess

def entrainement_model():
    print("Spécifier votre choix: breast, diabetes, heart, derm ou covid")
    x = input()
    subprocess.run(["python", "search.py", x])

def lancement_anchor():
    print("Spécifier votre choix: breast, diabetes, heart, derm ou covid")
    x = input()
    subprocess.run(["python", "Methodes/test_anchor.py", x])

def lancement_lore():
    print("Spécifier votre choix: breast, diabetes, heart, derm ou covid")
    x = input()
    subprocess.run(["python", "Methodes/test_lore.py", x])


match input("Voulez vous entrainer le modele ou non? Y/N  "):
    case "Y":
        entrainement_model()
    case "N":
        print("Choisir si vous voulez lancer Anchor ou Lore? A/L  ")
        x = input()
        if x=="A":
            lancement_anchor()
        elif x=="L":
            lancement_lore()
        else:
            print("Mauvais arguments")
            exit()
    case _:
        print("Mauvais arguments")
        exit()

    
