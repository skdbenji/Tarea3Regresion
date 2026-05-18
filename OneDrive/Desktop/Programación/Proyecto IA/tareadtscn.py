texto = """
La ciencia de datos es un campo interdisciplinario que utiliza
métodos científicos y algoritmos para extraer conocimiento de
los datos. La estadística y la programación son herramientas
fundamentales para un científico de datos. Los datos pueden
ser estructurados o no estructurados. El análisis de datos
permite tomar decisiones basadas en evidencia.
"""

def limpiar_texto(textolimpio):
    # Convertir a minúsculas
    textolimpio = textolimpio.lower()
    # Eliminar signos de puntuación 

    signos_puntuacion =  '\n.,;:!?()[]{}"\''
    


    for signo in signos_puntuacion:
        textolimpio = textolimpio.replace(signo, '')

            


    
    print(textolimpio)
    return textolimpio     


limpiar_texto(texto)
