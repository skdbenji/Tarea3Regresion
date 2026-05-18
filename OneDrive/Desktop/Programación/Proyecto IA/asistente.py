# Definimos la función que toma el mensaje del cliente
def asistente_dentista(mensaje_cliente):
    # Convertimos a minúsculas para que la búsqueda sea eficiente
    mensaje = mensaje_cliente.lower()
    
    # Lógica de Grafos: Evaluación de Nodos de Intención
    if "hora" in mensaje or "cita" in mensaje:
        return "¡Hola! Claro que sí. Tenemos disponibilidad mañana a las 10:00 AM. ¿Te anoto?"
    
    elif "precio" in mensaje or "valor" in mensaje:
        return "La limpieza dental está en $45.000 CLP. ¿Deseas agendar una evaluación?"
    
    else:
        return "Entiendo. Déjame comunicarte con un humano para ayudarte mejor."

# Simulamos la entrada de un cliente
entrada = input("Escribe tu duda al dentista: ")
respuesta = asistente_dentista(entrada)

print("\n[IA RESPONDE]:", respuesta)