# Qwen 0.5B en GRPO

Entrenamiento de un modelo pequeño para razonamiento matemático con aprendizaje por refuerzo

## Descripción

Este repositorio contiene un notebook innovador que combina el modelo **Qwen-0.5B** con la técnica de **GRPO** (Generalized Reward Policy Optimization) para entrenar una red neuronal capaz de razonar sobre problemas matemáticos de nivel escolar. Se aprovecha el benchmark **GSM8K** y se utiliza **vLLM** para mejorar la velocidad y eficiencia en la generación de texto.

¿Por qué es interesante este notebook?  
- **Exploración de nuevos métodos:** Integra aprendizaje por refuerzo en el entrenamiento de modelos de lenguaje, optimizando la respuesta mediante funciones de recompensa.
- **Formato estructurado y creativo:** Implementa un sistema de prompt basado en XML para obtener cadenas de razonamiento (chain-of-thought) y respuestas finales, lo que ayuda a desglosar el proceso cognitivo del modelo.
- **Uso de tecnologías punteras:** Desde la aceleración en generación de textos con vLLM hasta la utilización de librerías modernas como `trl` y `datasets` para entrenamiento con RL.

## Características Principales

- **Entrenamiento basado en RL:** Se definen múltiples funciones de recompensa para evaluar la calidad del razonamiento y la respuesta generada.
- **Uso del benchmark GSM8K:** Proporciona una sólida base de problemas matemáticos para medir la capacidad de razonamiento del modelo.
- **Optimización de recursos:** Gracias a vLLM, se logra una generación de textos más rápida y eficiente, permitiendo entrenamientos más ágiles.
- **Formato estructurado en la respuesta:** El sistema exige un formato específico con secciones `<reasoning>` y `<answer>` para facilitar la evaluación y comprensión de la cadena de pensamiento del modelo.

## Requisitos

Antes de comenzar, asegúrate de tener instaladas las siguientes dependencias:

- Python 3.8 o superior
- [vLLM](https://github.com/vllm-project/vllm)  
- [trl](https://github.com/lvwerra/trl)
- [datasets](https://huggingface.co/docs/datasets/)
- [transformers](https://huggingface.co/docs/transformers)
- [torch](https://pytorch.org/)

Se incluye un archivo `requirements.txt` para facilitar la instalación:

```txt
vllm
trl
datasets
transformers
torch
