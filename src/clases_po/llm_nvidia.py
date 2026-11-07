"""Cliente para el LLM de NVIDIA (API compatible con OpenAI).

Adaptación de la clase ``LLMNvidia`` que el profesor compartió en clase
(``PI3_2026Q4/src/clases_po/llm_nvidia.py``). Se conserva la llamada en
streaming del ejemplo oficial de NVIDIA, la extracción de JSON y el
patrón ``(datos, fuente)`` con respaldo, para que un notebook nunca se rompa
por una falla del servicio. La caché en disco no vive acá: la maneja cada
clase que usa el LLM (por ejemplo ``EnriquecedorFestivos``).

La clave del API nunca se imprime ni se guarda: se recibe por parámetro o se
lee de la variable de entorno ``NVIDIA_API_KEY``.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Callable


class LLMNvidia:
    """Conexión con el API de NVIDIA.

    Args:
        api_key: Clave ``nvapi-...``. Si no se pasa, se lee de la variable de
            entorno ``NVIDIA_API_KEY``.
        modelo: Identificador del modelo en NVIDIA.
        pensar: Valor de ``enable_thinking``. ``True`` razona más, pero es
            mucho más lento.
        mostrar_razonamiento: Imprime el ``reasoning_content`` mientras llega.
    """

    BASE_URL = "https://integrate.api.nvidia.com/v1"
    MODELO_POR_DEFECTO = "nvidia/nemotron-3-ultra-550b-a55b"

    def __init__(
        self,
        api_key: str | None = None,
        modelo: str = MODELO_POR_DEFECTO,
        pensar: bool = False,
        mostrar_razonamiento: bool = False,
    ) -> None:
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY", "")
        self.modelo = modelo
        self.pensar = pensar
        self.mostrar_razonamiento = mostrar_razonamiento
        self.client = None

        if self._api_key:
            try:
                from openai import OpenAI

                self.client = OpenAI(base_url=self.BASE_URL, api_key=self._api_key)
            except ImportError:
                print("Falta la librería: instalar con `pip install openai`.")

    # ------------------------------------------------------------------ estado
    @property
    def activo(self) -> bool:
        """True si hay cliente configurado y la conexión no ha fallado."""
        return self.client is not None

    def desactivar(self, motivo: str = "") -> None:
        """Apaga el cliente (por ejemplo, tras un error de red)."""
        self.client = None
        if motivo:
            print(f"LLM desactivado: {motivo}")

    def __repr__(self) -> str:
        estado = (
            "activo" if self.activo
            else "sin conexión al LLM: se usará el archivo guardado o, si no existe, el respaldo"
        )
        return f"LLMNvidia(modelo='{self.modelo}', pensar={self.pensar}, estado={estado})"

    # ------------------------------------------------------------------ llamadas
    def _crear_stream(self, mensajes: list[dict], temperature: float, max_tokens: int):
        return self.client.chat.completions.create(
            model=self.modelo,
            messages=mensajes,
            temperature=temperature,
            top_p=0.95,
            max_tokens=max_tokens,
            extra_body={"chat_template_kwargs": {"enable_thinking": self.pensar}},
            stream=True,
        )

    def _recorrer_stream(self, completion, imprimir: bool = False) -> str:
        """Recorre el streaming y junta el contenido que llega."""
        partes = []
        for chunk in completion:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            razonamiento = getattr(delta, "reasoning_content", None)
            if razonamiento and (imprimir or self.mostrar_razonamiento):
                print(razonamiento, end="")
            if delta.content is not None:
                partes.append(delta.content)
                if imprimir:
                    print(delta.content, end="")
        return "".join(partes)

    def consultar(
        self,
        prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 8192,
        solo_json: bool = True,
    ) -> str:
        """Envía un prompt y devuelve el texto completo de la respuesta."""
        if not self.activo:
            raise RuntimeError("El LLM no está configurado (falta NVIDIA_API_KEY).")
        mensajes = [{"role": "user", "content": prompt}]
        if solo_json:
            mensajes.insert(
                0,
                {
                    "role": "system",
                    "content": "Respondés únicamente con JSON válido, sin texto adicional.",
                },
            )
        return self._recorrer_stream(self._crear_stream(mensajes, temperature, max_tokens))

    def pedir_json(
        self,
        nombre: str,
        prompt: str,
        validar: Callable[[Any], Any],
        respaldo: Any,
        intentos: int = 3,
    ) -> tuple[Any, str]:
        """Pide un JSON, lo valida y reintenta.

        Devuelve ``(datos, fuente)`` con fuente ``"LLM"`` o ``"respaldo"``.
        ``validar(datos)`` debe devolver los datos limpios o lanzar una excepción.
        """
        if not self.activo:
            return respaldo, "respaldo"
        for intento in range(1, intentos + 1):
            try:
                datos = validar(self.extraer_json(self.consultar(prompt)))
                print(f"{nombre}: respuesta del LLM válida (intento {intento}).")
                return datos, "LLM"
            except Exception as error:  # noqa: BLE001
                print(f"{nombre}: intento {intento} no válido ({type(error).__name__}: {str(error)[:120]}).")
        print(f"{nombre}: se usa el catálogo de respaldo.")
        return respaldo, "respaldo"

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def extraer_json(texto: str) -> Any:
        """Quita bloques de razonamiento y ```json``` y devuelve el objeto JSON."""
        texto = re.sub(r"<think>.*?</think>", "", texto, flags=re.S)
        texto = re.sub(r"```(?:json)?", "", texto).strip()
        inicio = min([i for i in (texto.find("{"), texto.find("[")) if i >= 0], default=-1)
        fin = max(texto.rfind("}"), texto.rfind("]"))
        if inicio < 0 or fin < 0:
            raise ValueError("La respuesta no contiene JSON")
        return json.loads(texto[inicio : fin + 1])
