from abc import ABC, abstractmethod
from io import BytesIO

from langchain_core.tools.base import _DirectlyInjectedToolArg


class BaseMediaStorage(ABC, _DirectlyInjectedToolArg):
    """Базовый класс хранилища медиафайлов.

    Наследуется от приватного langchain_core._DirectlyInjectedToolArg —
    это позволяет объявлять параметр tool'а как `storage: BaseMediaStorage`
    (без Annotated[...]): LangChain сам исключит его из схемы для LLM и
    MediaInjectionMiddleware сам подставит инстанс при вызове.

    Внимание: _DirectlyInjectedToolArg — приватный класс (с "_"), не часть
    публичного контракта langchain_core, механизм может измениться в будущих
    версиях без предупреждения.
    """

    @abstractmethod
    async def put_media(self, buffer: BytesIO, mime_type: str) -> str:
        """Сохраняет данные, возвращает media_id."""
        ...

    @abstractmethod
    async def get_media(self, media_id: str) -> tuple[BytesIO, str] | None:
        """Возвращает (buffer, mime_type) по id, либо None, если не найдено.
        Реализация сама решает, удалять ли запись после чтения."""
        ...
