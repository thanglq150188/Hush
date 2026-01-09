"""Factory resource mở rộng để tạo instance từ config."""

import logging
from typing import Any, Callable, Dict, Optional, Type

from hush.core.utils.yaml_model import YamlModel

logger = logging.getLogger(__name__)


# Registry các config class: class_name -> class
CLASS_NAME_MAP: Dict[str, Type[YamlModel]] = {}

# Registry các factory handler: config_class -> handler_function
FACTORY_HANDLERS: Dict[Type[YamlModel], Callable[[YamlModel], Any]] = {}


def register_config_class(cls: Type[YamlModel]):
    """Đăng ký config class để deserialize.

    Gọi hàm này từ các package bên ngoài để đăng ký config class của họ.

    Args:
        cls: Config class (phải kế thừa từ YamlModel)

    Example:
        from hush.core.registry import register_config_class
        from my_package.configs import MyConfig

        register_config_class(MyConfig)
    """
    CLASS_NAME_MAP[cls.__name__] = cls
    logger.debug(f"Đã đăng ký config class: {cls.__name__}")


def register_config_classes(*classes: Type[YamlModel]):
    """Đăng ký nhiều config class cùng lúc.

    Args:
        *classes: Các config class cần đăng ký

    Example:
        register_config_classes(LLMConfig, EmbeddingConfig, RedisConfig)
    """
    for cls in classes:
        register_config_class(cls)


def register_factory_handler(
    config_class: Type[YamlModel],
    handler: Callable[[YamlModel], Any]
):
    """Đăng ký factory handler cho một loại config.

    Handler được gọi để tạo resource instance từ config.

    Args:
        config_class: Config class mà handler này hỗ trợ
        handler: Function nhận config và trả về resource instance

    Example:
        from hush.core.registry import register_factory_handler
        from my_package.configs import LLMConfig
        from my_package.factory import LLMFactory

        register_factory_handler(LLMConfig, LLMFactory.create)
    """
    FACTORY_HANDLERS[config_class] = handler
    logger.debug(f"Đã đăng ký factory handler cho: {config_class.__name__}")


def get_config_class(class_name: str) -> Optional[Type[YamlModel]]:
    """Lấy config class đã đăng ký theo tên.

    Args:
        class_name: Tên của config class

    Returns:
        Config class hoặc None nếu không tìm thấy
    """
    return CLASS_NAME_MAP.get(class_name)


class ResourceFactory:
    """Factory để tạo resource instance từ config.

    Sử dụng các handler đã đăng ký để tạo instance. Các package bên ngoài
    đăng ký handler của họ qua register_factory_handler().

    Example:
        # Trong package hush-providers:
        register_factory_handler(LLMConfig, LLMFactory.create)
        register_factory_handler(EmbeddingConfig, EmbeddingFactory.create)

        # Sau đó ResourceFactory có thể tạo bất kỳ resource đã đăng ký:
        config = OpenAIConfig(model="gpt-4", api_key="sk-xxx")
        llm = ResourceFactory.create(config)
    """

    @classmethod
    def create(cls, config: YamlModel) -> Optional[Any]:
        """Tạo resource instance từ config.

        Tìm handler phù hợp dựa trên config type (bao gồm cả class cha).

        Args:
            config: Object config của resource

        Returns:
            Resource instance, hoặc None nếu tạo thất bại

        Raises:
            ValueError: Nếu không có handler nào được đăng ký cho config type này
        """
        config_type = type(config)

        # Tìm handler khớp với config type này hoặc các class cha
        handler = None
        for check_type in config_type.__mro__:
            if check_type in FACTORY_HANDLERS:
                handler = FACTORY_HANDLERS[check_type]
                break

        if not handler:
            raise ValueError(
                f"Không có factory handler nào được đăng ký cho {config_type.__name__}. "
                f"Hãy đăng ký một handler bằng register_factory_handler()."
            )

        try:
            return handler(config)
        except Exception as e:
            logger.error(f"Không thể tạo resource cho {config_type.__name__}: {e}")
            return None
