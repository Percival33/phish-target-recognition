from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    host: str
    port: int
    debug_mode: bool
    model_config = SettingsConfigDict(env_file=(".env",), env_nested_delimiter="__")


settings = Settings()
