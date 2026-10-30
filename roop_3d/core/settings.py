"""Environment configuration owned exclusively by Roop 3D."""
from __future__ import annotations

import os
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: int = Field(default=8030, ge=1, le=65535)
    max_body_bytes: int = Field(default=2 * 1024 * 1024, ge=1, le=64 * 1024 * 1024)
    dev_cors: bool = False

    @field_validator("port")
    @classmethod
    def reserve_vite_port(cls, value: int) -> int:
        if value == 5174:
            raise ValueError("Port 5174 is reserved for development Vite HMR; choose another API port")
        return value

    @classmethod
    def from_env(cls) -> "Settings":
        values = {name: os.environ[variable] for name, variable in {
            "host": "ROOP3D_HOST", "port": "ROOP3D_PORT",
            "max_body_bytes": "ROOP3D_MAX_BODY_BYTES", "dev_cors": "ROOP3D_DEV_CORS",
        }.items() if variable in os.environ}
        return cls.model_validate(values)
