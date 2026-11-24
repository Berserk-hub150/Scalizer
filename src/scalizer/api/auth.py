"""Validate Keycloak access tokens; the API never handles user passwords."""
from dataclasses import dataclass
from pathlib import Path
import os

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


@dataclass(frozen=True)
class Settings:
    root: Path
    issuer: str = "http://localhost:8080/realms/scalizer"
    audience: str = "scalizer"
    origins: tuple[str, ...] = ("http://localhost:5173",)
    local_demo: bool = False
    max_upload_bytes: int = 50 * 1024 * 1024

    @classmethod
    def from_env(cls):
        return cls(
            root=Path(os.getenv("SCALIZER_API_DATA", "data/api")),
            issuer=os.getenv("SCALIZER_KEYCLOAK_ISSUER", cls.issuer).rstrip("/"),
            audience=os.getenv("SCALIZER_KEYCLOAK_AUDIENCE", "scalizer"),
            origins=tuple(x.strip() for x in os.getenv(
                "SCALIZER_CORS_ORIGINS", "http://localhost:5173").split(",") if x.strip()),
            local_demo=os.getenv("SCALIZER_LOCAL_DEMO", "false").lower() == "true",
        )


class AccessTokens:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jwks = jwt.PyJWKClient(
            settings.issuer + "/protocol/openid-connect/certs", timeout=5)

    def verify(self, token: str) -> dict:
        key = self.jwks.get_signing_key_from_jwt(token).key
        claims = jwt.decode(token, key, algorithms=["RS256"],
                            issuer=self.settings.issuer, audience=self.settings.audience,
                            options={"require": ["exp", "iat", "sub", "iss", "aud"]})
        if claims.get("typ") != "Bearer":
            raise jwt.InvalidTokenError("Expected an access token")
        return claims


bearer = HTTPBearer(auto_error=False)


async def authenticated(request: Request,
                        credentials: HTTPAuthorizationCredentials | None = Depends(bearer)):
    settings = request.app.state.settings
    if settings.local_demo:
        if request.client and request.client.host in {"127.0.0.1", "::1"}:
            return {"sub": "local-demo"}
        raise HTTPException(403, "Local demo accepts loopback requests only")
    if not credentials:
        raise HTTPException(401, "A Keycloak access token is required",
                            headers={"WWW-Authenticate": "Bearer"})
    try:
        return request.app.state.tokens.verify(credentials.credentials)
    except jwt.PyJWKClientConnectionError:
        raise HTTPException(503, "Keycloak signing keys are unavailable") from None
    except jwt.PyJWTError:
        raise HTTPException(401, "Invalid or expired access token",
                            headers={"WWW-Authenticate": "Bearer"}) from None

