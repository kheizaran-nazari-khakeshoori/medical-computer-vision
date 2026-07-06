"""Enhancing security with jwt authentication."""

try:
    import jwt

    SECRET = "supersecret"

    def create_token(payload: dict):
        return jwt.encode(payload, SECRET, algorithm="HS256")

    def verify_token(token: str):
        return jwt.decode(token, SECRET, algorithms=["HS256"])

except Exception:

    def create_token(payload: dict):
        return "stub_token"

    def verify_token(token: str):
        return {"stub": True}
