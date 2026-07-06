"""Role based access for clinicians."""

ROLES = {"admin", "radiologist", "viewer"}
_permissions = {
    "admin": {"read", "write", "delete"},
    "radiologist": {"read", "write"},
    "viewer": {"read"},
}


def has_permission(role: str, action: str) -> bool:
    return action in _permissions.get(role, set())


def check_access(user_role: str, required: str) -> bool:
    if user_role not in ROLES:
        return False
    return has_permission(user_role, required)
