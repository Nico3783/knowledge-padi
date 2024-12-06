import secrets
secret_key = secrets.token_hex(24)  # Generates a 48-character key
print(secret_key)
