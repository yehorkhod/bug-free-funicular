from requests import Response, post

BASE_URL: str = "http://localhost:8000"

response: Response = post(
    f"{BASE_URL}/upload-text",
    json={
        "dialog": [
            {"speaker": "User1", "content": "Привіт! Як справи?"},
            {"speaker": "User2", "content": "Все добре, дякую."},
            {"speaker": "User1", "content": "Підемо кави вип'ємо?"},
            {"speaker": "User2", "content": "Звичайно! Де зустрінемось?"},
            {"speaker": "User1", "content": "Біля кав'ярні на центральній."},
        ],
        "speaker": "User1",
    },
)
print(response.content)
assert response.status_code == 200, "Test #1 Failed"
print("Test #1 Passed\n")


response: Response = post(f"{BASE_URL}/train", json={"speaker": "User1"})
print(response.content)
assert response.status_code == 200, "Test #2 Failed"
print("Test #2 Passed\n")
