from app import app

# This function tests if home page is accessible
def test_home():
    response=app.test_client().get("/")

    assert response.status_code==200
    assert response.data== b"Hello World! VERSION 4"