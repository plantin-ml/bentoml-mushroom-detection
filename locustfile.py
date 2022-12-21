from locust import HttpUser, task, between

with open("tests/data/0d65e0bc-28a0-41cf-8bce-e2566509f20c.jpg", "rb") as f:
    test_image_bytes = f.read()


class PyTorchLoadTestUser(HttpUser):

    wait_time = between(0.3, 1.7)

    @task
    def predict_image(self):
        files = {"upload_files": ("test.jpg", test_image_bytes, "image/png")}
        self.client.post("/detect_by_file", files=files, auth=('plantin', 'k26vKCddex'))


# locust -H http://disease-detection.models.ml.myplantin.com   --web-port 8080