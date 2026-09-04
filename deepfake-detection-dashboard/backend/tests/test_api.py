import io
import unittest

from app import create_app


class StubBundle:
    metadata = {"model_version": "test"}


class StubService:
    bundle = StubBundle()

    def health(self):
        return {"ready": True, "device": "cpu"}

    def analyse_video(self, video_path, analysis_id, filename, include_gradcam):
        return {
            "analysis_id": analysis_id,
            "source": {"filename": filename},
            "status": "unable_to_assess",
        }


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app(
            {
                "TESTING": True,
                "LOAD_MODELS": False,
                "MAX_CONTENT_LENGTH": 1024 * 1024,
            },
            analysis_service=StubService(),
        )
        self.client = self.app.test_client()

    def test_health(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["ready"])

    def test_missing_file(self):
        response = self.client.post("/api/analyze", data={})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"]["code"], "missing_file")

    def test_extension_validation(self):
        response = self.client.post(
            "/api/analyze",
            data={"file": (io.BytesIO(b"not video"), "example.txt")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.get_json()["error"]["code"], "unsupported_file_type"
        )

    def test_uploaded_file_is_removed(self):
        response = self.client.post(
            "/api/analyze",
            data={"file": (io.BytesIO(b"placeholder"), "example.mp4")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["status"], "unable_to_assess")


if __name__ == "__main__":
    unittest.main()
