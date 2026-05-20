import requests


class LLMTopicNamer:

    def __init__(self, model="phi3"):
        self.model = model
        self.url = "http://localhost:11434/api/generate"

    def name_topic(self, words):

        word_list = ", ".join(words)

        prompt = f"""
        These words belong to the same vocabulary topic.

        Words:
        {word_list}

        Give a short 2-4 word topic name.
        Return ONLY the topic name.
        """

        response = requests.post(
            self.url,
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        return response.json()["response"].strip()