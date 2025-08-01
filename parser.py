from models import DialogMessage
import json
import os


class ChatParser:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, dialog: list[DialogMessage], speaker: str) -> str:
        """
        Parse dialog data and save messages from the specified speaker

        Args:
            dialog: list of messages
            speaker: Speaker to filter messages for

        Returns:
            Path to the saved file
        """
        # Filter messages from the specified speaker
        speaker_messages: list[dict[str, str]] = [
            message.model_dump()
            for message in dialog
            if message.speaker == speaker
        ]

        # Save messages
        path: str = os.path.join(self.output_dir, f"{speaker}_messages.json")
        with open(path, "w", encoding="utf-8") as file:
            json.dump({"messages": speaker_messages}, file)

        return path

    def load(self, speaker: str) -> list[DialogMessage]:
        """
        Load messages for a specific speaker from saved file

        Args:
            speaker: Speaker to load messages for

        Returns:
            list of messages
        """
        # Load messages
        path: str = os.path.join(self.output_dir, f"{speaker}_messages.json")
        with open(path, "r", encoding="utf-8") as file:
            data: list[DialogMessage] = [
                DialogMessage(
                    speaker=message["speaker"], content=message["content"]
                )
                for message in json.load(file)["messages"]
            ]

        return data
