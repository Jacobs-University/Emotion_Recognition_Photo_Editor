import speech_recognition as sr

class AudioUtils:
    """
        author: Parth Shukla

        This class will provide certain audio functionality, like:
            - doing speech to text
    """

    @staticmethod
    def record_caption():
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Say your caption!")
            audio = r.listen(source)
            text = (r.recognize_google(audio)).lower()
            print(text)
            return text