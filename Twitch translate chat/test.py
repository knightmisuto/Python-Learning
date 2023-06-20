from googletrans import Translator

translator = Translator()

text = "세계수잎 2연속나옴"
translation = translator.translate(text, dest="en", src='ko')
print(translation.text)