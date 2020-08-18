# Paraphraser-zh
This project allows users to paraphrase Chinese(ZH) and English(EN) sentences through an API.

</br>

## Model
This paraphraser is powered by a round-trip translation. ZH-EN and EN-ZH machine translation models are trained separately using fairseq, with reference to [fairseq's back translation example](https://github.com/pytorch/fairseq/tree/master/examples/backtranslation), with [WMT English-Chinese(EN-ZH) translation dataset](http://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/). The models are then used to paraphrase a sentence by translating it to an intermediate language and back.

Download models [here](https://drive.google.com/drive/folders/1KWVRWsB9C72213eRbrVR_o97Oo0s2evh).

</br>

## Requirements
- python 3.x

- fairseq 0.9.0

...

</br>

## Quickstart
1. Download all the files and models.
2. Place the model files into the *model* folder.
3. Run ```$ python app.py``` on a terminal.
4. Paste the following link on a browser.

http://localhost:8000/?lang=xx&text=blablabla

* Replace *localhost* with your local host (e.g. 0.0.0.0)
* Replace *xx* with the language you want to paraphrase, either **zh** or **en**.
* Replace *blablabla* with the sentence you want to paraphrase.

</br>

## Examples

Chinese:


* http:// localhost :8000/?lang=ZH&text=上個月你有沒有去小明的生日派對?

  * {"lang":"zh","original":"上個月你有沒有去小明的生日派對?","paraphrased":"上個月你去小明的生日聚會了嗎?"}


* http:// localhost :8000/?lang=zh&text=我們這些東西可以在必要時扔掉。

  * {"lang":"zh","original":"我們這些東西可以在必要時扔掉。","paraphrased":"我們可以在必要時把這些東西扔掉。"}


English:


* http:// localhost :8000/?lang=EN&text=When I learned that there was a gift for each child, I was delighted.

  * {"lang":"en","original":"When I learned that there was a gift for each child, I was delighted.","paraphrased":"i was very happy when i learned that every child had a gift."}


* http:// localhost :8000/?lang=en&text=stuffy nose and elevated temperature are signs you may have the flu.

  * {"lang":"en","original":"stuffy nose and elevated temperature are signs you may have the flu.","paraphrased":"a stuffy nose and high temperature are signs that you may be infected with flu."}


Other languages (not supported):


* http:// localhost :8000/?lang=fr&text=Le français est une langue indo-européenne de la famille des langues romanes.

  * {"Error":"Wrong language input."}
