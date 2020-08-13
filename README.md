# Paraphraser-zh
This project allows users to paraphrase Chinese(ZH) and English(EN) sentences throuhgh an API.

## Model
This paraphraser is powered by a round-trip translation. ZH-EN and EN-ZH machine translation models are trained separately using fairseq, with reference to [fairseq's back translation example](https://github.com/pytorch/fairseq/tree/master/examples/backtranslation), with [WMT English-Chinese(EN-ZH) translation dataset](http://data.statmt.org/wmt18/translation-task/preprocessed/zh-en/). The models are then used to paraphrase a sentence by translating it to an intermediate language and back.

## Requirements


## Quickstart
```
$ python app.py
```

http://localhost:8000/?lang=zh&text=blablabla


## Examples

{'lang':lang, 'original': sent, 'paraphrased': result}
