# $ python app.py
# $ curl localhost:8000 -i
# http://localhost:8000/?lang=ZH&text=xxx

### ***chinese result is unicode, need python to decrypt

from sanic import Sanic
from sanic.response import json
import paraphraser

app = Sanic(__name__)

args_ze, task_ze, max_positions_ze, use_cuda_ze, generator_ze, models_ze, tgt_dict_ze, src_dict_ze, align_dict_ze,args_ez, task_ez, max_positions_ez, use_cuda_ez, generator_ez, models_ez, tgt_dict_ez, src_dict_ez, align_dict_ez = paraphraser.play_setup()

@app.route('/')
async def test(request):

    lang = request.args['lang'][0]
    sent = request.args['text'][0]
    
    lang = lang.lower()
    if (lang != 'en' and lang != 'zh'):
        return json({"Error": "Wrong language input."})

    result = paraphraser.play(lang,sent,args_ze, task_ze, max_positions_ze, use_cuda_ze, generator_ze, models_ze, tgt_dict_ze, src_dict_ze, align_dict_ze,args_ez, task_ez, max_positions_ez, use_cuda_ez, generator_ez, models_ez, tgt_dict_ez, src_dict_ez, align_dict_ez)
    
    return json({'lang':lang, 'original': sent, 'paraphrased': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)