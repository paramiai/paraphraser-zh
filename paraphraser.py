#chinese and english paraphraser, based on zh2en and en2zh translation

#imports

from collections import namedtuple
import fileinput
import logging
import math
import sys
import os

import torch

from fairseq import checkpoint_utils, distributed_utils, options, tasks, utils
from fairseq.data import encoders

from cantokenizer import CanTokenizer
from os.path import join as join_path

import re, string





#defines

logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
    )
logger = logging.getLogger('fairseq_cli.interactive')


Batch = namedtuple('Batch', 'ids src_tokens src_lengths')
Translation = namedtuple('Translation', 'src_str hypos pos_scores alignments')

def buffered_read(input, buffer_size):
    buffer = []
    with fileinput.input(files=[input], openhook=fileinput.hook_encoded("utf-8")) as h:
        for src_str in h:
            buffer.append(src_str.strip())
            if len(buffer) >= buffer_size:
                yield buffer
                buffer = []

    if len(buffer) > 0:
        yield buffer


def make_batches(lines, args, task, max_positions, encode_fn):
    tokens = [
        task.source_dictionary.encode_line(
            encode_fn(src_str), add_if_not_exist=False
        ).long()
        for src_str in lines
    ]
    lengths = [t.numel() for t in tokens]
    itr = task.get_batch_iterator(
        dataset=task.build_dataset_for_inference(tokens, lengths),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=max_positions,
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test
    ).next_epoch_itr(shuffle=False)
    for batch in itr:
        yield Batch(
            ids=batch['id'],
            src_tokens=batch['net_input']['src_tokens'], src_lengths=batch['net_input']['src_lengths'],
        )

tokenizer = CanTokenizer('vocab.txt',
    unk_token = "[UNK]",
    sep_token = "[SEP]",
    cls_token = "[CLS]",
    nl_token = "[NL]",
    pad_token = "[PAD]",
    mask_token = "[MASK]",)

def encode_fn(x):
    return ' '.join(tokenizer.encode(x, add_special_tokens=False).tokens)

def decode_fn_en2zh(x):
    x = re.sub(r' ##|\[(?:UNK|CLS|SEP|MASK)\]( )?', r'\1', x)
    x = re.sub(r'([\u4E00-\u9FFF]) ?', r'\1', x)
    x = re.sub(' ', '', x)
    return x

def decode_fn_zh2en(x):
    x = re.sub(r' ##|\[(?:UNK|CLS|SEP|MASK)\]( )?', r'\1', x)
    x = re.sub(r'([\u4E00-\u9FFF]) ?', r'\1', x)
    
    short = ['s','t','m','re','ve','d','ll']
    for s in short:
        x = re.sub("' " + s, "'" + s, x)
    
    count = 0
    char = list(x)
    for c in char:
        if (c in string.punctuation and count != 0):
            char[count-1] = ''
        count += 1
    x = ''.join(char)
    
    return x




def setup(source_lang,target_lang):
    sys.argv = sys.argv[:1]
    sys.argv.append('--path')
    sys.argv.append('model/checkpoints_' + source_lang + '_' + target_lang +'.pt')
    sys.argv.append('model/')
    sys.argv.append('--beam')
    sys.argv.append('5')
    sys.argv.append('--source-lang')
    sys.argv.append(source_lang)
    sys.argv.append('--target-lang')
    sys.argv.append(target_lang)
    sys.argv.append('--tokenizer')
    sys.argv.append('space')
    sys.argv.append('--bpe')
    sys.argv.append('bert')
    sys.argv.append('--bpe-vocab-file')
    sys.argv.append('model/' + '/dict.' + source_lang + '.txt')
#     sys.argv.append('--no-repeat-ngram-size')
#     sys.argv.append('2')
    sys.argv
    
    parser = options.get_interactive_generation_parser()
    args = options.parse_args_and_arch(parser)
    
    utils.import_user_module(args)

    if args.buffer_size < 1:
        args.buffer_size = 1
    if args.max_tokens is None and args.max_sentences is None:
        args.max_sentences = 1

    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert not args.max_sentences or args.max_sentences <= args.buffer_size, \
        '--max-sentences/--batch-size cannot be larger than --buffer-size'

    #logger.info(args) #print many info

    use_cuda = torch.cuda.is_available() and not args.cpu
    
    # Setup task, e.g., translation
    task = tasks.setup_task(args)

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        args.path.split(os.pathsep),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Set dictionaries
    src_dict = task.source_dictionary
    tgt_dict = task.target_dictionary

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Initialize generator
    generator = task.build_generator(models, args)
    
    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    max_positions = utils.resolve_max_positions(
        task.max_positions(),
        *[model.max_positions() for model in models]
    )

    if args.buffer_size > 1:
        logger.info('Sentence buffer size: %s', args.buffer_size)
    
    return args, task, max_positions, use_cuda, generator, models, tgt_dict, src_dict, align_dict




def interact(sent,args, task, max_positions, use_cuda, generator, models, tgt_dict, src_dict, align_dict, decode_fn):
    inputs = []
    inputs.append(sent)
    results = []
    
    start_id = 0
    for orig, batch in zip(inputs, make_batches(inputs, args, task, max_positions, encode_fn)):
        src_tokens = batch.src_tokens
        src_lengths = batch.src_lengths
        if use_cuda:
            src_tokens = src_tokens.cuda()
            src_lengths = src_lengths.cuda()

        sample = {
            'net_input': {
                'src_tokens': src_tokens,
                'src_lengths': src_lengths,
            },
        }
        translations = task.inference_step(generator, models, sample)
        for i, (id, hypos) in enumerate(zip(batch.ids.tolist(), translations)):
            src_tokens_i = utils.strip_pad(src_tokens[i], tgt_dict.pad())
            results.append((start_id + id, src_tokens_i, hypos, orig))

    # sort output to match input order
    for id, src_tokens, hypos, orig in sorted(results, key=lambda x: x[0]):
        #print('O-{}\t{}'.format(id, orig))
        if src_dict is not None:
            src_str = src_dict.string(src_tokens, args.remove_bpe)
            #print('S-{}\t{}'.format(id, src_str))

        # Process top predictions
        for hypo in hypos[:min(len(hypos), args.nbest)]:
            hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                hypo_tokens=hypo['tokens'].int().cpu(),
                src_str=src_str,
                alignment=hypo['alignment'],
                align_dict=align_dict,
                tgt_dict=tgt_dict,
                remove_bpe=args.remove_bpe,
            )
            
            # print("> before decoding: " + hypo_str)
            detok_hypo_str = decode_fn(hypo_str)
            score = hypo['score'] / math.log(2)  # convert to base 2
            # original hypothesis (after tokenization and BPE)
            #print('H-{}\t{}\t{}'.format(id, score, hypo_str))
            # detokenized hypothesis
            #print('D-{}\t{}\t{}'.format(id, score, detok_hypo_str))
            if False:
                print('P-{}\t{}'.format(
                    id,
                    ' '.join(map(
                        lambda x: '{:.4f}'.format(x),
                        # convert from base e to base 2
                        hypo['positional_scores'].div_(math.log(2)).tolist(),
                    ))
                ))
            if args.print_alignment:
                alignment_str = " ".join(["{}-{}".format(src, tgt) for src, tgt in alignment])
                print('A-{}\t{}'.format(
                    id,
                    alignment_str
                ))
                
            return detok_hypo_str

        
        
        
        
def play_setup():
    args_ze, task_ze, max_positions_ze, use_cuda_ze, generator_ze, models_ze, tgt_dict_ze, src_dict_ze, align_dict_ze = setup('zh','en')
    args_ez, task_ez, max_positions_ez, use_cuda_ez, generator_ez, models_ez, tgt_dict_ez, src_dict_ez, align_dict_ez = setup('en','zh') 
    return args_ze, task_ze, max_positions_ze, use_cuda_ze, generator_ze, models_ze, tgt_dict_ze, src_dict_ze, align_dict_ze,args_ez, task_ez, max_positions_ez, use_cuda_ez, generator_ez, models_ez, tgt_dict_ez, src_dict_ez, align_dict_ez
 
    
    
    

def play(choice,sent,args_ze, task_ze, max_positions_ze, use_cuda_ze, generator_ze, models_ze, tgt_dict_ze, src_dict_ze, align_dict_ze,args_ez, task_ez, max_positions_ez, use_cuda_ez, generator_ez, models_ez, tgt_dict_ez, src_dict_ez, align_dict_ez):
    
    # zh --> en --> zh 
    if (choice == 'zh'):     
        
        temp = interact(sent,args_ze, task_ze, max_positions_ze, use_cuda_ze, generator_ze, models_ze, tgt_dict_ze, src_dict_ze, align_dict_ze, decode_fn_zh2en)
        
        paraphrased = interact(temp,args_ez, task_ez, max_positions_ez, use_cuda_ez, generator_ez, models_ez, tgt_dict_ez, src_dict_ez, align_dict_ez, decode_fn_en2zh)
    
    #en --> zh --> en
    elif (choice == 'en'):
        
        temp = interact(sent,args_ez, task_ez, max_positions_ez, use_cuda_ez, generator_ez, models_ez, tgt_dict_ez, src_dict_ez, align_dict_ez, decode_fn_en2zh)
        
        paraphrased = interact(temp,args_ze, task_ze, max_positions_ze, use_cuda_ze, generator_ze, models_ze, tgt_dict_ze, src_dict_ze, align_dict_ze, decode_fn_zh2en)
        
    else:
        return 'error'
    
    return paraphrased        
