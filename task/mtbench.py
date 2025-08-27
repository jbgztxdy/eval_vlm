import json
import os
import time
import shortuuid
from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_judge_prompts,
    check_data,
    play_a_match_pair,
    play_a_match_single,
    get_model_list,
    Judge,
    MatchPair,
    MatchSingle,
    NEED_REF_CATS,
)
from FastChat.fastchat.llm_judge.gen_judgment import make_judge_single, make_match_single
from .base import BaseTask
import copy
from main import MODELNAME
from fastchat.conversation import Conversation, get_conv_template
from fastchat.llm_judge.common import load_questions, temperature_config
import random


class MTBench(BaseTask):
    
    max_new_tokens = 1024
    
    def load_text_dataset(self):
        question_file = "/data/zhangyichi/yangjingwen/Eval/FastChat/fastchat/llm_judge/data/mt_bench/question.jsonl"
        self.questions = load_questions(question_file)
        random.shuffle(self.questions)
        self.text_dataset = []
        conv = None
        if  "qwen" in MODELNAME.lower():
            conv = get_conv_template("qwen-7b-chat")
        for question in self.questions:
            if conv is None:
                conv = get_conv_template("one_shot")
            for i in range(len(question["turns"])):
                qs = question["turns"][i]
                conv.append_message(conv.roles[0], qs)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                self.text_dataset.append(prompt)
        

        
    def evaluate(self, **kwargs):
        outputs = kwargs.get("outputs")
        # Dump answers
        answer_file = f"/data/zhangyichi/yangjingwen/Eval/FastChat/fastchat/llm_judge/data/mt_bench/model_answer/{MODELNAME}.jsonl"
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        choices = []
        for i, question in enumerate(self.questions):
            turns = []
            turns.append(outputs[i])
            choices.append({"index": 0, "turns": turns})
            with open(os.path.expanduser(answer_file), "a") as fout:
                ans_json = {
                    "question_id": question["question_id"],
                    "answer_id": shortuuid.uuid(),
                    "model_id": MODELNAME,
                    "choices": choices,
                    "tstamp": time.time(),
                }
                fout.write(json.dumps(ans_json) + "\n")
        judge_model = "gpt-4"
        judge_prompts = load_judge_prompts("/data/zhangyichi/yangjingwen/Eval/FastChat/fastchat/llm_judge/data/judge_prompts.jsonl")
        judges = make_judge_single(judge_model, judge_prompts)
        play_a_match_func = play_a_match_single
        # output_file = (
        #     f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
        # )
        make_match_func = make_match_single
        baseline_model = None
        matches = []
        matches += make_match_func(
            question_default, models, model_answers, judges["default"], baseline_model
        )
        matches += make_match_func(
            question_math,
            models,
            model_answers,
            judges["math"],
            baseline_model,
            ref_answers,
        )
        matches += make_match_func(
            question_default,
            models,
            model_answers,
            judges["default-mt"],
            baseline_model,
            multi_turn=True,
        )
        matches += make_match_func(
            question_math,
            models,
            model_answers,
            judges["math-mt"],
            baseline_model,
            ref_answers,
            multi_turn=True,
        )