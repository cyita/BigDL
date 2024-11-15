# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import argparse
import torch
import time

from transformers import AutoTokenizer, TextStreamer

sentence = open("longbench_2k.txt", 'r', encoding='utf-8').read()
# in_len_list = [1024, 2048, 3072, 4096]
in_len_list = [1024]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GLM4 benchmark scripts')
    parser.add_argument('--repo-id-or-model-path', type=str, default=r"D:\llm-models\glm4-mini-chat-v020",                    
                        help='The huggingface repo id for the Qwen2 model to be downloaded'
                        ', or the path to the huggingface checkpoint folder')
    parser.add_argument('--n-predict', type=int, default=128,
                        help='Max tokens to predict')
    
    args = parser.parse_args()
    model_path = args.repo_id_or_model_path
    n_predict = args.n_predict

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path,
                                              trust_remote_code=True)

    from ipex_llm.transformers import AutoModelForCausalLM
    # Load model in 4 bit,
    # which convert the relevant layers in the model into INT4 format
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 load_in_4bit=True,
                                                 optimize_model=True,
                                                 trust_remote_code=True,
                                                 use_cache=True).eval()
    model = model.half().to("xpu")

    # here the prompt tuning refers to https://huggingface.co/THUDM/glm-4-9b-chat#%E4%BD%BF%E7%94%A8-transformers-%E5%90%8E%E7%AB%AF%E8%BF%9B%E8%A1%8C%E6%8E%A8%E7%90%86
    full_input_str = tokenizer.apply_chat_template([{"role": "user", "content": sentence}],
                                                   add_generation_prompt=True,
                                                   tokenize=False,
                                                   add_special_tokens=False)

    with torch.inference_mode():
        for in_len in in_len_list:
            print('='*30, f'In-len {in_len}', '='*30)
            full_inputs = tokenizer([full_input_str], return_tensors="pt").to("xpu")

            half_idx = in_len // 2
            input_ids = \
                torch.cat((full_inputs.input_ids[:, :half_idx], full_inputs.input_ids[:, -(in_len-half_idx):]), dim=1)
            
            print("---Start warmup---")
            # warmup for each input length
            output_ids = model.generate(
                input_ids,
                max_new_tokens=args.n_predict,
                do_sample=False,
            )
            print("---Warmup finish---")

            streamer = TextStreamer(tokenizer, skip_prompt=True)
            first_token_list = []
            rest_token_list = []

            for i in range(3):
                st = time.perf_counter()
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=args.n_predict, 
                    streamer=streamer, 
                    do_sample=False,
                )
                torch.xpu.synchronize()
                end = time.perf_counter()

                output_ids = output_ids.cpu()
                output_ids = [
                    output_token_ids[len(input_token_ids):] for input_token_ids, output_token_ids in zip(input_ids, output_ids)
                ]

                output_str = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]

                first_token_latency = model.first_token_time
                rest_token_latency = (end - st - model.first_token_time)/(model.n_token_generated - 1)

                print('-'*30, f'Trial {i}', '-'*30)
                print(f'    Input tokens number: {len(input_ids[0])}')
                print(f'    Output tokens number: {len(output_ids[0])} or {model.n_token_generated}')
                print(f"    1st token latency: {first_token_latency:.6f} ms")
                print(f'    Average latency: {rest_token_latency:.6f} ms/t')
                print('-'*80)

                first_token_list.append(first_token_latency)
                rest_token_list.append(rest_token_latency)

                print('-'*20, 'perf mode info', '-'*20)
                if getattr(model, 'n_drafted', None) is not None:
                    draft_len = model.n_drafted/len(model.draft_num)
                    accept_rate = model.n_matched/model.n_drafted
                    print(f"Draft Number: {model.draft_num}")
                    print(f"Accept Number: {model.accept_num}")
                    print(f"Draft len: {draft_len}")
                    print(f"Accept len: {model.n_matched/len(model.accept_num)}")
                    print(f"Accept rate: {accept_rate*100}%")
                    print(f"1st token: {first_token_latency} ms")
                    print(f"2+ token: {rest_token_latency} ms")
                print('-'*50)

            print('-'*30, 'Summary', '-'*30)
            print(f'    Input tokens number: {len(input_ids[0])}')
            print(f'    Output tokens number: {len(output_ids[0])}')
            print(f"    1st token latency list: {first_token_list} ms")
            print(f"    1st token latency avg.: {(sum(first_token_list)/len(first_token_list)):.6f} ms")
            print(f"    Rest token latency list: {rest_token_list} ms")
            print(f"    Rest token latency avg.: {(sum(rest_token_list)/len(rest_token_list)):.6f} ms")
            print('-'*30, 'Output', '-'*30)
            print(output_str)
            print('='*80)
