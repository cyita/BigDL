#
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


import os
import torch
import time
import argparse
from ipex_llm.transformers.npu_model import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers.utils import logging

logger = logging.get_logger(__name__)

long_prompt = """You are given a report by a government agency. Write a one-page summary of the report.

Report:
Justice Management Division (JMD) JMD provides the Federal Bureau of Prisons senior management with guidance as it relates to Department of Justice (DOJ) policy for all matters pertaining to organization, management, and administration, including the use of human capital flexibilities such as retention incentives. BOP is responsible for incarcerating all federal offenders sentenced to prison. To carry out its mission, BOP, under the oversight of DOJ’s JMD, manages the human resource operations of its institutions, including the use of retention incentives. BOP administers, monitors, and oversees retention incentives through its Central Office, regional offices, and institutions. Central Office. The Central Office serves as BOP’s headquarters and provides oversight of BOP operations and program areas. Within the Central Office is BOP’s Human Resource Management Division (HRMD) which is responsible for developing, implementing and administering human resource policies and programs, including the use of retention incentives that meet OPM and DOJ requirements. In addition, the Central Office’s Program Review Division (PRD) is responsible for assessing BOP programs, including human resources, to ensure that they are managed and operated effectively. Regional offices. BOP has six regional offices that cover the Mid- Atlantic, North Central, Northeast, South Central, Southeast, and Western regions of the United States. These offices, each led by a regional director, oversee the operations of the 122 federal institutions within their respective geographic regions of the country. According to BOP officials, regional office staff also provide local level oversight of institutions’ human capital programs, such as retention incentives, among other things. Institutions. BOP institutions are managed by a warden and other officials, including an executive assistant and associate warden who generally provide overall direction and, in part, administer the institution’s human capital policies, including policies on retention incentives. Correctional services staff represent the largest segment of each institution’s workforce and are responsible for the correctional treatment, custody, and supervision of inmates. Non-correctional services staff include, among others, those employees assigned to non-correctional services management, facility operations, and the health services unit. Workers in health services and psychology services are responsible for providing inmates with medical, dental, and mental health services and include, for example, received retention incentives from fiscal years 2014 through 2016. Each application file was reviewed by two GAO analysts who each assessed the extent to which each application contained the appropriate justification, approval signatures, and other documentation such as an application checklist and whether the application was an initial or continuation application. To determine the extent to which BOP plans for and evaluates the use of retention incentives, we interviewed BOP officials regarding their experiences with retention incentives, how they use retention incentives to strategically manage their workforce needs, how the agency evaluates the effectiveness of retention incentives, and how retention incentives contribute to BOP’s broader human capital goals. We then compared these efforts to our work on strategic human capital planning, specifically in terms of planning for and evaluating the use of human capital flexibilities. Additionally, we interviewed the warden and human capital officers at four BOP institutions mentioned above to obtain illustrative examples of how workforce planning occurs at these institutions. We also reviewed the DOJ’s Office of Inspector General Report 16-02 “Review of the Federal Bureau of Prisons’ Medical Staffing Challenges” (March 2016) and our past work to better understand the challenges that BOP faces in retaining medical professionals and other staff. Table 2 provides the Bureau of Prisons’ (BOP) fiscal year 2016 retention incentive expenditures by various occupations and groups of occupations, such as medical professionals, correctional officers, and other occupations. A range of occupations are reflected in the table primarily as a result of four California institutions—United States Penitentiary (USP) Atwater, Federal Correctional Institution (FCI) Herlong, FCI Mendota, and Federal Correctional Complex Victorville—providing retention incentives to all employees at General Schedule grades level 12 and below and those in the Federal Wage System. In addition to the contact named above, Dawn Locke (Assistant Director) and Meghan Squires (Analyst-in-Charge) managed the work. Also, David Alexander, Renee Caputo, Willie Commons III, Jamarla Edwards, Robert Goldenkoff, Chelsa Gurkin, Eric Hauswirth, Janice Latimer, Lerone Reid, Rachel Stoiko, and Adam Vogt made significant contributions to this report.

"""

short_prompt = "What is AI?"

def get_prompt(message: str, chat_history: list[tuple[str, str]],
               system_prompt: str) -> str:
    texts = [f'<s>[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n']
    # The first user input is _not_ stripped
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(f'{user_input} [/INST] {response.strip()} </s><s>[INST] ')
    message = message.strip() if do_strip else message
    texts.append(f'{message} [/INST]')
    return ''.join(texts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict Tokens using `generate()` API for npu model"
    )
    parser.add_argument(
        "--repo-id-or-model-path",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="The huggingface repo id for the Llama2 model to be downloaded"
        ", or the path to the huggingface checkpoint folder",
    )
    parser.add_argument("--lowbit-path", type=str,
        default="",
        help="The path to the lowbit model folder, leave blank if you do not want to save. \
            If path not exists, lowbit model will be saved there. \
            Else, lowbit model will be loaded.",
    )
    parser.add_argument('--prompt', type=str, default=long_prompt,
                        help='Prompt to infer')
    parser.add_argument("--n-predict", type=int, default=32, help="Max tokens to predict")
    parser.add_argument("--max-context-len", type=int, default=1024)
    parser.add_argument("--max-prompt-len", type=int, default=512)
    parser.add_argument("--quantization_group_size", type=int, default=0)
    parser.add_argument("--disable-transpose-value-cache", action="store_true", default=False)

    args = parser.parse_args()
    model_path = args.repo_id_or_model_path

    if not args.lowbit_path or not os.path.exists(args.lowbit_path):
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     optimize_model=True,
                                                     pipeline=True,
                                                     max_context_len=args.max_context_len,
                                                     max_prompt_len=args.max_prompt_len,
                                                     quantization_group_size=args.quantization_group_size,
                                                     torch_dtype=torch.float16,
                                                     attn_implementation="eager",
                                                     transpose_value_cache=not args.disable_transpose_value_cache)
    else:
        model = AutoModelForCausalLM.load_low_bit(
            args.lowbit_path,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            max_context_len=args.max_context_len,
            max_prompt_len=args.max_prompt_len,
            pipeline=True,
            transpose_value_cache=not args.disable_transpose_value_cache,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    if args.lowbit_path and not os.path.exists(args.lowbit_path):
        model.save_low_bit(args.lowbit_path)

    DEFAULT_SYSTEM_PROMPT = """\
    """

    print("-" * 80)
    print("done")
    with torch.inference_mode():
        print("finish to load")
        for i in range(5):
            prompt = get_prompt(args.prompt, [], system_prompt=DEFAULT_SYSTEM_PROMPT)
            _input_ids = tokenizer.encode(prompt, return_tensors="pt")[:, :args.max_prompt_len]
            print("input length:", len(_input_ids[0]))
            st = time.time()
            output = model.generate(
                _input_ids, max_new_tokens=args.n_predict, do_print=True
            )
            end = time.time()
            print(f"Inference time: {end-st} s")
            input_str = tokenizer.decode(_input_ids[0], skip_special_tokens=False)
            print("-" * 20, "Input", "-" * 20)
            print(input_str)
            output_str = tokenizer.decode(output[0, len(_input_ids[0]):], skip_special_tokens=False)
            print("-" * 20, "Output", "-" * 20)
            print(output_str)

    print("-" * 80)
    print("done")
    print("success shut down")
