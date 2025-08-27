from vllm import LLM
import PIL


llm = LLM(model="llava-hf/llava-1.5-7b-hf")

# Refer to the HuggingFace repo for the correct format to use
prompt = "USER: <image>\nWho is she ? \nASSISTANT:"

# Load the image using PIL.Image
image = PIL.Image.open('emma.png')

sampling_params = llm.get_default_sampling_params()
print(sampling_params)
sampling_params.max_tokens = 1024


# Single prompt inference
outputs = llm.generate({
    "prompt": prompt,
    "multi_modal_data": {"image": image},
},sampling_params)

for o in outputs:
    generated_text = o.outputs[0].text
    print(generated_text)


# import base64

# import requests
# from openai import OpenAI

# from vllm.utils import FlexibleArgumentParser

# # Modify OpenAI's API key and API base to use vLLM's API server.
# openai_api_key = "EMPTY"
# openai_api_base = "http://127.0.0.1:8002/v1"

# client = OpenAI(
#     # defaults to os.environ.get("OPENAI_API_KEY")
#     api_key=openai_api_key,
#     base_url=openai_api_base,
# )

# models = client.models.list()
# model = models.data[0].id

# # Text-only inference
# def run_text_only() -> None:
#     chat_completion = client.chat.completions.create(
#         messages=[{
#             "role": "user",
#             "content": "Count to 10."
#         }],
#         model=model,
#         max_completion_tokens=1024,
#     )

#     result = chat_completion.choices[0].message.content
#     print("Chat completion output:", result)
    
    
# # Text-only inference
# def run_text_noise_only() -> None:
#     chat_completion = client.chat.completions.create(
#         messages=[{
#             "role": "user",
#             "content": "asdfneknrl j2oi3snadf, "
#             },{
#             "role": "user",
#             "content": "Count to 10."
#         }],
#         model=model,
#         max_completion_tokens=1024,
#     )

#     result = chat_completion.choices[0].message.content
#     print("Chat completion output:", result)
    
    


# # Single-image input inference
# def run_single_image() -> None:

#     ## Use image url in the payload
#     image_url = "file:///data/zhangyichi/Modality-Bridging/emma.png"
#     chat_completion_from_url = client.chat.completions.create(
#         messages=[{
#             "role":
#             "user",
#             "content": [
#                 {
#                     "type": "text",
#                     "text": "Count to 10."
#                 },
#                 {
#                     "type": "image_url",
#                     "image_url": {
#                         "url": image_url
#                     },
#                 },
#             ],
#         }],
#         model=model,
#         max_completion_tokens=1024,
#     )

#     result = chat_completion_from_url.choices[0].message.content
#     print("Chat completion output from image url:", result)


# run_text_only()
# run_text_noise_only()
# run_single_image()